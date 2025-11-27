import sys 
sys.path.append('/host/d/Github/')

import math
import copy
import os
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from skimage.measure import block_reduce

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from CT_registration_diffusion.model.attend import Attend
import CT_registration_diffusion.functions_collection as ff
import CT_registration_diffusion.Data_processing as Data_processing
import CT_registration_diffusion.model.spatial_transform as spatial_transform
import CT_registration_diffusion.model.loss as my_loss
import CT_registration_diffusion.model.model as my_model


class Trainer(object):
    def __init__(
        self,
        model,
        generator_train,
        generator_val,
        train_batch_size,
        regularization_weight,

        *,
        accum_iter = 1, # gradient accumulation steps
        train_num_steps = 1000, # total training epochs
        results_folder = None,
        train_lr = 1e-4,
        train_lr_decay_every = 100, 
        save_models_every = 1,
        validation_every = 1,
        
        ema_update_every = 10,
        ema_decay = 0.95,
        adam_betas = (0.9, 0.99),
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
         
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = model  
        # put MSE loss
        self.similarity_metric = nn.MSELoss()
        self.regularization_metric = my_loss.GradSmoothLoss() 
        self.regularization_weight = regularization_weight

        # sampling and training hyperparameters
        self.batch_size = train_batch_size
        self.accum_iter = accum_iter
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
        self.validation_every = validation_every

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
    
        ff.make_folder([self.results_folder])

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if my_model.exists(self.accelerator.scaler) else None,
        }
        
        torch.save(data, os.path.join(self.results_folder, 'model-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):

        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(trained_model_filename, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        self.scheduler.load_state_dict(data['decay_steps'])

        if my_model.exists(self.accelerator.scaler) and my_model.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self, pre_trained_model = None ,start_step = None):

        accelerator = self.accelerator
        device = accelerator.device

        # load pre-trained
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)

        if start_step is not None:
            self.step = start_step

        training_log = []
        val_loss = np.inf; val_similarity_loss = np.inf; val_regularization_loss = np.inf

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []; average_similarity_loss = []; average_regularization_loss = []
                count = 1
                # load data
                for batch in self.dl:
                    if count == 1 or count % self.accum_iter == 1 or count == len(self.dl) - 1 or count == len(self.dl):
                        self.opt.zero_grad()
         
                    # load data
                    batch_moving, batch_fixed = batch 
                    data_moving = batch_moving.to(device)
                    data_fixed = batch_fixed.to(device)

                    # concatenate moving and fixed images along the channel dimension
                    data_input = torch.cat((data_moving, data_fixed), dim=1)

                    with self.accelerator.autocast():
                        output_MVF = self.model(data_input)

                        # apply MVF to moving image to get warped moving image using warp_from_mvf in spatial_transform.py
                        warped_moving = spatial_transform.warp_from_mvf(data_moving, output_MVF)
                        
                        # loss
                        similarity_loss = self.similarity_metric(warped_moving, data_fixed)
                        regularization_loss = self.regularization_metric(output_MVF)
                        loss = similarity_loss + self.regularization_weight * regularization_loss
                        
                    # accumulate the gradient, typically used when batch size is small
                    if count % self.accum_iter == 0 or count == len(self.dl) - 1 or count == len(self.dl):
                        self.accelerator.backward(loss)
                        accelerator.wait_for_everyone()
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.opt.step()

                    count += 1
                    average_loss.append(loss.item()); average_similarity_loss.append(similarity_loss.item()); average_regularization_loss.append(regularization_loss.item())

                   
                average_loss = sum(average_loss) / len(average_loss)
                average_similarity_loss = sum(average_similarity_loss) / len(average_similarity_loss)
                average_regularization_loss = sum(average_regularization_loss) / len(average_regularization_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}, average similarity loss: {average_similarity_loss:.4f}, average regularization loss: {average_regularization_loss:.4f}')
               
                accelerator.wait_for_everyone()

                self.step += 1

                # save the model
                if self.step !=0 and my_model.divisible_by(self.step, self.save_model_every):
                   self.save(self.step)
                
                if self.step !=0 and my_model.divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()
                    
                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and my_model.divisible_by(self.step, self.validation_every):
                    print('validation at step: ', self.step)
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []; val_similarity_loss = []; val_regularization_loss = []
                        for batch in self.dl_val:
                            batch_moving, batch_fixed = batch 
                            data_moving = batch_moving.to(device)
                            data_fixed = batch_fixed.to(device)

                            # concatenate moving and fixed images along the channel dimension
                            data_input = torch.cat((data_moving, data_fixed), dim=1)

                            with self.accelerator.autocast():
                                output_MVF = self.model(data_input)

                                # apply MVF to moving image to get warped moving image using warp_from_mvf in spatial_transform.py
                                warped_moving = spatial_transform.warp_from_mvf(data_moving, output_MVF)
                                
                                # loss
                                similarity_loss = self.similarity_metric(warped_moving, data_fixed)
                                regularization_loss = self.regularization_metric(output_MVF)
                                loss = similarity_loss + self.regularization_weight * regularization_loss

                            val_loss.append(loss.item()); val_similarity_loss.append(similarity_loss.item()); val_regularization_loss.append(regularization_loss.item())
                        val_loss = sum(val_loss) / len(val_loss)
                        val_similarity_loss = sum(val_similarity_loss) / len(val_similarity_loss)
                        val_regularization_loss = sum(val_regularization_loss) / len(val_regularization_loss)
                        print(f'validation loss: {val_loss:.4f}, validation similarity loss: {val_similarity_loss:.4f}, validation regularization loss: {val_regularization_loss:.4f}')
                    self.model.train(True)

                # save the training log
                training_log.append([self.step,self.scheduler.get_last_lr()[0],average_loss,average_similarity_loss,average_regularization_loss, val_loss, val_similarity_loss, val_regularization_loss])
                df = pd.DataFrame(training_log,columns = ['iteration','learning_rate','train_loss','train_similarity_loss','train_regularization_loss','val_loss','val_similarity_loss','val_regularization_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)
                        
                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end(); self.ds_val.on_epoch_end()

                pbar.update(1)

        accelerator.print('training complete')