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

import CT_registration_diffusion.functions_collection as ff
import CT_registration_diffusion.Data_processing as Data_processing
import CT_registration_diffusion.model.model as my_model
import CT_registration_diffusion.model.spatial_transform as spatial_transform



class Predictor(object):
    def __init__(
        self,
        model,
        generator,
        batch_size,
        device = 'cuda',

    ):
        super().__init__()

        # model
        self.model = model  
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.image_size = generator.image_size
        self.batch_size = batch_size

        # dataset and dataloader
        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())    
        self.background_cutoff = self.generator.background_cutoff
        self.maximum_cutoff = self.generator.maximum_cutoff
        self.normalize_factor = self.generator.normalize_factor

        self.dl = dl
 
        # EMA:
        self.ema = EMA(model)
        self.ema.to(self.device)

    def load_model(self, trained_model_filename):

        data = torch.load(trained_model_filename, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.ema.load_state_dict(data["ema"])

    
    def predict_MVF_and_apply(self, trained_model_filename):
        self.load_model(trained_model_filename)
        self.ema.ema_model.eval()
        with torch.inference_mode():
            for batch_input in tqdm(self.dl):
                moving_image, fixed_image = batch_input
                data_moving  = moving_image.to(self.device)
                data_fixed = fixed_image.to(self.device)
                data_input = torch.cat((data_moving, data_fixed), dim=1)

                pred_MVF = self.ema.ema_model(data_input)

                pred_MVF_numpy = torch.clone(pred_MVF).detach().cpu().numpy().squeeze()

                # apply the MVF to moving image to get the warped image
                warped_moving_image = spatial_transform.warp_from_mvf(data_moving, pred_MVF)
                warped_moving_image_numpy = warped_moving_image.detach().cpu().numpy().squeeze()

                # de-normalize the warped image
                warped_moving_image_numpy = Data_processing.normalize_image(warped_moving_image_numpy, normalize_factor =self.generator.normalize_factor, image_max = self.generator.maximum_cutoff, image_min = self.generator.background_cutoff, invert = True)

                
            return pred_MVF, pred_MVF_numpy, warped_moving_image_numpy
        