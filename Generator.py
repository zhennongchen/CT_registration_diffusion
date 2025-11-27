# generator

import sys 
sys.path.append('/host/d/Github/')
import os
import torch
import numpy as np
import nibabel as nb
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import CT_registration_diffusion.functions_collection as ff
import CT_registration_diffusion.Data_processing as Data_processing

# define augmentation functions here if needed
# random functionf
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 1):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if len(i.shape) == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree
        else:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    
    if len(i.shape) == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate]), x_translate,y_translate
    else:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate



class Dataset_4DCT(Dataset):
    def __init__(
        self,
        image_folder_list,
        image_size = [224,224,96], # target image size after center-crop

        num_of_pairs_each_case = 1, # number of image pairs to be sampled from each 4DCT case
        preset_paired_tf = None, # preset paired time frames if needed, e.g., [[0,3],[1,2]], otherwise randomly pick two time frames
        only_use_tf0_as_moving = None, # if set True, only use time frame 0 as moving image, otherwise randomly select moving time frame

        cutoff_range = [-200,250], # default cutoff range for CT images
        normalize_factor = 'equatoin',
        shuffle = False,

        augment = True, # whether to do data augmentation
        augment_frequency = 0.5, # frequency of augmentation
      
    ):
        super().__init__()
        self.image_folder_list = image_folder_list
        self.image_size = image_size

        self.num_of_pairs_each_case = num_of_pairs_each_case
        self.preset_paired_tf = preset_paired_tf
        if self.preset_paired_tf is not None:
            assert self.num_of_pairs_each_case == len(self.preset_paired_tf)
        self.only_use_tf0_as_moving = only_use_tf0_as_moving; assert self.only_use_tf0_as_moving in [True, False]
       
        self.background_cutoff = cutoff_range[0]
        self.maximum_cutoff = cutoff_range[1]
        self.normalize_factor = normalize_factor

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency

        self.num_files = len(image_folder_list)

        self.index_array = self.generate_index_array()
        # self.current_moving_file = None
        # self.current_moving_data = None
        # self.current_fixed_file = None
        # self.current_fixed_data = None
       

    def generate_index_array(self): 
        np.random.seed()
        index_array = []
        
        # loop through all files
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)
        
        for f in f_list:
            # loop through all pairs in each file 
            for p in range(self.num_of_pairs_each_case):
                index_array.append([f,p])
      
        return index_array

    def __len__(self):
       return self.num_files * self.num_of_pairs_each_case
    
    def load_data(self, file_path):
        image = nb.load(file_path).get_fdata()
        return image

  
    def __getitem__(self, index):
        file_index, pair_index = self.index_array[index]
        current_image_folder = self.image_folder_list[file_index]
        
        # randomly pick two time frames or using preset paired time frames
        timeframes = ff.find_all_target_files(['img*'], current_image_folder)
        if self.preset_paired_tf is not None:
            t1, t2 = self.preset_paired_tf[pair_index]
            # print('这里的time frame配对是预设的,不是随机选取的, pick time frames:', t1, t2)
            if self.only_use_tf0_as_moving == True:
                assert t1 == 0, 'when only_use_tf0_as_moving is set True, the preset paired time frames must have time frame 0 as moving image'
        else:
            if self.only_use_tf0_as_moving == True:
                t1 = 0
                t2 = np.random.choice([i for i in range(len(timeframes)) if i != t1])
            else:
                t1, t2 = np.random.choice(len(timeframes), size=2, replace=False)
            # print('这里的time frame配对是随机选取的, pick time frames:', t1, t2)
        moving_file = timeframes[t1]
        fixed_file = timeframes[t2]
        # print('in this folder, I pick moving file:', moving_file, ' fixed file:', fixed_file)

        # load image
        moving_image = self.load_data(moving_file)
        fixed_image = self.load_data(fixed_file)

        # augmentation for noise if needed
        if self.augment == True and (np.random.rand() < self.augment_frequency):
            # add noise, make sure the noise added to both images are the same
            standard_deviation = np.random.uniform(5,15) # standard deviation of the noise
            noise = np.random.normal(0, standard_deviation, moving_image.shape)
            moving_image = moving_image + noise
            fixed_image = fixed_image + noise

        # preprocess if needed
        # cutoff 
        # print('before cutoff, image range:', np.min(moving_image), np.max(moving_image))
        if self.background_cutoff is not None and self.maximum_cutoff is not None:
            moving_image = Data_processing.cutoff_intensity(moving_image, self.background_cutoff, self.maximum_cutoff)
            fixed_image = Data_processing.cutoff_intensity(fixed_image, self.background_cutoff, self.maximum_cutoff)
      
        # normalization to [-1,1]
        moving_image = Data_processing.normalize_image(moving_image, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False)
        fixed_image = Data_processing.normalize_image(fixed_image, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False)
        # print('after cutoff and normalization, image range:', np.min(moving_image), np.max(moving_image))
        
        # augmentation if needed
        # step 2: rotate [-10,10] degrees according to z-axis
        # step 3: translate [-10,10] pixels
        if self.augment == True and (np.random.rand() < self.augment_frequency):

            # rotate (according to z-axis), make sure rotate angle is the same for both images, use function random_rotate above
            moving_image, z_rotate_degree = random_rotate(moving_image,  order = 1)
            moving_image, x_translate, y_translate = random_translate(moving_image)
            fixed_image, _ = random_rotate(fixed_image, z_rotate_degree, order = 1)
            fixed_image, _, _ = random_translate(fixed_image, x_translate, y_translate)

        # print('after preprocessing and augmentation, image shape:', moving_image.shape)

        # make it a standard dimension for deep learning model: [channel, x,y,z]
        moving_image = np.expand_dims(moving_image, axis=0)  # add channel dimension
        fixed_image = np.expand_dims(fixed_image, axis=0)  # add channel

        return moving_image, fixed_image
    
    def on_epoch_end(self):
        self.index_array = self.generate_index_array()
    

