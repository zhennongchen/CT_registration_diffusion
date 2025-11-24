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

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def Upsample2D(dim, dim_out = None, upsample_factor = (2,2)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,1)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample2D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(dim, default(dim_out, dim), 1)
    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=(2,2, 1), stride=(2,2, 1), padding=0),
        nn.Conv3d(dim, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class RMSNorm3D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1 , 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# building block modules
    
class ConvBlock2D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None, act = 'ReLU'):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding = dilation, dilation = dilation)

        self.norm = nn.GroupNorm(groups, dim_out)  

        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('activation function not supported')
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ConvBlock3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None, act = 'ReLU'):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = dilation, dilation = dilation)
        self.norm = nn.GroupNorm(groups, dim_out) 
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('activation function not supported')

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# Attention:
class LinearAttention2D(nn.Module): # input dimension is dim, same dimension for input and output
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)  # split input into q k v evenly
        # here each q, k ,v has the dim = [b, hidden_dim, h, w]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # here each q, k ,v has the dim = [b, heads, hidden_dim, h*w]

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication
        # k*v:  [b, heads, hidden_dim, h*w] mul [b, heads, hidden_dim, h*w] -> [b, heads, hidden_dim, hidden_dim]

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # context*q: [b, heads, hidden_dim, hidden_dim] mul [b, heads, hidden_dim, h*w] -> [b, heads, hidden_dim, h*w]

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # out: [b, heads, hidden_dim, h*w] -> [b, heads*hidden_dim, h, w]
        return self.to_out(out)
    

class Attention2D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim) 
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # here each q, k ,v has the dim = [b, hidden_dim, h, w]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)
        # here each q, k ,v has the dim = [b, heads, h*w, hidden_dim]

        out = self.attend(q, k, v)
        # first q*k: [b, heads, h*w, hidden_dim] mul [b, heads, h*w, hidden_dim] -> [b, heads, h*w, h*w]   (einsum(f"b h i d, b h j d -> b h i j", q, k) * scale)
        # second *v: [b, heads, h*w, h*w] mul [b, heads, h*w, hidden_dim] -> [b, heads, h*w, hidden_dim]  (einsum(f"b h i j, b h j d -> b h i d", attn, v))

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
    
    
class LinearAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            RMSNorm3D(dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=self.heads), qkv)  # h = head, c = dim_head

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z = d)
        return self.to_out(out)


class Attention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.attend = Attend(flash = False)

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x) 

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h (x y z) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = d)
        return self.to_out(out)
    

class ResnetBlock2D(nn.Module): 
    # conv + conv + attention + residual
    def __init__(self, dim, dim_out, groups = 8, use_full_attention = None, attn_head = 4, attn_dim_head = 32, act = 'ReLU'):
        '''usee which attention: 'Full' or 'Linear'''
        super().__init__()
    
        self.block1 = ConvBlock2D(dim, dim_out, groups = groups, act = act)
        self.block2 = ConvBlock2D(dim_out, dim_out, groups = groups , act = act)
        
        if use_full_attention == True:
            self.attention = Attention2D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        elif use_full_attention == False:
            self.attention = LinearAttention2D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        else:
            self.attention = nn.Identity()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x) 

        h = self.block2(h)

        h = self.attention(h)

        return h + self.res_conv(x)
    
    
class ResnetBlock3D(nn.Module): 
    # conv + conv + attention + residual
    def __init__(self, dim, dim_out, groups = 8, use_full_attention = None, attn_head = 4, attn_dim_head = 32 , act = 'ReLU'):
        '''usee which attention: 'Full' or 'Linear'''
        super().__init__()
    
        self.block1 = ConvBlock3D(dim, dim_out, groups = groups, act = act)
        self.block2 = ConvBlock3D(dim_out, dim_out, groups = groups, act = act)
        
        if use_full_attention == True:
            self.attention = Attention3D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        elif use_full_attention == False:
            self.attention = LinearAttention3D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        else:
            self.attention = nn.Identity()

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x) 

        h = self.block2(h)

        h = self.attention(h)

        return h + self.res_conv(x)
    
class Unet(nn.Module):
    def __init__(
        self,
        problem_dimension = '2D',  # '2D' or '3D'
  
        input_channels = 2, # =1 if only moving image as input; =2 if both fixed and moving images as input
        out_channels = 3,  # =2 for 2D deformation field; =3 for 3D deformation field

        initial_dim = 16,  # initial feature dimension after first conv layer
        dim_mults = (2,4,8,16),
        groups = 4,
      
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn_paths = (None, None, None, True), # these are for downsampling and upsampling paths
        full_attn_bottleneck = None, # this is for the middle bottleneck layer
        act = 'ReLU',
    ):
        super().__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.problem_dimension = problem_dimension
        self.groups = groups

        conv_layer = nn.Conv2d if self.problem_dimension == '2D' else nn.Conv3d
        ResnetBlock = ResnetBlock2D if self.problem_dimension == '2D' else ResnetBlock3D
        Attention = Attention2D if self.problem_dimension == '2D' else Attention3D
        LinearAttention = LinearAttention2D if self.problem_dimension == '2D' else LinearAttention3D
        downsample_layer = Downsample2D if self.problem_dimension == '2D' else Downsample3D
        upsample_layer = Upsample2D if self.problem_dimension == '2D' else Upsample3D

        self.init_conv = conv_layer(self.input_channels, initial_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [initial_dim, *map(lambda m: initial_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 

        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn_paths, num_stages)
        self.full_attn_bottleneck = full_attn_bottleneck
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            # print(' in downsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out, ' layer_full_attn is: ', layer_full_attn, ' layer_attn_heads is: ', layer_attn_heads, ' layer_attn_dim_head is: ', layer_attn_dim_head)
            is_last = ind >= (num_resolutions - 1)

            # in each downsample stage, 
            # we have a resnetblock and then downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head, act = act, groups = self.groups),
                downsample_layer(dim_in, dim_out) if not is_last else conv_layer(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block = ResnetBlock(mid_dim, mid_dim, use_full_attention = self.full_attn_bottleneck, attn_head = attn_heads[-1], attn_dim_head = attn_dim_head[-1], act = act, groups = self.groups)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            # print(' in upsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out, ' layer_full_attn is: ', layer_full_attn, ' layer_attn_heads is: ', layer_attn_heads, ' layer_attn_dim_head is: ', layer_attn_dim_head)
            is_last = ind == (len(in_out) - 1)
          
            # in each upsample stage,
            # we have a resnetblock and then upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head, act = act, groups = self.groups),
                upsample_layer(dim_out, dim_in) if not is_last else  conv_layer(dim_out, dim_in, 5, padding = 2)  
            ]))

      
        self.final_res_block = ResnetBlock(initial_dim * 2, initial_dim, use_full_attention = None, attn_head = attn_heads[0], attn_dim_head = attn_dim_head[0], act = act, groups = self.groups)
        self.final_conv = conv_layer(initial_dim, self.out_channels, 1) 

    def forward(self, x):

        x = self.init_conv(x)
        # print('initial x shape is: ', x.shape)
        x_init = x.clone()

        h = []
        for block, downsample in self.downs:
            x = block(x)
            h.append(x)

            x = downsample(x)
        
        x = self.mid_block(x)
        # print('middle x shape is: ', x.shape)
        
        for block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)   # h.pop() is the output of the corresponding downsample stage
            x = block(x)
            x = upsample(x)

        x = torch.cat((x, x_init), dim = 1)

        x = self.final_res_block(x)
        final_image = self.final_conv(x)
        # print('final image shape is: ', final_image.shape)
      
        return final_image




# Sampling class
class Sampler(object):
    def __init__(
        self,
        model,
        generator,
        batch_size,
        image_size,
        device = 'cuda',

    ):
        super().__init__()

        # model
        self.model = model  
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.image_size = image_size
        self.batch_size = batch_size

        # dataset and dataloader
        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())    
        self.background_cutoff = self.generator.background_cutoff
        self.maximum_cutoff = self.generator.maximum_cutoff
        self.normalize_factor = self.generator.normalize_factor

        self.dl = dl
        self.cycle_dl = cycle(dl)
 
        # EMA:
        self.ema = EMA(model)
        self.ema.to(self.device)

    def load_model(self, trained_model_filename):

        data = torch.load(trained_model_filename, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.ema.load_state_dict(data["ema"])

    
    def sample_2D(self, trained_model_filename, reference_img):
        
        background_cutoff = self.background_cutoff; maximum_cutoff = self.maximum_cutoff; normalize_factor = self.normalize_factor
        self.load_model(trained_model_filename) 
        
        device = self.device

        self.ema.ema_model.eval()
        # check whether model is on GPU:
        print('model device: ', next(self.ema.ema_model.parameters()).device)

        pred_img = np.zeros((self.image_size[0], self.image_size[1], reference_img.shape[-1]), dtype = np.float32)

        # start to run
        with torch.inference_mode():
            print('gt_img shape: ', reference_img.shape)
            for z_slice in range(0,reference_img.shape[-1]):
                batch_input, batch_gt = next(self.cycle_dl)
                data_input = batch_input.to(device)
                            
                pred_img_slice = self.ema.ema_model(data_input)
                pred_img_slice = pred_img_slice.detach().cpu().numpy().squeeze()
                # print('pred_img_slice shape: ', pred_img_slice.shape)
                pred_img[:,:,z_slice] = pred_img_slice

        
        pred_img = Data_processing.crop_or_pad(pred_img, [reference_img.shape[0], reference_img.shape[1],reference_img.shape[-1]], value = np.min(reference_img))
        pred_img = Data_processing.normalize_image(pred_img, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
        pred_img = Data_processing.correct_shift_caused_in_pad_crop_loop(pred_img)
      
        return pred_img