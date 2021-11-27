__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.ConvLSTMCell import ConvLSTMCell
#from core.layers.octconv import *
#from core.models.pytorch_compact_bilinear_pooling.compact_bilinear_pooling import  CompactBilinearPooling
import math
import numpy as np
import torch.nn.functional as F
#from pytorch_wavelets import DWTForward, DWTInverse

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size*configs.img_channel
        #print("frame_channel", self.frame_channel)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        #self.alpha_in = 0.5
        #self.alpha_out = 0.5
        #padding_size = configs.filter_size // 2
        cell_list = []
        #out_channels_first = num_hidden[0]
        width = int((configs.img_width/2) // configs.patch_size )
        #print('width_real', width)
        # for Haar filter, single level 
        #input_channels = self.frame_channel                           # 16 for MNIST, 64 for KTH 
        
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i],width, width, configs.filter_size, stride=configs.stride, layer_norm=configs.layer_norm))
                                        
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)
        #self.out = nn.Sigmoid()

    def forward(self, frames_dwt, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames_dwt = frames_dwt.permute(0, 1, 4, 2, 3).contiguous()
        #print('frames_dwt_forward', frames_dwt.shape)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames_dwt.shape[0]
        height = frames_dwt.shape[3]
        width = frames_dwt.shape[4]
        #print('height', height, 'width', width)  
        next_frames_dwt = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros  = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            
        for t in range(self.configs.total_length-1):
            if t < self.configs.input_length:
                net = frames_dwt[:,t]
            else:   
                #x_gen_dwt is the residual which was generated if the residual, i.e. for x9 as input (x10 - x9) is the output
                x_gen_dwt = net + x_gen_dwt
                net = mask_true[:, t-self.configs.input_length] * frames_dwt[:, t] + \
                       (1 - mask_true[:, t - self.configs.input_length]) * x_gen_dwt
                           
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
           
            for i in range(1, self.num_layers): 
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])
           
            x_gen_dwt = self.conv_last(h_t[self.num_layers-1])

            next_frames_dwt.append(x_gen_dwt)
            
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames_dwt = torch.stack(next_frames_dwt, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        #print('next_frames_dwt_res', next_frames_dwt_res.shape)
        return next_frames_dwt                  # (8, 16, 16, 16, 16)

##############################
#        Discriminator
##############################


class DiscriminatorA(nn.Module):
    def __init__(self, configs):
        super(DiscriminatorA, self).__init__()
        channels = 4
        height = configs.img_width//2
        width = configs.img_width//2

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
