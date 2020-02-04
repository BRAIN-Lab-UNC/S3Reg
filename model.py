#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: zfq

"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import Get_neighs_order, Get_upconv_index
from layers import onering_conv_layer, pool_layer, upconv_layer
#from unet_parts import *

class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()


#        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica unet
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x
    
    
class Unet(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch, level=7, n_res=5, rotated=0):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            level (int) - - input surface's icosahedron level. default: 7, for 40962 vertices
                            2:42, 3:162, 4:642, 5:2562, 6:10242
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or 
                               90 degrees along x axis (1)
        """
        super(Unet, self).__init__()
        
        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than iput level"
        assert n_res >=2, "number of resolution levels should be at least larger than 2"     
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*32)
        
        conv_layer = onering_conv_layer
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
      
        self.up = nn.ModuleList([])
        for i in range(n_res-1):
            self.up.append(up_block(conv_layer, chs[n_res-i], chs[n_res-1-i], neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
            
        self.outc = nn.Linear(chs[1], out_ch)
                
        self.n_res = n_res
        
    def forward(self, x):
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        x = xs[-1]
        for i in range(self.n_res-1):
            x = self.up[i](x, xs[self.n_res-1-i])

        x = self.outc(x) # N * 2
        return x
        



class VGG12(nn.Module):
    """Define the VGG

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        """
        super(VGG12, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.oneside_pad1 = nn.ZeroPad2d((0,1,0,1))
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.oneside_pad2 = nn.ZeroPad2d((0,1,0,1))
        self.block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.oneside_pad3 = nn.ZeroPad2d((0,1,0,1))
        self.block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.out =  nn.Sequential(
            nn.Linear(256, 512),
            nn.Linear(512, out_ch)
        )
        
        self.spatial_transform = SpatialTransformer()
        
    def forward(self, src, tgt):
        x = torch.cat((src, tgt), dim=1)   # N*2*65*65
        x = self.block1(x)        # N*32*65*65
        x = self.oneside_pad1(x)   # N*32*66*66
        x = self.block2(x)        # N*64*33*33
        x = self.oneside_pad2(x)   # N*64*34*34
        x = self.block3(x)        # N*128*17*17
        x = self.oneside_pad3(x)   # N*128*18*18
        x = self.block4(x)        # N*256*9*9
        x = torch.mean(x, dim=(2,3))  # N*256
        flow = self.out(x)               # N*2
        
        y = self.spatial_transform(tgt, flow)  # N*1
    
        return flow, y




class SpatialTransformer(nn.Module):
    """
   
    """
    def __init__(self, mode='bilinear'):
        """
       
        """
        super(SpatialTransformer, self).__init__()

        self.mode = mode

    def forward(self, tgt, flow):   
        """
        Push the src and flow through the spatial transform block
            tgt: N*1*65*65
            flow: N*2
        """
        assert flow.shape[0] == tgt.shape[0], "Flow size shoulbe be consistent with images"
        assert flow.shape == (flow.shape[0], 2), "Error!"
        
        side = int(tgt.shape[2]/2) # 32
        
        new_locs = flow*side
        right_top = torch.ceil(new_locs).long()
        right_top[right_top <= -side] = -side+1  # rights top should beyond the left bottom boundary
        left_bottom = right_top - 1
        
        left_top = torch.cat((left_bottom[:,0].unsqueeze(1), right_top[:,1].unsqueeze(1)), dim=1)
        right_bottom = torch.cat((right_top[:,0].unsqueeze(1), left_bottom[:,1].unsqueeze(1)), dim=1)
        
        weight_mat = torch.cat((((right_top[:,0].float()-new_locs[:,0])*(new_locs[:,1]-right_bottom[:,1].float())).unsqueeze(1),
                                ((right_top[:,0].float()-new_locs[:,0])*(right_top[:,1].float()-new_locs[:,1])).unsqueeze(1),
                                ((new_locs[:,0]-left_bottom[:,0].float()) *(right_top[:,1].float()-new_locs[:,1])).unsqueeze(1),
                                ((new_locs[:,0]-left_bottom[:,0].float()) *(new_locs[:,1]-right_bottom[:,1].float())).unsqueeze(1)), dim=1)
        
        right_top_glo = torch.zeros_like(right_top)
        right_top_glo[:,1] = right_top[:,0] + side
        right_top_glo[:,0] = side - right_top[:,1]
        right_top = right_top_glo
        left_bottom = torch.zeros_like(right_top)
        left_bottom[:,0] = right_top[:,0] + 1
        left_bottom[:,1] = right_top[:,1] - 1
         
        left_top = torch.cat((right_top[:,0].unsqueeze(1), left_bottom[:,1].unsqueeze(1)), dim=1)
        right_bottom = torch.cat((left_bottom[:,0].unsqueeze(1), right_top[:,1].unsqueeze(1)), dim=1)
       
        left_top_value = tgt[list(range(tgt.shape[0])), [0], left_top[:,0], left_top[:,1]]
        left_bottom_value = tgt[list(range(tgt.shape[0])), [0], left_bottom[:,0], left_bottom[:,1]]
        right_bottom_value = tgt[list(range(tgt.shape[0])), [0], right_bottom[:,0], right_bottom[:,1]]
        right_top_value = tgt[list(range(tgt.shape[0])), [0], right_top[:,0], right_top[:,1]]
        
        value = torch.cat((left_top_value.unsqueeze(1),left_bottom_value.unsqueeze(1),right_bottom_value.unsqueeze(1),right_top_value.unsqueeze(1)), dim=1)
       
        y = torch.sum(value*weight_mat, dim=1)
        
        return y



#class Unet_2d(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(Unet_2d, self).__init__()
#        self.inc = inconv(in_ch, 32)
#        self.down1 = down(32, 64)
#        self.down2 = down(64, 128)
#        self.down3 = down(128, 256)
#        self.up2 = up(256, 128)
#        self.up3 = up(128, 64)
#        self.up4 = up(64, 32)
#        self.outc = outconv(32, out_ch)
#
#    def forward(self, x):
#        print(x.size())
#        x1 = self.inc(x)
#        print(x.size())
#        x2 = self.down1(x1)
#        x3 = self.down2(x2)
#        x = self.down3(x3)
#        x = self.up2(x, x3)
#        x = self.up3(x, x2)
#        x = self.up4(x, x1)
#        x = self.outc(x)
#        return x