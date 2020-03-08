#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: zfq

"""

import torch
import torch.nn as nn
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



class svgg(nn.Module):
    def __init__(self, in_ch, out_ch, level, n_res, rotated=0):
        super(svgg, self).__init__()
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        conv_layer = onering_conv_layer

        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*32)

        sequence = []
        sequence.append(conv_layer(chs[0], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
            
        for l in range(1, len(chs)-1):
            sequence.append(pool_layer(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))
            sequence.append(conv_layer(chs[l+1], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*sequence)    
        self.fc =  nn.Sequential(
                nn.Linear(chs[-1], out_ch)
                )

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        return x



class res_block(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders, first_in_block=False):
        super(res_block, self).__init__()
        
        self.conv1 = onering_conv_layer(c_in, c_out, neigh_orders)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = onering_conv_layer(c_out, c_out, neigh_orders)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.first = first_in_block
    
    def forward(self, x):
        res = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.first:
            res = torch.cat((res,res),1)
        x = x + res
        x = self.relu(x)
        
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResNet, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        self.conv1 =  onering_conv_layer(in_c, 64, neigh_orders_40962)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(0.2)
        
        self.pool1 = pool_layer(neigh_orders_40962, 'max')
        self.res1_1 = res_block(64, 64, neigh_orders_10242)
        self.res1_2 = res_block(64, 64, neigh_orders_10242)
        self.res1_3 = res_block(64, 64, neigh_orders_10242)
        
        self.pool2 = pool_layer(neigh_orders_10242, 'max')
        self.res2_1 = res_block(64, 128, neigh_orders_2562, True)
        self.res2_2 = res_block(128, 128, neigh_orders_2562)
        self.res2_3 = res_block(128, 128, neigh_orders_2562)
        
        self.pool3 = pool_layer(neigh_orders_2562, 'max')
        self.res3_1 = res_block(128, 256, neigh_orders_642, True)
        self.res3_2 = res_block(256, 256, neigh_orders_642)
        self.res3_3 = res_block(256, 256, neigh_orders_642)
        
        self.pool4 = pool_layer(neigh_orders_642, 'max')
        self.res4_1 = res_block(256, 512, neigh_orders_162, True)
        self.res4_2 = res_block(512, 512, neigh_orders_162)
        self.res4_3 = res_block(512, 512, neigh_orders_162)
                
        self.pool5 = pool_layer(neigh_orders_162, 'max')
        self.res5_1 = res_block(512, 1024, neigh_orders_42, True)
        self.res5_2 = res_block(1024, 1024, neigh_orders_42)
        self.res5_3 = res_block(1024, 1024, neigh_orders_42)
        
        self.fc = nn.Linear(1024, out_c)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pool1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        
        x = self.pool2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        
        x = self.pool3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
                
        x = self.pool4(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        
        x = self.pool5(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        x = self.out(x)
        return x


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