#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:01:40 2020

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob
import argparse
import math 
import time
import os

from sphericalunet.utils.utils import normalize, get_upsample_order
from sphericalunet.utils.vtk import write_vtk, read_vtk
from sphericalunet.utils.interp_torch import  regOnThisLevel, resampleStdSphereSurf
from sphericalunet.utils.create_reg_parameters import readRegConfig, createRegConfig, BrainSphere
from sphericalunet.model import Unet


###########################################################
""" hyper-parameters """

device =torch.device('cpu') # torch.device('cpu'), or torch.device('cuda:0')

# CUDA_LAUNCH_BLOCKING=1

###########################################################
""" pre-defined registration parameters """

config = readRegConfig('./regConfig_3level.txt')
config['device'] = device

in_ch = 2   # one for fixed sulc, one for moving sulc
out_ch = 2  # two components for tangent plane deformation vector 
batch_size = 1

ns_vertex = np.array([12,42,162,642,2562,10242,40962,163842])
n_vertexs = []
for i_level in range(len(config['levels'])):
    n_vertexs.append(ns_vertex[config['levels'][i_level]-1])
n_levels = len(n_vertexs)

config['n_vertexs'] = n_vertexs
config['n_levels'] = n_levels

config = createRegConfig(config)
atlas= read_vtk('/media/ychenp/Seagate/ADNI1FromLi/atlas/SphereSurf.vtk')
atlas['sulc'] = normalize(atlas['sulc'])
atlas['curv'] = normalize(atlas['curv'])
config['atlas']  = atlas

        
###############################################################################
"""  find files  """

files = glob.glob('/media/ychenp/Seagate/ADNI1FromLi/data/ADNI1Surf_S3Reg/*/*/surf/lh.SphereSurf.Resampled.RigidAlignedUsingSearch.Resampled.npy')


train_dataset = BrainSphere(files, n_vertexs[-1], val=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False, pin_memory=True)

############################################################################

def initModel(i_level, config):
    
    level = config['levels'][i_level]
    n_res = level-2 if level<6 else 4
    
    model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
    model_0.to(config['device'])
    model_0.load_state_dict(torch.load('/media/ychenp/DATA/unc/Data/GroupwiseReg/pretrained_models/'+ \
                   config['features'][i_level]+"_"+str(n_vertexs[i_level])+ "_pretrained_0.mdl"))
          
    model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
    model_1.to(config['device'])
    model_1.load_state_dict(torch.load('/media/ychenp/DATA/unc/Data/GroupwiseReg/pretrained_models/'+ \
                   config['features'][i_level]+"_"+str(n_vertexs[i_level])+ "_pretrained_1.mdl"))
        
    model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
    model_2.to(config['device'])
    model_2.load_state_dict(torch.load('/media/ychenp/DATA/unc/Data/GroupwiseReg/pretrained_models/'+ \
                   config['features'][i_level]+"_"+str(n_vertexs[i_level])+ "_pretrained_2.mdl"))
        
    return [model_0, model_1, model_2]

modelss = []
for n_vertex in n_vertexs:
    tmp = initModel(n_vertexs.index(n_vertex), config)
    modelss.append(tmp)
config['modelss'] = modelss
    
for models in modelss:
    for model in models:
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

###############################################################################

for models in modelss:
    for model in models:
        model.eval()

#    dataiter = iter(train_dataloader)
#    norm_data, orig_data, file = dataiter.next()
#    epoch = 0

upsample_neighbors_163842 = get_upsample_order(163842)
fixed_xyz_0 = config['fixed_xyz_0']
atlas = config['atlas']
fixed = torch.from_numpy(np.concatenate((atlas['sulc'][:,np.newaxis], atlas['curv'][:,np.newaxis]),axis=1)).to(device)

with torch.no_grad():
    for batch_idx, (norm_data, orig_data, file) in enumerate(train_dataloader):
        
        print(file[0])
        if os.path.exists(file[0].replace('.npy', '.AlignToAtlas.vtk')):
            continue
        
        t1 = time.time()
        
        moving = norm_data.squeeze(0).to(device)

        total_deform = 0.0
        for i_level in range(n_levels):
            total_deform = regOnThisLevel(i_level, fixed, moving, config, total_deform, val=True)
           
    
        total_deform = resampleStdSphereSurf(n_vertexs[-1], 163842, total_deform, 
                                             upsample_neighbors_163842, device)
        surf = {'vertices': total_deform.cpu().numpy()*100.0,
                'faces': atlas['faces'],
                'sulc': orig_data[0,:,0].cpu().numpy(),
                'curv': orig_data[0,:,1].cpu().numpy()}
        
        t2 = time.time()
        print("Cost: ", (t2-t1), "s")    

        write_vtk(surf, file[0].replace('.npy', '.AlignToAtlas.vtk'))
        