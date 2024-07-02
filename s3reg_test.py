#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 04:04:14 2023

@author: fenqiang
"""

longleaf = False

import torch
import numpy as np
import glob
import os
from s3pipe.atlas.atlas import SingleBrainSphere
from s3pipe.surface.s3reg import createRegConfig, regOnSingleLevel 
from s3pipe.utils.utils import get_sphere_template
from s3pipe.utils.vtk import read_vtk, write_vtk

 
###############################################################################
""" hyper-parameters """
config = {'hemi': 'rh'}
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
config['device'] = device
config['features'] = ['fgd']
config['levels'] = [7]
config['bi'] = True
config['diffe'] = True
config['num_composition'] = 6
config['weight_smooth'] = [15]
config['centra'] = False
config['val'] = True
config['model_dir'] = '/media/fenqiang/fq/S3_pipeline/pretrained_models'

if longleaf:
    if config['levels'][0] == 7:
        files = sorted(glob.glob('/work/users/f/e/fenqiang/HCP_fgd/HCP_fgd_fq_reg/*/*.'+config['hemi']+'.Sphere.164k.10242moved.164kmoved.Resp164k.npy'))
        config['atlas_file'] = '/work/users/f/e/fenqiang/HCP_fgd/atlas/fgd.'+config['hemi']+'.iter1.SphereSurf.vtk'
    elif config['levels'][0] == 6:
        files = sorted(glob.glob('/work/users/f/e/fenqiang/HCP_fgd/HCP_fgd_fq_reg/*/*.'+config['hemi']+'.Sphere.164k.npy'))
        config['atlas_file'] = '/work/users/f/e/fenqiang/HCP_fgd/atlas/fgd.'+config['hemi']+'.iter0.SphereSurf.vtk'
else:
    if config['levels'][0] == 7:
        files = sorted(glob.glob('/media/fenqiang/MyBook/HCP_fgd_reg/data/*/*.'+config['hemi']+'.Sphere.164k.10242moved.164kmoved.Resp164k.npy'))
        config['atlas_file'] = '/media/fenqiang/MyBook/HCP_fgd_reg/atlas/fgd.'+config['hemi']+'.iter1.SphereSurf.vtk'
    elif config['levels'][0] == 6:
        files = sorted(glob.glob('/media/fenqiang/MyBook/HCP_fgd_reg/data/*/*.'+config['hemi']+'.Sphere.164k.npy'))
        config['atlas_file'] = '/media/fenqiang/MyBook/HCP_fgd_reg/atlas/fgd.'+config['hemi']+'.iter0.SphereSurf.vtk'
    
config = createRegConfig(config)
for models in config['modelss']:
    for model in models:
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

batch_size = 1
train_dataset = SingleBrainSphere(files, config)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False, pin_memory=True)
    
###############################################################################
"""  begin testing, pay no attention! """
for models in config['modelss']:
    for model in models:
        model.eval()

dataiter = iter(train_dataloader)
data, file = next(dataiter)
fixed_xyz_0 = config['fixed_xyz_0']
sphere_temp = get_sphere_template(config['n_vertexs'][-1])

def eval(train_dataloader):
    for batch_idx, (data, file) in enumerate(train_dataloader):
        file = file[0]
        print(file)
        moving = data.to(device)
    
        total_deform = regOnSingleLevel(0, moving, config)
        
        deform_tangent_vector = (1.0/torch.sum(fixed_xyz_0 * total_deform, 1).unsqueeze(1) * total_deform - fixed_xyz_0)*100.0
        real_feat = np.load(file)
        surf = {'vertices': sphere_temp['vertices'],
                'faces': sphere_temp['faces'],
                'fgd': real_feat[0:config['n_vertexs'][-1]],
                'deform_tangent_vector': deform_tangent_vector.detach().cpu().numpy()}
        write_vtk(surf, file.replace('.npy', '.'+str(config['n_vertexs'][-1])+'deform.vtk'))
        
        surf = {'vertices': total_deform.detach().cpu().numpy()*100.0,
                'faces': sphere_temp['faces'],
                'fgd': real_feat[0:config['n_vertexs'][-1]]}
        write_vtk(surf, file.replace('.npy', '.'+str(config['n_vertexs'][-1])+'moved.vtk'))
        
     
eval(train_dataloader)
