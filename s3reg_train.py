#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:44:09 2020

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

# longleaf = True

import torch
import numpy as np
import glob
import os
from s3pipe.surface.atlas import SingleBrainSphere
from s3pipe.surface.s3reg import createRegConfig, regOnSingleLevel
from torch.utils.tensorboard import SummaryWriter

# abspath = os.path.abspath(os.path.dirname(__file__))

###############################################################################
""" hyper-parameters. For training, it is actually unnecessary to use config file, just set all parameters here"""
config = {'hemi': 'lh'}
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
config['device'] = device
config['features'] = ['sulc']
config['levels'] = [5]   # 5: 2562, 6: 10242, 7: 40962
config['learning_rate'] = 0.00001
config['bi'] = True
config['weight_l2'] = [0.5]
config['weight_corr'] = [1.0]
config['weight_smooth'] = [3.0]
config['weight_phi_consis'] = [1.0]
config['weight_centra'] = [0]
config['diffe'] = True
config['num_composition'] = 6
config['centra'] = False
config['model_dir'] = '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models'
config['atlas_file'] = '/proj/ganglilab/users/Fenqiang/Atlas/72/72_'+config['hemi']+'.SphereSurf.vtk'

if longleaf:
    # os.system('rm -rf  /proj/ganglilab/users/Fenqiang/S3Pipeline/train_and_test/log/s3reg_'+config['hemi']+'_'+config['features'][0])
    # writer = SummaryWriter('/proj/ganglilab/users/Fenqiang/S3Pipeline/train_and_test/log/s3reg_'+config['hemi']+'_'+config['features'][0])
    os.system('rm -rf  /proj/ganglilab/users/laifa/S3Pipeline_editable/train_and_test/log/s3reg_'+config['hemi']+'_'+config['features'][0])
    writer = SummaryWriter('/proj/ganglilab/users/laifa/S3Pipeline_editable/train_and_test/log/s3reg_'+config['hemi']+'_'+config['features'][0])
    if config['levels'][0] == 7:
        # files = sorted(glob.glob('/work/users/f/e/fenqiang/HCP_fgd/HCP_fgd_fq_reg/*/*.'+config['hemi']+'.Sphere.164k.10242moved.164kmoved.Resp164k.npy'))
        # config['atlas_file'] = '/work/users/f/e/fenqiang/HCP_fgd/atlas/fgd.'+config['hemi']+'.iter1.SphereSurf.vtk'
        files = sorted(glob.glob('/work/users/l/a/laifama/Fengqiang/'))
    elif config['levels'][0] == 6:
        # files = sorted(glob.glob('/work/users/f/e/fenqiang/HCP_fgd/HCP_fgd_fq_reg/*/*.'+config['hemi']+'.Sphere.164k.npy'))
        # config['atlas_file'] = '/work/users/f/e/fenqiang/HCP_fgd/atlas/fgd.'+config['hemi']+'.iter0.SphereSurf.vtk'
        files = sorted(glob.glob('/work/users/l/a/laifama/LifeSpanDatasetFQProcessed_laifaeditable/*/*/*.'+config['hemi']+'.InnerSurf.S3MapToSphe.2562sulcmoved.vtk'))
    elif config['levels'][0] == 5:
        files = sorted(glob.glob('/work/users/l/a/laifama/LifeSpanDatasetFQProcessed_laifaeditable/*/*/*.'+config['hemi']+'.InnerSurf.S3MapToSphe.RigidAlignToAtlas.Resp163842.npy'))
else:
    os.system('rm -rf  /media/fenqiang/fq/S3_pipeline/main/log/pretrain_fgd_'+config['hemi'])
    writer = SummaryWriter('/media/fenqiang/fq/S3_pipeline/main/log/pretrain_fgd_'+config['hemi'])
    if config['levels'][0] == 7:
        files = sorted(glob.glob('/media/fenqiang/MyBook/HCP_fgd_reg/data/*/*.'+config['hemi']+'.Sphere.164k.10242moved.164kmoved.Resp164k.npy'))
        config['atlas_file'] = '/media/fenqiang/MyBook/HCP_fgd_reg/atlas/fgd.'+config['hemi']+'.iter1.SphereSurf.vtk'
    elif config['levels'][0] == 6:
        files = sorted(glob.glob('/media/fenqiang/MyBook/HCP_fgd_reg/data/*/*.'+config['hemi']+'.Sphere.164k.npy'))
        config['atlas_file'] = '/media/fenqiang/MyBook/HCP_fgd_reg/atlas/fgd.'+config['hemi']+'.iter0.SphereSurf.vtk'
    
print('len(files):', len(files))
    
config = createRegConfig(config)
for models in config['modelss']:
    for model in models:
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

batch_size = 1
train_dataset = SingleBrainSphere(files, config)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, pin_memory=True)
    
###############################################################################
"""  begin training, pay no attention! """
n_levels = config['n_levels']
optimizerss = config['optimizerss']

# # specific atlases for more accurate and faster training of BCP data with age in days
# atlases = []
# ages = ['01', '03', '06', '09', '12', '18', '24', '36', '48', '60', '72']
# for age in ages:
#     atlas = load_atlas('/media/fenqiang/DATA/unc/Data/Template/UNC-Infant-Cortical-Surface-Atlas/'+ age +'/'+ age +'_'+ config['hemi'] +'.SphereSurf.vtk', 
#                        config['n_vertexs'][-1], config['device'])
#     atlases.append(atlas)


# def set_requires_grad(nets, requires_grad=False):
#     """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
#     Parameters:
#         nets (network list)   -- a list of networks
#         requires_grad (bool)  -- whether the networks require gradients or not
#     """
#     if not isinstance(nets, list):
#         nets = [nets]
#     for net in nets:
#         if net is not None:
#             for param in net.parameters():
#                 param.requires_grad = requires_grad
                

for models in config['modelss']:
    for model in models:
        model.train()

# dataiter = iter(train_dataloader)
# data, file = next(dataiter)
# epoch = 0


for epoch in range(100):
    lr = []
    
    for i_level in range(n_levels):
        tmp = config['learning_rate']   # * (i_level * _k + _m)
        lr.append(tmp)
        for optimizer in optimizerss[i_level]:
            optimizer.param_groups[0]['lr' ] = tmp
    print("learning rate = ", lr)
    
    for batch_idx, (data, file) in enumerate(train_dataloader):
        # print(file[0])
        moving = data.to(device)
    
        total_deform = 0.
        total_deform, loss, loss_centra, loss_corr, \
                loss_l2, loss_phi_consistency, \
                    loss_smooth = regOnSingleLevel(0, moving, config, total_deform)
        print("[Epoch {}/{}/{}] [loss_l2: {:5.4f}] [loss_corr: {:5.4f}] [loss_smooth: {:5.4f}] [loss_phi_consistency: {:5.4f}] ".format(epoch, 
                                            batch_idx, 0, loss_l2, 
                                            loss_corr, loss_smooth, 
                                            loss_phi_consistency))
        
    
        # for i_level in range(n_levels):
        #     lo = lo + config['weight_level'][i_level] * losses[i_level]

        # if epoch < 1:
        #     for optimizer in optimizerss[2]:
        #         optimizer.zero_grad()
        #     lo.backward()
        #     for optimizer in optimizerss[2]:
        #         optimizer.step()
        # else:
            
        for optimizers in optimizerss:
            for optimizer in optimizers:
                optimizer.zero_grad()
        loss.backward()
        for optimizers in optimizerss:
            for optimizer in optimizers:
                optimizer.step()

        writer.add_scalars('Train/loss', {'loss_l2': loss_l2*config['weight_l2'][-1],
                                          'loss_corr': loss_corr*config['weight_corr'][-1], 
                                          'loss_smooth': loss_smooth*config['weight_smooth'][-1], 
                                          'loss_phi_consistency': loss_phi_consistency*config['weight_phi_consis'][-1]},
                                          epoch*len(train_dataloader)+batch_idx)

         
        if batch_idx % 100 == 0:
            torch.save(config['modelss'][i_level][0].state_dict(), '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models/S3Reg_'+ \
                       config['hemi'] + '_' + config['features'][i_level]+"_"+str(config['n_vertexs'][i_level])+ '_smooth_' + \
                           str(float(config['weight_smooth'][i_level])) + "_pretrained_0.mdl")
            torch.save(config['modelss'][i_level][1].state_dict(), '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models/S3Reg_'+ \
                       config['hemi'] + '_' + config['features'][i_level]+"_"+str(config['n_vertexs'][i_level])+ '_smooth_' + \
                           str(float(config['weight_smooth'][i_level])) + "_pretrained_1.mdl")
            torch.save(config['modelss'][i_level][2].state_dict(), '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models/S3Reg_'+ \
                       config['hemi'] + '_' + config['features'][i_level]+"_"+str(config['n_vertexs'][i_level])+ '_smooth_' + \
                           str(float(config['weight_smooth'][i_level])) + "_pretrained_2.mdl")

    for i_level in range(n_levels):
        torch.save(config['modelss'][i_level][0].state_dict(), '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models/S3Reg_'+ \
                   config['hemi'] + '_' + config['features'][i_level]+"_"+str(config['n_vertexs'][i_level])+ '_smooth_' + \
                       str(float(config['weight_smooth'][i_level])) + "_pretrained_0.mdl")
        torch.save(config['modelss'][i_level][1].state_dict(), '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models/S3Reg_'+ \
                   config['hemi'] + '_' + config['features'][i_level]+"_"+str(config['n_vertexs'][i_level])+ '_smooth_' + \
                       str(float(config['weight_smooth'][i_level])) + "_pretrained_1.mdl")
        torch.save(config['modelss'][i_level][2].state_dict(), '/proj/ganglilab/users/laifa/S3Pipeline_editable/pretrained_models/S3Reg_'+ \
                   config['hemi'] + '_' + config['features'][i_level]+"_"+str(config['n_vertexs'][i_level])+ '_smooth_' + \
                       str(float(config['weight_smooth'][i_level])) + "_pretrained_2.mdl")