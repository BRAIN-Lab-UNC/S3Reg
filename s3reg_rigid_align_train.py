#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:37:17 2023

@author: Fenqiang Zhao,

Contact: zhaofenqiang0221@gmail.com
"""

import torch
import numpy as np
import glob
import os

from s3pipe.utils.utils import S3_normalize
from s3pipe.utils.vtk import read_vtk
from s3pipe.models.models import RigidAlignNet
from s3pipe.utils.interp_torch import get_latlon_img, get_bi_inter, get_rot_mat_zyx_torch, \
    bilinearResampleSphereSurfImg_torch

from torch.utils.tensorboard import SummaryWriter
os.system('rm -rf /media/fenqiang/fq/s3pipeline/main/log/rigidalign')
writer = SummaryWriter('/media/fenqiang/fq/s3pipeline/main/log/rigidalign')


class SingleBrainSphere(torch.utils.data.Dataset):
    def __init__(self, files, n_vertex):
        self.files = files
        self.n_vertex = n_vertex
        
    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        assert data.shape == (163842, ), 'data.shape != (163842, )'
        data = S3_normalize(data)
        return data.astype(np.float32), file
    
    def __len__(self):
        return len(self.files)


hemi = 'lh'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 0.00001
bi = True
batch_size = 1
n_vertex = 163842
metric = 'corr'
weight_corr = 0.45

# toy example
# atlas_file = '/media/fenqiang/fq/LongitudinalRegistrationAndParcellation/RigidAlignNet/data_equator_rec.vtk'
# files = glob.glob('/media/fenqiang/fq/LongitudinalRegistrationAndParcellation/RigidAlignNet/data_equator_rec.*.resp163842.npy')
atlas_file = '/media/fenqiang/DATA/unc/Data/Template/fs_atlas/72_'+ hemi+ '.SphereSurf.vtk'
files = sorted(glob.glob('/media/fenqiang/fq/LongitudinalRegistrationAndParcellation/All_BCP_data_20220830/*/*.'+hemi+'.sphere.Resp163842.npy'))

train_dataset = SingleBrainSphere(files, n_vertex)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, pin_memory=True)

model = RigidAlignNet(in_ch=2, out_ch=3, level=8, n_res=6, rotated=0, complex_chs=16)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.to(device)
model_name = '/media/fenqiang/fq/s3pipeline/main/pretrained_models/rigid_align_'+hemi+'_sulc_163842.mdl'
model.load_state_dict(torch.load(model_name))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


###############################################################################

atlas = read_vtk(atlas_file)
fixed_xyz = torch.from_numpy(atlas['vertices'].astype(np.float32)).to(device)
fixed_sulc = S3_normalize(atlas['Convexity'])[0: n_vertex]
fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
bi_inter= get_bi_inter(n_vertex, device)[0]

# dataiter = iter(train_dataloader)
# data, file = next(dataiter)
# epoch = 0


for epoch in range(1000):
    model.train()
    # optimizer.param_groups[0]['lr' ] = learning_rate
    print("learning rate = ", learning_rate)
    for batch_idx, (data, file) in enumerate(train_dataloader):
        # print(file[0])
        moving = data.to(device).unsqueeze(0)
        data_1 = torch.cat((moving, fixed_sulc), 1)
        rot_xyz = model(data_1).squeeze()
        # print('rot_xyz:', rot_xyz[0].item(), rot_xyz[1].item(), rot_xyz[2].item())
        rot_mat = get_rot_mat_zyx_torch(rot_xyz[0], rot_xyz[1], rot_xyz[2], device)
        curr_vertices = torch.matmul(rot_mat, torch.permute(fixed_xyz, (1,0)))
        curr_vertices = torch.permute(curr_vertices, (1,0))
        
        if bi:
            fixed_img = get_latlon_img(bi_inter, fixed_sulc.squeeze().unsqueeze(1))
            feat_inter = bilinearResampleSphereSurfImg_torch(curr_vertices, fixed_img, radius=100)
        else:
            raise NotImplementedError('error')
        feat_inter = feat_inter.squeeze()
        moving = moving.squeeze()
        loss_corr =  1 - ((feat_inter - feat_inter.mean()) * \
                      (moving - moving.mean())).mean() / feat_inter.std() / moving.std()
        loss_mse = torch.mean(torch.square(feat_inter - moving))
        loss = loss_mse + loss_corr * weight_corr
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("[Epoch {}/{}] [loss_corr: {:5.4f}] [loss_mse: {:5.4f}]".format(epoch,
                                                                              batch_idx, 
                                                                              loss_corr.item(), 
                                                                              loss_mse.item())) 
        writer.add_scalars('Train/loss', {'loss_corr': loss_corr.item()*weight_corr,
                                          'loss_mse': loss_mse.item()},
                                          epoch*len(train_dataloader)+batch_idx)

    torch.save(model.state_dict(), model_name)
    
    
        
        
        
        
        