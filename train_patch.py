#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:35:37 2020

@author: fenqiang
"""

import torch
from torch.nn import init

import torchvision
import numpy as np
import glob
import itertools
import math

from utils import Get_neighs_order, get_orthonormal_vectors, get_patch_indices
from utils_vtk import read_vtk
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

from model import VGG12


###########################################################
""" hyper-parameters """

in_ch = 1   # one for sulc
out_ch = 2  # two components for tangent plane deformation vector 
device = torch.device('cuda:0')
learning_rate = 0.01
batch_size_sur = 1
batch_size_patch = 16
data_for_test = 0.3
weight_corr = 0.2
weight_smooth = 0.8
weight_l2 = 2.0
weight_l1 = 1.0
fix_vertex_dis = 0.3

shape = [65,65]
n_vertex = 10242

###########################################################
""" split files, only need 18 month now"""

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.'+str(163842)+'.npy')) 
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]
test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, patch interpolation indices and weights """

inter_indices, inter_weights = get_patch_indices(n_vertex)

fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.vtk')
fixed_faces = fixed['faces']
fixed_faces = torch.from_numpy(fixed_faces[:,[1,2,3]]).cuda(device)
fixed_faces = torch.sort(fixed_faces, axis=1)[0]
fixed_xyz = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz = torch.from_numpy(fixed_xyz.astype(np.float32)).cuda(device)
fixed_sulc = (fixed['sulc'] + 11.5)/(13.65+11.5)
fixed_sulc = fixed_sulc[:, np.newaxis]
fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)
fixed_patches = torch.sum(fixed_sulc[torch.from_numpy(inter_indices.flatten().astype(np.int64))].reshape((n_vertex, -1, 3)) * (torch.from_numpy(inter_weights.astype(np.float32)).cuda(device)), dim=2)
fixed_patches = torch.reshape(fixed_patches, (n_vertex, shape[0], shape[1]))

En = get_orthonormal_vectors(n_vertex)
En = torch.from_numpy(En.astype(np.float32)).cuda(device)

grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
grad_filter[6] = -6 

ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
neigh_orders = Get_neighs_order()[8-level]
assert len(neigh_orders) == n_vertex * 7, "neigh_orders wrong!"


#############################################################
class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = files

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        sulc = (data[:,1]+11.5)/(13.65+11.5)
        
        return sulc.astype(np.float32)

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_sur, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(test_files)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_sur, shuffle=False, pin_memory=True)

model = VGG12(in_ch=2, out_ch=2)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999))
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel')


def get_learning_rate(epoch):
    limits = [5, 15, 30]
    lrs = [1, 0.5, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

#def val_during_training(dataloader):
#    model.eval()
#
##    mae_s = torch.tensor([])
##    mre_s = torch.tensor([])    
#    mae_t = torch.tensor([])
#    mre_t = torch.tensor([])
#    for batch_idx, (data, raw_target) in enumerate(dataloader):
#        data, raw_target = data.squeeze(0).cuda(cuda), raw_target.squeeze(0).cuda(cuda)
#        with torch.no_grad():
#            prediction = model(data).squeeze()
##        prediction_s = prediction[:,[0]].cpu() 
##        raw_target_s = raw_target[:,[0]].cpu()
##        mae_s = torch.cat((mae_s, torch.abs(prediction_s - raw_target_s)), 0)
##        mre_s = torch.cat((mre_s, torch.abs(prediction_s - raw_target_s)/raw_target_s), 0)
#        prediction_t = prediction.cpu()
#        raw_target_t = raw_target.cpu()
#        mae_t = torch.cat((mae_t, torch.abs(prediction_t - raw_target_t)), 0)
#        mre_t = torch.cat((mre_t, torch.abs(prediction_t - raw_target_t)/raw_target_t), 0)
#        
#        
##    m_mae_s, m_mre_s = torch.mean(mae_s), torch.mean(mre_s)
#    m_mae_t, m_mre_t = torch.mean(mae_t), torch.mean(mre_t)
#
#    return   m_mae_t, m_mre_t


for epoch in range(300):
    lr = get_learning_rate(epoch)
    for p in optimizer.param_groups:
        p['lr'] = lr
        print("learning rate = {}".format(p['lr']))
    
#    dataiter = iter(train_dataloader)
#    moving = dataiter.next()
    
    for batch_idx, (moving) in enumerate(train_dataloader):

        model.train()
        moving = torch.transpose(moving, 0, 1).cuda(device)
        moving_patches = torch.sum(moving[torch.from_numpy(inter_indices.flatten().astype(np.int64))].reshape((n_vertex, -1, 3)) * (torch.from_numpy(inter_weights.astype(np.float32)).cuda(device)), dim=2)
        moving_patches = torch.reshape(moving_patches, (n_vertex, shape[0], shape[1]))
        
        phi_2d = torch.zeros(len(En), 2).cuda(device)
        loss_l1 = 0.0
        for batch_pat_idx in range(math.ceil(n_vertex/batch_size_patch)):
            print(batch_pat_idx)
            idx_lower = batch_size_patch * batch_pat_idx
            idx_upper = idx_lower + batch_size_patch 
            
            moving_patch = moving_patches[idx_lower:idx_upper,:,:].unsqueeze(1)
            fixed_patch = fixed_patches[idx_lower:idx_upper,:,:].unsqueeze(1)
            phi_2d[idx_lower:idx_upper,:], y = model(moving_patch, fixed_patch)
            
            loss_l1 = loss_l1 + torch.mean(torch.abs(y - moving[idx_lower:idx_upper].squeeze()))
            
        phi_2d = torch.transpose(phi_2d, 0, 1)
        phi_3d = torch.zeros(3, len(En)).cuda(device)
        for j in range(len(En_2562)):
            phi_3d[:,j] = torch.squeeze(torch.mm(En_2562[j,:,:], torch.unsqueeze(phi_2d[j,:],1)))
       
            
            
        phi_3d = torch.transpose(phi_3d, 0, 1)
 
         = torch.mean(torch.abs(fixed_inter - moving))
        loss_corr = 1 - ((fixed_inter - fixed_inter.mean()) * (moving - moving.mean())).mean() / fixed_inter.std() / moving.std()  # compute correlation between fixed and moving_warp_phi
        loss_l2 = torch.mean((fixed_inter - moving)**2)
        tmp = torch.abs(torch.mm(phi_3d[:,[0]][neigh_orders_2562].view(len(phi_2d), 7), grad_filter)) + \
              torch.abs(torch.mm(phi_3d[:,[1]][neigh_orders_2562].view(len(phi_2d), 7), grad_filter)) + \
              torch.abs(torch.mm(phi_3d[:,[2]][neigh_orders_2562].view(len(phi_2d), 7), grad_filter))
                                 
        # set pole circles' smooth loss as 0
        loss_smooth = torch.mean(tmp)
        
        loss = weight_l1 * loss_l1 + weight_corr * loss_corr + weight_smooth * loss_smooth + weight_l2 * loss_l2
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ("[Epoch {}] [Batch {}/{}] [loss_l1: {:5.4f}] [loss_l2: {:5.4f}] [loss_corr: {:5.4f}] [loss_smooth: {:5.4f}]".format(epoch, batch_idx, len(train_dataloader),
                                                            loss_l1.item(), loss_l2.item(), loss_corr.item(), loss_smooth.item()))
        writer.add_scalars('Train/loss', {'loss_l1': loss_l1.item(), 'loss_l2': loss_l2.item(), 'loss_corr': loss_corr.item(), 'loss_smooth': loss_smooth.item()}, 
                                          epoch*len(train_dataloader) + batch_idx)
    
    torch.save(Unet_2562.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/regis_sulc_2562_with_l2_3d_smooth0p8.mdl")
