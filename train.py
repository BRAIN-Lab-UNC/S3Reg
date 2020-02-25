#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:29:13 2019

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob

from utils import Get_neighs_order, get_orthonormal_vectors, get_z_weight
from utils_vtk import read_vtk
from utils_torch import resampleSphereSurf
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/40962')

from model import Unet

###########################################################
""" hyper-parameters """

in_ch = 2   # one for sulc in fixed, one for sulc in moving
out_ch = 2  # two components for tangent plane deformation vector 
device = torch.device('cuda:0') # torch.device('cpu'), or torch.device('cuda:0')
learning_rate = 0.05
batch_size = 1
data_for_test = 0.3
#weight_corr = 0.3
weight_smooth = 0.04
weight_l2 = 5.0
weight_l1 = 1.0
weight_phi_consis = 0.5
regis_feat = 'curv' # 'sulc' or 'curv'

n_vertex = 40962

###########################################################
""" split files, only need 18 month now"""

#files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.'+str(n_vertex)+'.npy'))
#files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

#files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_2562_3d_smooth0p33_phiconsis1_3model/training_10242/*sucu_resampled.10242.npy'))
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_10242_3d_smooth0p8_phiconsis1_3model_one_step_truncated/training_40962/*.40962.npy'))

test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, global parameter pre-defined """

fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_0.vtk')
if regis_feat == 'sulc':
    fixed_sulc = (fixed['sulc'] + 11.5)/(13.65+11.5)
elif regis_feat == 'curv':
    fixed_sulc = (fixed['curv'] + 2.32)/(2.08+2.32)
else:
    raise NotImplementedError('feat should be curv or sulc.')
fixed_sulc = fixed_sulc[:, np.newaxis]
fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)

fixed_xyz_0 = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).cuda(device)
fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_1.vtk')
fixed_xyz_1 = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_1 = torch.from_numpy(fixed_xyz_1.astype(np.float32)).cuda(device)
fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_2.vtk')
fixed_xyz_2 = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_2 = torch.from_numpy(fixed_xyz_2.astype(np.float32)).cuda(device)

grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
grad_filter[6] = -6 

ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
n_res = level-1 if level<6 else 5

neigh_orders = Get_neighs_order(0)[8-level]
neigh_orders = torch.from_numpy(neigh_orders).cuda(device)
assert len(neigh_orders) == n_vertex * 7, "neigh_orders wrong!"

z_weight_0 = get_z_weight(n_vertex, 0)
z_weight_0 = torch.from_numpy(z_weight_0.astype(np.float32)).cuda(device)
index_0_0 = (z_weight_0 == 1).nonzero()
index_0_1 = (z_weight_0 < 1).nonzero()
assert len(index_0_0) + len(index_0_1) == n_vertex, "error!"
z_weight_1 = get_z_weight(n_vertex, 1)
z_weight_1 = torch.from_numpy(z_weight_1.astype(np.float32)).cuda(device)
index_1_0 = (z_weight_1 == 1).nonzero()
index_1_1 = (z_weight_1 < 1).nonzero()
assert len(index_1_0) + len(index_1_1) == n_vertex, "error!"
z_weight_2 = get_z_weight(n_vertex, 2)
z_weight_2 = torch.from_numpy(z_weight_2.astype(np.float32)).cuda(device)
index_2_0 = (z_weight_2 == 1).nonzero()
index_2_1 = (z_weight_2 < 1).nonzero()
assert len(index_2_0) + len(index_2_1) == n_vertex, "error!"

index_01 = np.intersect1d(index_0_0.detach().cpu().numpy(), index_1_0.detach().cpu().numpy())
index_02 = np.intersect1d(index_0_0.detach().cpu().numpy(), index_2_0.detach().cpu().numpy())
index_12 = np.intersect1d(index_1_0.detach().cpu().numpy(), index_2_0.detach().cpu().numpy())
index_01 = torch.from_numpy(index_01).cuda(device)
index_02 = torch.from_numpy(index_02).cuda(device)
index_12 = torch.from_numpy(index_12).cuda(device)
rot_mat_01 = torch.tensor([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                           [0., 1., 0.],
                           [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]]).cuda(device)
rot_mat_12 = torch.tensor([[1., 0., 0.],
                           [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                           [0, np.sin(np.pi/2), np.cos(np.pi/2)]]).cuda(device)
rot_mat_02 = torch.mm(rot_mat_12, rot_mat_01)

En_0 = get_orthonormal_vectors(n_vertex, rotated=0)
En_0 = torch.from_numpy(En_0.astype(np.float32)).cuda(device)
En_1 = get_orthonormal_vectors(n_vertex, rotated=1)
En_1 = torch.from_numpy(En_1.astype(np.float32)).cuda(device)
En_2 = get_orthonormal_vectors(n_vertex, rotated=2)
En_2 = torch.from_numpy(En_2.astype(np.float32)).cuda(device)

#############################################################

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, files, regis_feat):
        self.files = files
        self.regis_feat = regis_feat

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        if self.regis_feat == 'sulc':
            sulc = (data[:,1]+11.5)/(13.65+11.5)
        else:
            sulc = (data[:,0]+2.32)/(2.08+2.32)
        
        return sulc.astype(np.float32)

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files, regis_feat)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#val_dataset = BrainSphere(test_files)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
model_0.cuda(device)
#print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=learning_rate,  betas=(0.9, 0.999))

model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
model_1.cuda(device)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate,  betas=(0.9, 0.999))

model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
model_2.cuda(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate,  betas=(0.9, 0.999))

optimizers = [optimizer_0, optimizer_1, optimizer_2]

def get_learning_rate(epoch):
    limits = [5, 15, 30]
    lrs = [1, 0.5, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


for epoch in range(80):
    lr = get_learning_rate(epoch)
    for optimizer in optimizers:
        optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(lr))
    
#    dataiter = iter(train_dataloader)
#    moving = dataiter.next()
    
    for batch_idx, (moving) in enumerate(train_dataloader):

        model_0.train()
        model_1.train()
        model_2.train()
        
        moving = torch.transpose(moving, 0, 1).cuda(device)
        data = torch.cat((moving, fixed_sulc), 1)
    
        # registration field phi
        phi_2d_0 = model_0(data)
        phi_2d_1 = model_1(data)
        phi_2d_2 = model_2(data)
#        phi_2d = torch.tanh(phi_2d) * fix_vertex_dis
        
        phi_3d_0 = torch.zeros(3, len(En_0)).cuda(device)
        for j in range(len(En_0)):
            phi_3d_0[:,j] = torch.squeeze(torch.mm(En_0[j,:,:], torch.unsqueeze(phi_2d_0[j,:],1)))
        phi_3d_1 = torch.zeros(3, len(En_1)).cuda(device)
        for j in range(len(En_1)):
            phi_3d_1[:,j] = torch.squeeze(torch.mm(En_1[j,:,:], torch.unsqueeze(phi_2d_1[j,:],1)))
        phi_3d_2 = torch.zeros(3, len(En_2)).cuda(device)
        for j in range(len(En_2)):
            phi_3d_2[:,j] = torch.squeeze(torch.mm(En_2[j,:,:], torch.unsqueeze(phi_2d_2[j,:],1)))
        
        """ deformation consistency  """
        phi_3d_0_to_1 = torch.mm(rot_mat_01, phi_3d_0)
        phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
        phi_3d_1_to_2 = torch.mm(rot_mat_12, phi_3d_1)
        phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
        phi_3d_0_to_2 = torch.mm(rot_mat_02, phi_3d_0)
        phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)
        
        """ warp moving image """
        phi_3d_0 = torch.transpose(phi_3d_0, 0, 1)
        moving_warp_phi_3d_0 = fixed_xyz_0 + phi_3d_0
        moving_warp_phi_3d_0 = moving_warp_phi_3d_0/(torch.norm(moving_warp_phi_3d_0, dim=1, keepdim=True).repeat(1,3)) # normalize the deformed vertices onto the sphere
        phi_3d_1 = torch.transpose(phi_3d_1, 0, 1)
        moving_warp_phi_3d_1 = fixed_xyz_1 + phi_3d_1
        moving_warp_phi_3d_1 = moving_warp_phi_3d_1/(torch.norm(moving_warp_phi_3d_1, dim=1, keepdim=True).repeat(1,3)) # normalize the deformed vertices onto the sphere
        phi_3d_2 = torch.transpose(phi_3d_2, 0, 1)
        moving_warp_phi_3d_2 = fixed_xyz_2 + phi_3d_2
        moving_warp_phi_3d_2 = moving_warp_phi_3d_2/(torch.norm(moving_warp_phi_3d_2, dim=1, keepdim=True).repeat(1,3)) # normalize the deformed vertices onto the sphere
        
        """ compute interpolation values on fixed surface """
        fixed_inter_0 = resampleSphereSurf(moving_warp_phi_3d_0, fixed_xyz_0, fixed_sulc, neigh_orders, device)
        fixed_inter_1 = resampleSphereSurf(moving_warp_phi_3d_1, fixed_xyz_1, fixed_sulc, neigh_orders, device)
        fixed_inter_2 = resampleSphereSurf(moving_warp_phi_3d_2, fixed_xyz_2, fixed_sulc, neigh_orders, device)
        
        loss_l1 = torch.mean(torch.abs(fixed_inter_0 - moving) * z_weight_0.unsqueeze(1)) + \
                  torch.mean(torch.abs(fixed_inter_1 - moving) * z_weight_1.unsqueeze(1)) + \
                  torch.mean(torch.abs(fixed_inter_2 - moving) * z_weight_2.unsqueeze(1))
                  
#        loss_corr = 1 - ((fixed_inter - fixed_inter.mean()) * (moving - moving.mean())).mean() / fixed_inter.std() / moving.std()  # compute correlation between fixed and moving_warp_phi
        loss_l2 = torch.mean((fixed_inter_0 - moving)**2 * z_weight_0.unsqueeze(1)) + \
                  torch.mean((fixed_inter_1 - moving)**2 * z_weight_1.unsqueeze(1)) + \
                  torch.mean((fixed_inter_2 - moving)**2 * z_weight_2.unsqueeze(1))
                  
        tmp_0 = torch.abs(torch.mm(phi_3d_0[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_0[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_0[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1)
        tmp_1 = torch.abs(torch.mm(phi_3d_1[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_1[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_1[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1)
        tmp_2 = torch.abs(torch.mm(phi_3d_2[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_2[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_2[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1)
        loss_smooth = torch.mean(tmp_0) + torch.mean(tmp_1) + torch.mean(tmp_2)
        
        loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[index_01] - phi_3d_1[index_01])) + \
                               torch.mean(torch.abs(phi_3d_1_to_2[index_12] - phi_3d_2[index_12])) + \
                               torch.mean(torch.abs(phi_3d_0_to_2[index_02] - phi_3d_2[index_02]))
         
        loss = weight_l1 * loss_l1 + weight_smooth * loss_smooth + weight_l2 * loss_l2 + weight_phi_consis * loss_phi_consistency #+ weight_corr * loss_corr
    
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
       
        print("[Epoch {}] [Batch {}/{}] [loss_l1: {:5.4f}] [loss_l2: {:5.4f}] [loss_smooth: {:5.4f}] [loss_phi_consistency: {:5.4f}]".format(epoch, batch_idx, len(train_dataloader),
                                                            loss_l1.item(), loss_l2.item(), loss_smooth.item(), loss_phi_consistency.item()))
        writer.add_scalars('Train/loss', {'loss_l1': loss_l1.item(), 'loss_l2': loss_l2.item(), 'loss_smooth': loss_smooth.item(), 'loss_phi_consistency': loss_phi_consistency.item()}, 
                                          epoch*len(train_dataloader) + batch_idx)
    
    torch.save(model_0.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/regis_"+regis_feat+"_"+str(n_vertex)+"_3d_smooth0p04_phiconsis0p5_3model_0.mdl")
    torch.save(model_1.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/regis_"+regis_feat+"_"+str(n_vertex)+"_3d_smooth0p04_phiconsis0p5_3model_1.mdl")
    torch.save(model_2.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/regis_"+regis_feat+"_"+str(n_vertex)+"_3d_smooth0p04_phiconsis0p5_3model_2.mdl")
    
    
