#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:44:09 2020

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob
import math

from sphericalunet.utils.utils import get_neighs_order, get_vertex_dis
from sphericalunet.utils.vtk import read_vtk
from sphericalunet.utils.utils_torch import resampleSphereSurf, bilinearResampleSphereSurfImg, bilinearResampleSphereSurf, getEn, get_bi_inter, get_latlon_img, convert2DTo3D, getOverlapIndex
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log/a')

from model import Unet

###########################################################
""" hyper-parameters """

device = torch.device('cuda:0') # torch.device('cpu'), or torch.device('cuda:0')
learning_rate = 0.001

weight_l2 = 1.0
weight_smooth_sulc = [5.0, 10.0, 20.0, 35.0]  # 642, 2562, 10242, 40962
weight_smooth_curv = [35.0]  # 40962
weight_phi_consis = [1.0, 1.0, 1.0, 1.0]
weight_corr = [1.0, 1.0, 1.0, 1.0]

n_vertex = 642
regis_feat = 'sulc' # 'sulc' or 'curv', or 'sc'

diffe = False
if diffe:
    num_composition = 6
truncated = False
if truncated:
    max_disp = get_vertex_dis(n_vertex)/100.0 * 0.2

###########################################################

ind = [642,2562,10242,40962].index(n_vertex)
if regis_feat == 'sulc':
    weight_smooth = weight_smooth_sulc[ind]
elif regis_feat == 'curv':
    weight_smooth = weight_smooth_curv[0]
else:
    print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
weight_phi_consis = weight_phi_consis[ind]
weight_corr = weight_corr[ind]


bi = True
in_ch = 2   # one for fixed sulc, one for moving sulc
out_ch = 2  # two components for tangent plane deformation vector 
batch_size = 1
data_for_test = 0.2

###########################################################
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/M3_regis_sulc_642_nondiffe_smooth80_phiconsis10_corr10/training_40962/sub*.lh.SphereSurf.Orig.resampled.642.norm.DL.moved_3.upto40962.resampled.40962.npy'))

test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, global parameter pre-defined """

def get_atlas(n_vertex, regis_feat, device):
    fixed_0 = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+str(n_vertex)+'.rotated_0.norm.vtk')
    
    if regis_feat == 'sulc':
        fixed_sulc = fixed_0['sulc']
    elif regis_feat == 'curv':
        fixed_sulc = fixed_0['curv']
    else:
        raise NotImplementedError('feat should be curv or sulc.')
        
    fixed_sulc = fixed_sulc[:, np.newaxis]
    fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)
    
    fixed_xyz_0 = fixed_0['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).cuda(device)
    
    fixed_1 = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+str(n_vertex)+'.rotated_1.norm.vtk')
    fixed_xyz_1 = fixed_1['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_1 = torch.from_numpy(fixed_xyz_1.astype(np.float32)).cuda(device)
    
    fixed_2 = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+str(n_vertex)+'.rotated_2.norm.vtk')
    fixed_xyz_2 = fixed_2['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_2 = torch.from_numpy(fixed_xyz_2.astype(np.float32)).cuda(device)
    
    return fixed_xyz_0, fixed_xyz_1, fixed_xyz_2, fixed_sulc


############################################################################
    
fixed_xyz_0, fixed_xyz_1, fixed_xyz_2, fixed_sulc = get_atlas(n_vertex, regis_feat, device)

grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
grad_filter[6] = -6 

neigh_orders = get_neighs_order('neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_' + str(0) + '.mat')
neigh_orders = torch.from_numpy(neigh_orders).to(device)

En_0, En_1, En_2 = getEn(n_vertex, device)

rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, z_weight_0, z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, index_1_0, index_2_0, index_double_02, index_double_12, index_double_01, index_triple_computed = getOverlapIndex(n_vertex, device)
merge_index = getOverlapIndex(n_vertex, device)
#merge_index_642 = getOverlapIndex(642, device)

if bi:
    bi_inter_0, bi_inter_1, bi_inter_2 = get_bi_inter(n_vertex, device)
    img0 = get_latlon_img(bi_inter_0, fixed_sulc)
    img1 = get_latlon_img(bi_inter_1, fixed_sulc)
    img2 = get_latlon_img(bi_inter_2, fixed_sulc)
else:
    bi_inter_0 = None
    bi_inter_1 = None
    bi_inter_2 = None

#img0 = torch.transpose(img0, 0, 2).transpose(1, 2).unsqueeze(0)
#img1 = torch.transpose(img1, 0, 2).transpose(1, 2).unsqueeze(0)
#img2 = torch.transpose(img2, 0, 2).transpose(1, 2).unsqueeze(0)


#############################################################

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, files, regis_feat):
        self.files = files
        self.regis_feat = regis_feat

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        if self.regis_feat == 'sulc':
            sulc = data[:,1]
        else:
            sulc = data[:,0]
            
        return sulc.astype(np.float32)

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files, regis_feat)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#val_dataset = BrainSphere(test_files)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
n_res = level-1 if level<6 else 5

model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
model_0.to(device)
optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=learning_rate,  betas=(0.9, 0.999))

model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
model_1.to(device)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate,  betas=(0.9, 0.999))

model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
model_2.to(device)
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


def diffeomorp(fixed_xyz, phi_3d, num_composition=6, bi=False, bi_inter=None, neigh_orders=None, device=None):
    if bi:
        assert bi_inter is not None, "bi_inter is None!"
        
    warped_vertices = fixed_xyz + phi_3d
    warped_vertices = warped_vertices/(torch.norm(warped_vertices, dim=1, keepdim=True).repeat(1,3))
    
    # compute exp
    for i in range(num_composition):
        if bi:
            warped_vertices = bilinearResampleSphereSurf(warped_vertices, warped_vertices.clone(), bi_inter)
        else:
            warped_vertices = resampleSphereSurf(fixed_xyz, warped_vertices, warped_vertices, neigh_orders, device)
        
        warped_vertices = warped_vertices/(torch.norm(warped_vertices, dim=1, keepdim=True).repeat(1,3))
    
    return warped_vertices




def multi_level_loss(n_vertex, fixed_sulc, moving_sulc, fixed_xyz, phi_3d, num_composition, neigh_orders, merge_index, phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, phi_3d_2_orig, phi_3d_0_to_2, phi_3d_orig, device, bi=False, img0=None, bi_inter=None):
    if bi:
        assert img0 is not None and bi_inter is not None, "img0 is None!"
        
    # divide to small veloctiy field
    phi_3d = phi_3d/math.pow(2,num_composition)
    print(torch.norm(phi_3d,dim=1).max().item())
    moving_warp_phi_3d = diffeomorp(fixed_xyz, phi_3d, num_composition=num_composition, bi=bi, bi_inter=bi_inter, neigh_orders=neigh_orders, device=device)
    
    """ compute interpolation values on fixed surface """
    if bi:
        fixed_inter = bilinearResampleSphereSurf(moving_warp_phi_3d, img0)
    else:
        fixed_inter = resampleSphereSurf(fixed_xyz, moving_warp_phi_3d, fixed_sulc, neigh_orders, device)
    
    loss_corr = 1 - ((fixed_inter - fixed_inter.mean()) * (moving_sulc - moving_sulc.mean())).mean() / fixed_inter.std() / moving_sulc.std()
               
    loss_l2 = torch.mean((fixed_inter - moving_sulc)**2)
              
    loss_smooth = torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
                  torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
                  torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:,[2]][neigh_orders].view(n_vertex, 7), grad_filter))
    loss_smooth = torch.mean(loss_smooth)
                  
    loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[merge_index[7]] - phi_3d_1_orig[merge_index[7]])) + \
                           torch.mean(torch.abs(phi_3d_1_to_2[merge_index[8]] - phi_3d_2_orig[merge_index[8]])) + \
                           torch.mean(torch.abs(phi_3d_0_to_2[merge_index[9]] - phi_3d_2_orig[merge_index[9]]))
     
    return  loss_l2, loss_corr, loss_smooth, loss_phi_consistency


for epoch in range(80):
    lr = get_learning_rate(epoch)
    for optimizer in optimizers:
        optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(lr))
    
    for batch_idx, (moving_0) in enumerate(train_dataloader):
        
        model_0.train()
        model_1.train()
        model_2.train()
        
        moving = torch.transpose(moving_0, 0, 1).to(device)
        data = torch.cat((moving, fixed_sulc), 1)
        
        # tangent vector field phi
        phi_2d_0_orig = model_0(data)
        phi_2d_1_orig = model_1(data)
        phi_2d_2_orig = model_2(data)
        
        phi_3d_0_orig = convert2DTo3D(phi_2d_0_orig, En_0, device)
        phi_3d_1_orig = convert2DTo3D(phi_2d_1_orig, En_1, device)
        phi_3d_2_orig = convert2DTo3D(phi_2d_2_orig, En_2, device)
        
        """ deformation consistency  """
        phi_3d_0_to_1 = torch.mm(rot_mat_01, torch.transpose(phi_3d_0_orig, 0, 1))
        phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
        phi_3d_1_to_2 = torch.mm(rot_mat_12, torch.transpose(phi_3d_1_orig, 0, 1))
        phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
        phi_3d_0_to_2 = torch.mm(rot_mat_02, torch.transpose(phi_3d_0_orig, 0, 1))
        phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)
        
        """ first merge """
        phi_3d = torch.zeros(len(En_0), 3).cuda(device)
        phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2_orig[index_double_02])/2.0
        phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2_orig[index_double_12])/2.0
        tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1_orig[index_double_01])/2.0
        phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp,0,1)), 0, 1)
        phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + phi_3d_2_orig[index_triple_computed] + phi_3d_0_to_2[index_triple_computed])/3.0
        phi_3d_orig = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)
        
       
        if diffe:
            """ diffeomorphism  """
            # divide to small veloctiy field
            phi_3d = phi_3d_orig/math.pow(2,num_composition)
            # print(torch.norm(phi_3d,dim=1).max().item())
            moving_warp_phi_3d = diffeomorp(fixed_xyz_0, phi_3d, num_composition=num_composition, bi=bi, bi_inter=bi_inter_0, neigh_orders=neigh_orders, device=device)
        
        else:
            """ Non diffeomorphism  """
            # print(torch.norm(phi_3d_orig,dim=1).max().item())
            moving_warp_phi_3d = fixed_xyz_0 + phi_3d_orig
            moving_warp_phi_3d = moving_warp_phi_3d/(torch.norm(moving_warp_phi_3d, dim=1, keepdim=True).repeat(1,3))
        
        
        # truncate
        # if truncated:
        #     tmp = torch.norm(phi_3d, dim=1) > max_disp
        #     phi_3d_tmp = phi_3d.clone()
        #     phi_3d_tmp[tmp] = phi_3d[tmp] / (torch.norm(phi_3d[tmp], dim=1, keepdim=True).repeat(1,3)) * max_disp
        #     phi_3d = phi_3d_tmp
        

        """ compute interpolation values on fixed surface """
        if bi:
            fixed_inter = bilinearResampleSphereSurfImg(moving_warp_phi_3d, img0)
        else:
            fixed_inter = resampleSphereSurf(fixed_xyz_0, moving_warp_phi_3d, fixed_sulc, neigh_orders, device)
                
        
        loss_corr = 1 - ((fixed_inter - fixed_inter.mean()) * (moving - moving.mean())).mean() / fixed_inter.std() / moving.std()
        loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[merge_index[7]] - phi_3d_1_orig[merge_index[7]])) + \
                               torch.mean(torch.abs(phi_3d_1_to_2[merge_index[8]] - phi_3d_2_orig[merge_index[8]])) + \
                            torch.mean(torch.abs(phi_3d_0_to_2[merge_index[9]] - phi_3d_2_orig[merge_index[9]]))
        # if epoch < 10:
        loss_l2 = torch.mean((fixed_inter - moving)**2)
        loss_smooth = torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
                      torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
                      torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:,[2]][neigh_orders].view(n_vertex, 7), grad_filter))
        loss_smooth = torch.mean(loss_smooth)

        loss = weight_l2 * loss_l2 + weight_corr * loss_corr + weight_smooth * loss_smooth + weight_phi_consis * loss_phi_consistency 

        
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
       
        print("[Epoch {}] [Batch {}/{}] [loss_l2: {:5.4f}] [loss_corr: {:5.4f}] [loss_smooth: {:5.4f}] [loss_phi_consistency: {:5.4f}]".format(epoch, batch_idx, len(train_dataloader),
                                        loss_l2.item(), loss_corr.item(), loss_smooth.item(), loss_phi_consistency.item()))
        writer.add_scalars('Train/loss', {'loss_l2': loss_l2.item()*weight_l2,
                                          'loss_corr': loss_corr.item()*weight_corr, 
                                          'loss_smooth': loss_smooth.item()*weight_smooth, 
                                          'loss_phi_consistency': loss_phi_consistency.item()*weight_phi_consis}, 
                                          epoch*len(train_dataloader) + batch_idx)
    
    torch.save(model_0.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/regis_"+regis_feat+"_"+str(n_vertex)+"_0.mdl")
    torch.save(model_1.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/regis_"+regis_feat+"_"+str(n_vertex)+"_1.mdl")
    torch.save(model_2.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/regis_"+regis_feat+"_"+str(n_vertex)+"_2.mdl")
    
    