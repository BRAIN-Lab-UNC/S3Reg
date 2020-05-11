#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 04:12:15 2020

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob
import math
import time
import os

from utils import get_neighs_order, get_z_weight, get_vertex_dis, get_upsample_order
from utils_vtk import read_vtk, write_vtk
from utils_torch import resampleSphereSurf, bilinearResampleSphereSurfImg, bilinearResampleSphereSurf, getEn, get_bi_inter, get_latlon_img, convert2DTo3D, getOverlapIndex
from utils_interpolation import resampleSphereSurf as resampleSphereSurf_np

from model import Unet

###########################################################
""" hyper-parameters """

device = torch.device('cuda:1') # torch.device('cpu'), or torch.device('cuda:0')
regis_feat = 'sulc' # 'sulc' or 'curv'
n_vertex = 40962

norm_method = '2' # '1': use individual max min, '2': use fixed max min
model_name = "M3_regis_"+regis_feat+"_"+str(n_vertex)+"_nondiffe_smooth5_phiconsis10_corr1"

diffe = False
bi = True
num_composition = 6
num_composition_deform = 6

truncated = False
max_disp = get_vertex_dis(n_vertex)/100.0 * 0.2


###########################################################

if not os.path.exists('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/'+model_name):
    os.makedirs('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/'+model_name)
in_ch = 2   # one for sulc in fixed, one for sulc in moving
out_ch = 2  # two components for tangent plane deformation vector 
batch_size = 1
data_for_test = 0.3

###########################################################
if n_vertex == 642:
    files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/sub*/sub*.lh.SphereSurf.Orig.resampled.'+ str(n_vertex) + '.npy'))
elif n_vertex == 2562:
    files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/M3_regis_sulc_642_nondiffe_smooth15_phiconsis4_corr4/training_2562/*.lh.SphereSurf.Orig.resampled.642.DL.moved_3.upto2562.resampled.2562.npy'))
elif n_vertex == 10242:
    files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/M3_regis_sulc_2562_nondiffe_smooth10_phiconsis2_corr1/training_10242/*.lh.SphereSurf.Orig.resampled.642.DL.moved_3.upto2562.resampled.2562.DL.moved_3.upto10242.resampled.10242.npy'))
elif n_vertex == 40962:
    files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/M3_regis_sulc_10242_diffe_biTrue_smooth5_phiconsis5_corr1/training_40962/*.lh.SphereSurf.Orig.resampled.642.DL.moved_3.upto2562.resampled.2562.DL.moved_3.upto10242.resampled.10242.DL.moved_3.upto40962.resampled.40962.npy'))

test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, global parameter pre-defined """

def get_atlas(n_vertex, regis_feat, norm_method, device):
    fixed_0 = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+str(n_vertex)+'.rotated_0.norm.vtk')
    
    if regis_feat == 'sulc':
        fixed_sulc = fixed_0['sulc']
    elif regis_feat == 'curv':
        fixed_sulc = fixed_0['curv']
    else:
        raise NotImplementedError('feat should be curv or sulc.')
        
    fixed_sulc_ma = fixed_sulc.max()
    fixed_sulc_mi = fixed_sulc.min()
        
    if norm_method == '1':
        fixed_sulc = (fixed_sulc - fixed_sulc.min())/(fixed_sulc.max()-fixed_sulc.min()) * 2. 
    elif norm_method == '2':
        if regis_feat == 'sulc':
            fixed_sulc = (fixed_sulc + 11.5)/(13.65+11.5)
        else:
            fixed_sulc = (fixed_sulc + 1.)/(1.+1.)
    else:
        raise NotImplementedError('norm_method should be 1 or 2.')
    
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
    
    return fixed_0, fixed_1, fixed_2, fixed_xyz_0, fixed_xyz_1, fixed_xyz_2, fixed_sulc, fixed_sulc_ma, fixed_sulc_mi


############################################################################

fixed_0, fixed_1, fixed_2, fixed_xyz_0, fixed_xyz_1, fixed_xyz_2, fixed_sulc, fixed_sulc_ma, fixed_sulc_mi = get_atlas(n_vertex, regis_feat, norm_method, device)

grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
grad_filter[6] = -6 

ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
n_res = level-1 if level<6 else 5

neigh_orders = get_neighs_order('neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_' + str(0) + '.mat')
neigh_orders = torch.from_numpy(neigh_orders).to(device)

En_0, En_1, En_2 = getEn(n_vertex, device)

rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, z_weight_0, z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, index_1_0, index_2_0, index_double_02, index_double_12, index_double_01, index_triple_computed = getOverlapIndex(n_vertex, device)

if bi:
    bi_inter_0, bi_inter_1, bi_inter_2 = get_bi_inter(n_vertex, device)
    img0 = get_latlon_img(bi_inter_0, fixed_sulc)
    img1 = get_latlon_img(bi_inter_1, fixed_sulc)
    img2 = get_latlon_img(bi_inter_2, fixed_sulc)
else:
    bi_inter_0 = None
    bi_inter_1 = None
    bi_inter_2 = None

#############################################################

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, files, regis_feat, norm_method):
        self.files = files
        self.regis_feat = regis_feat
        self.norm_method = norm_method

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        if self.regis_feat == 'sulc':
            sulc = data[:,1]
        else:
            sulc = data[:,0]
            
        ma = sulc.max()
        mi = sulc.min()
        
        if self.norm_method == '1':
            sulc = (sulc - sulc.min())/(sulc.max()-sulc.min()) * 2. 
        else:
            if self.regis_feat == 'sulc':
                sulc = (sulc + 11.5)/(13.65+11.5)
            else:
                sulc = (sulc + + 1.)/(1.+1.)
        
        return sulc.astype(np.float32), file, ma, mi

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files, regis_feat, norm_method)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
val_dataset = BrainSphere(test_files, regis_feat, norm_method)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
model_0.to(device)
model_0.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/' + model_name + '_0.mdl'))

model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
model_1.cuda(device)
model_1.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/' + model_name + '_1.mdl'))

model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
model_2.cuda(device)
model_2.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/' + model_name + '_2.mdl'))



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


#    dataiter = iter(train_dataloader)
#    moving_0, file, ma, mi = dataiter.next()
    

def test(dataloader):
    
    model_0.eval()
    model_1.eval()
    model_2.eval()
    
    t = []
    with torch.no_grad():
        
        for batch_idx, (moving_0, file, ma, mi) in enumerate(dataloader):
            print(file[0])
            
            t0 = time.time()
            
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
            
            # merge 
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
                print(torch.norm(phi_3d,dim=1).max().item())
                moving_warp_phi_3d = diffeomorp(fixed_xyz_0, phi_3d, num_composition=num_composition_deform, bi=bi, bi_inter=bi_inter_0, neigh_orders=neigh_orders, device=device)
            else:
                """ Non diffeomorphism  """     
                moving_warp_phi_3d = fixed_xyz_0 + phi_3d_orig
                moving_warp_phi_3d = moving_warp_phi_3d/(torch.norm(moving_warp_phi_3d, dim=1, keepdim=True).repeat(1,3))
            
            # truncate
            if truncated:
                tmp = torch.norm(phi_3d, dim=1) > max_disp
                phi_3d_tmp = phi_3d.clone()
                phi_3d_tmp[tmp] = phi_3d[tmp] / (torch.norm(phi_3d[tmp], dim=1, keepdim=True).repeat(1,3)) * max_disp
                phi_3d = phi_3d_tmp
		    
#            moving_warp_phi_3d = diffeomorp(fixed_xyz_0[0:40962], phi_3d[0:40962], num_composition=num_composition_deform, bi=True, bi_inter=bi_inter_0, neigh_orders=neigh_orders, device=device)
#            moving_warp_phi_3d = fixed_xyz_0[0:642] + phi_3d[0:642]
#            moving_warp_phi_3d = moving_warp_phi_3d/(torch.norm(moving_warp_phi_3d, dim=1, keepdim=True).repeat(1,3))    
#            moving_warp_phi_3d = resampleSphereSurf_np(fixed_xyz_0[0:642].detach().cpu().numpy(), fixed_xyz_0[0:2562].detach().cpu().numpy(), moving_warp_phi_3d.detach().cpu().numpy(), True, upsample_neighbors_2562)
#            moving_warp_phi_3d = resampleSphereSurf_np(fixed_xyz_0[0:2562].detach().cpu().numpy(), fixed_xyz_0[0:10242].detach().cpu().numpy(), moving_warp_phi_3d, True, upsample_neighbors_10242)
#            moving_warp_phi_3d = resampleSphereSurf_np(fixed_xyz_0[0:10242].detach().cpu().numpy(), fixed_xyz_0[0:40962].detach().cpu().numpy(), moving_warp_phi_3d, True, upsample_neighbors_40962)
#            moving_warp_phi_3d = torch.from_numpy(moving_warp_phi_3d.astype(np.float32)).to(device)
            
#            diffe_deform = 1.0/torch.sum(fixed_xyz_0 * moving_warp_phi_3d, 1).unsqueeze(1) * moving_warp_phi_3d - fixed_xyz_0            
               
                
            t1 = time.time()
            print((t1-t0)*1000, "ms")
            t.append((t1-t0)*1000)
            
            if norm_method == '1':
                tmp = (moving.detach().cpu().numpy()) / 2.* (ma[0].item() - mi[0].item()) + mi[0].item()
            else:
                if regis_feat == 'sulc':
                    tmp = moving.detach().cpu().numpy() * (13.65+11.5) - 11.5
                else:
                    tmp = moving.detach().cpu().numpy() * (1.+1.) - 1
                    
            moved = {'vertices': moving_warp_phi_3d.detach().cpu().numpy()*100.0,
                     'faces': fixed_0['faces'],
                     regis_feat: tmp}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/' + model_name +'/' + file[0].replace('.npy', '.DL.moved_3.vtk').split('/')[-1])

            if diffe:
                real_deform = 1.0/torch.sum(fixed_xyz_0 * moving_warp_phi_3d, 1).unsqueeze(1) * moving_warp_phi_3d - fixed_xyz_0
                origin = {'vertices': fixed_0['vertices'],
                          'faces': fixed_0['faces'],
                          'phi_3d_orig': phi_3d_orig.detach().cpu().numpy() * 100.0,
                          'real_deform': real_deform.detach().cpu().numpy() * 100.0,
                          regis_feat: tmp}
                write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/' + model_name + '/' + file[0].replace('.npy', '.DL.origin_3.vtk').split('/')[-1]) 
            else:
                origin = {'vertices': fixed_0['vertices'],
                          'faces': fixed_0['faces'],
                          'phi_3d': phi_3d_orig.detach().cpu().numpy() * 100.0,                     
                          regis_feat: tmp}
                write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/' + model_name + '/' + file[0].replace('.npy', '.DL.origin_3.vtk').split('/')[-1])
            
    return t

t = test(val_dataloader) 
print(np.asarray(t).mean())
t = test(train_dataloader) 
print(np.asarray(t).mean())

