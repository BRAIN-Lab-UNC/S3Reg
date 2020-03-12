#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:29:19 2020

@author: fenqiang
"""



import torch

import torchvision
import numpy as np
import glob
import math
import time

from utils import Get_neighs_order, get_z_weight, get_vertex_dis
from utils_vtk import read_vtk, write_vtk
from utils_torch import resampleSphereSurf, bilinearResampleSphereSurf, bilinearResampleSphereSurf_v2, getEn

from model import Unet

###########################################################
""" hyper-parameters """

device = torch.device('cuda:1') # torch.device('cpu'), or torch.device('cuda:0')
regis_feat = 'sulc' # 'sulc' or 'curv'
model_name = 'M3_regis_sulc_40962_3d_smooth10_phiconsis100_corr0p6'
n_vertex = 40962

###########################################################

bi = True
num_composition = 7
in_ch = 2   # one for sulc in fixed, one for sulc in moving
out_ch = 2  # two components for tangent plane deformation vector 
batch_size = 1
data_for_test = 0.3
max_disp = get_vertex_dis(n_vertex)/100.0 * 0.4

###########################################################
""" split files, only need 18 month now"""

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.'+str(n_vertex)+'.npy'))
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, global parameter pre-defined """


def getOverlapIndex(n_vertex, device):
    """
    Compute the overlap indices' index for the 3 deforamtion field
    """
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
    rot_mat_20 = torch.inverse(rot_mat_02)
    
    tmp = torch.cat((index_0_0, index_1_0, index_2_0))
    tmp, indices = torch.sort(tmp.squeeze())
    output, counts = torch.unique_consecutive(tmp, return_counts=True)
    assert len(output) == n_vertex, "len(output) = n_vertex, error"
    assert output[0] == 0, "output[0] = 0, error"
    assert output[-1] == n_vertex-1, "output[-1] = n_vertex-1, error"
    assert counts.max() == 3, "counts.max() == 3, error"
    assert counts.min() == 2, "counts.min() == 3, error"
    index_triple_computed = (counts == 3).nonzero().squeeze()
    tmp = np.intersect1d(index_02.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_02 = torch.from_numpy(np.setdiff1d(index_02.cpu().numpy(), index_triple_computed.cpu().numpy())).cuda(device)
    tmp = np.intersect1d(index_12.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_12 = torch.from_numpy(np.setdiff1d(index_12.cpu().numpy(), index_triple_computed.cpu().numpy())).cuda(device)
    tmp = np.intersect1d(index_01.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_01 = torch.from_numpy(np.setdiff1d(index_01.cpu().numpy(), index_triple_computed.cpu().numpy())).cuda(device)
    assert len(index_double_01) + len(index_double_12) + len(index_double_02) + len(index_triple_computed) == n_vertex, "double computed and three computed error"

    return rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, index_double_02, index_double_12, index_double_01, index_triple_computed



def get_bi_inter(n_vertex, device):
    inter_indices_0 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_0.npy')
    inter_indices_0 = torch.from_numpy(inter_indices_0.astype(np.int64)).to(device)
    inter_weights_0 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_0.npy')
    inter_weights_0 = torch.from_numpy(inter_weights_0.astype(np.float32)).to(device)
    
    inter_indices_1 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_1.npy')
    inter_indices_1 = torch.from_numpy(inter_indices_1.astype(np.int64)).to(device)
    inter_weights_1 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_1.npy')
    inter_weights_1 = torch.from_numpy(inter_weights_1.astype(np.float32)).to(device)
    
    inter_indices_2 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_2.npy')
    inter_indices_2 = torch.from_numpy(inter_indices_2.astype(np.int64)).to(device)
    inter_weights_2 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_2.npy')
    inter_weights_2 = torch.from_numpy(inter_weights_2.astype(np.float32)).to(device)
    
    return (inter_indices_0, inter_weights_0), (inter_indices_1, inter_weights_1), (inter_indices_2, inter_weights_2)


def get_latlon_img(bi_inter, feat):
    inter_indices, inter_weights = bi_inter
    width = int(np.sqrt(len(inter_indices)))
    img = torch.sum(((feat[inter_indices.flatten()]).reshape(inter_indices.shape[0], inter_indices.shape[1], feat.shape[1])) * ((inter_weights.unsqueeze(2)).repeat(1,1,feat.shape[1])), 1)
    img = img.reshape(width, width, feat.shape[1])
    
    return img


def get_atlas(n_vertex, regis_feat, device):
    fixed_0 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_0.vtk')
    
    if regis_feat == 'sulc':
        fixed_sulc = fixed_0['sulc']
    elif regis_feat == 'curv':
        fixed_sulc = fixed_0['curv']
    else:
        raise NotImplementedError('feat should be curv or sulc.')
    
    fixed_sulc_ma = fixed_sulc.max()
    fixed_sulc_mi = fixed_sulc.min()
    fixed_sulc = (fixed_sulc - fixed_sulc_mi)/(fixed_sulc_ma - fixed_sulc_mi) * 2. - 1.
    fixed_sulc = fixed_sulc[:, np.newaxis]
    fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)
    
    fixed_xyz_0 = fixed_0['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).cuda(device)
    
    fixed_1 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_1.vtk')
    fixed_xyz_1 = fixed_1['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_1 = torch.from_numpy(fixed_xyz_1.astype(np.float32)).cuda(device)
    
    fixed_2 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_2.vtk')
    fixed_xyz_2 = fixed_2['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_2 = torch.from_numpy(fixed_xyz_2.astype(np.float32)).cuda(device)
    
    return fixed_0, fixed_1, fixed_2, fixed_xyz_0, fixed_xyz_1, fixed_xyz_2, fixed_sulc, fixed_sulc_ma, fixed_sulc_mi


############################################################################

fixed_0, fixed_1, fixed_2, fixed_xyz_0, fixed_xyz_1, fixed_xyz_2, fixed_sulc, fixed_sulc_ma, fixed_sulc_mi = get_atlas(n_vertex, regis_feat, device)

grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
grad_filter[6] = -6 

ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
n_res = level-1 if level<6 else 5

neigh_orders = Get_neighs_order(0)[8-level]
neigh_orders = torch.from_numpy(neigh_orders).to(device)
assert len(neigh_orders) == n_vertex * 7, "neigh_orders wrong!"

En_0, En_1, En_2 = getEn(n_vertex, device)

bi_inter_0, bi_inter_1, bi_inter_2 = get_bi_inter(n_vertex, device)
img0 = get_latlon_img(bi_inter_0, fixed_sulc)
img1 = get_latlon_img(bi_inter_1, fixed_sulc)
img2 = get_latlon_img(bi_inter_2, fixed_sulc)

rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, index_double_02, index_double_12, index_double_01, index_triple_computed = getOverlapIndex(n_vertex, device)

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
            ma = data[:,1].max()
            mi = data[:,1].min()
            sulc = (data[:,1] - mi)/(ma - mi) * 2. - 1.
        else:
            ma = data[:,0].max()
            mi = data[:,0].min()
            sulc = (data[:,0] - mi)/(ma - mi) * 2. - 1.
        
        return sulc.astype(np.float32), file, ma, mi

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files, regis_feat)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
val_dataset = BrainSphere(test_files, regis_feat)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
model_0.to(device)
model_0.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/' + model_name + '_0.mdl'))

model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
model_1.cuda(device)
model_1.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/' + model_name + '_1.mdl'))

model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
model_2.cuda(device)
model_2.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/' + model_name + '_2.mdl'))



def diffeomorp(fixed_xyz, phi_3d, num_composition=6, bi=False, bi_inter=None, neigh_orders=None, device=None):
    if bi:
        assert bi_inter is not None, "bi_inter is None!"
        
    warped_vertices = fixed_xyz + phi_3d
    warped_vertices = warped_vertices/(torch.norm(warped_vertices, dim=1, keepdim=True).repeat(1,3))
    
    # compute exp
    for i in range(num_composition):
        if bi:
            warped_vertices = bilinearResampleSphereSurf_v2(warped_vertices, warped_vertices.clone(), bi_inter)
        else:
            warped_vertices = resampleSphereSurf(fixed_xyz, warped_vertices, warped_vertices, neigh_orders, device)
        
        warped_vertices = warped_vertices/(torch.norm(warped_vertices, dim=1, keepdim=True).repeat(1,3))
    
    return warped_vertices


def convert2DTo3D(phi_2d, En):
    """
    phi_2d: N*2
    En: N*6
    """
    phi_3d = torch.zeros(len(En), 3).to(device)
    tmp = En * phi_2d.repeat(1,3)
    phi_3d[:,0] = tmp[:,0] + tmp[:,1]
    phi_3d[:,1] = tmp[:,2] + tmp[:,3]
    phi_3d[:,2] = tmp[:,4] + tmp[:,5]
    return phi_3d


def test(dataloader):
    
    model_0.eval()
    model_1.eval()
    model_2.eval()
    
    t = []
    with torch.no_grad():
        
        for batch_idx, (moving_0, file, ma, mi) in enumerate(dataloader):
            
            t0 = time.time()
            
            moving = torch.transpose(moving_0, 0, 1).to(device)
            data = torch.cat((moving, fixed_sulc), 1)
            
            # registration field phi
            phi_2d_0_orig = model_0(data)/5.0
            phi_2d_1_orig = model_1(data)/5.0
            phi_2d_2_orig = model_2(data)/5.0
            
            phi_3d_0_orig = convert2DTo3D(phi_2d_0_orig, En_0)
            phi_3d_1_orig = convert2DTo3D(phi_2d_1_orig, En_1)
            phi_3d_2_orig = convert2DTo3D(phi_2d_2_orig, En_2)
            
            # diffeomorphic
            phi_2d_0 = phi_2d_0_orig/math.pow(2,num_composition)
            phi_2d_1 = phi_2d_1_orig/math.pow(2,num_composition)
            phi_2d_2 = phi_2d_2_orig/math.pow(2,num_composition)
            
            # truncate
            tmp = torch.norm(phi_2d_0, dim=1) > max_disp
            phi_2d_0_0 = phi_2d_0.clone()
            phi_2d_0_0[tmp] = phi_2d_0[tmp] / (torch.norm(phi_2d_0[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            phi_2d_0 = phi_2d_0_0
            
            tmp = torch.norm(phi_2d_1, dim=1) > max_disp
            phi_2d_1_1 = phi_2d_1.clone()
            phi_2d_1_1[tmp] = phi_2d_1[tmp] / (torch.norm(phi_2d_1[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            phi_2d_1 = phi_2d_1_1
            
            tmp = torch.norm(phi_2d_2, dim=1) > max_disp
            phi_2d_2_2 = phi_2d_2.clone()
            phi_2d_2_2[tmp] = phi_2d_2[tmp] / (torch.norm(phi_2d_2[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            phi_2d_2 = phi_2d_2_2
            
            # convert 2d to 3d
            phi_3d_0 = convert2DTo3D(phi_2d_0, En_0)
            phi_3d_1 = convert2DTo3D(phi_2d_1, En_1)
            phi_3d_2 = convert2DTo3D(phi_2d_2, En_2)
            
            """ deformation consistency  """
            phi_3d_0_to_1 = torch.mm(rot_mat_01, torch.transpose(phi_3d_0, 0, 1))
            phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
            phi_3d_1_to_2 = torch.mm(rot_mat_12, torch.transpose(phi_3d_1, 0, 1))
            phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
            phi_3d_0_to_2 = torch.mm(rot_mat_02, torch.transpose(phi_3d_0, 0, 1))
            phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)
            
            """ warp moving image """
            moving_warp_phi_3d_0 = diffeomorp(fixed_xyz_0, phi_3d_0, num_composition=num_composition, bi=bi, bi_inter=bi_inter_0, neigh_orders=neigh_orders, device=device)
            moving_warp_phi_3d_1 = diffeomorp(fixed_xyz_1, phi_3d_1, num_composition=num_composition, bi=bi, bi_inter=bi_inter_1, neigh_orders=neigh_orders, device=device)
            moving_warp_phi_3d_2 = diffeomorp(fixed_xyz_2, phi_3d_2, num_composition=num_composition, bi=bi, bi_inter=bi_inter_2, neigh_orders=neigh_orders, device=device)
            
            # get defrom from warped_vertices 
            diffe_deform_0 = 1.0/torch.sum(fixed_xyz_0 * moving_warp_phi_3d_0, 1).unsqueeze(1) * moving_warp_phi_3d_0 - fixed_xyz_0
            diffe_deform_1 = 1.0/torch.sum(fixed_xyz_1 * moving_warp_phi_3d_1, 1).unsqueeze(1) * moving_warp_phi_3d_1 - fixed_xyz_1
            diffe_deform_2 = 1.0/torch.sum(fixed_xyz_2 * moving_warp_phi_3d_2, 1).unsqueeze(1) * moving_warp_phi_3d_2 - fixed_xyz_2
            
            # merge 
            phi_3d = torch.zeros(len(En_0), 3).cuda(device)
            phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2[index_double_02])/2.0
            phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2[index_double_12])/2.0
            tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1[index_double_01])/2.0
            phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp,0,1)), 0, 1)
            phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + phi_3d_2[index_triple_computed] + phi_3d_0_to_2[index_triple_computed])/3.0
            phi_3d = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)
            
            moving_warp_phi_3d = diffeomorp(fixed_xyz_0, phi_3d, num_composition=num_composition, bi=bi, bi_inter=bi_inter_0, neigh_orders=neigh_orders, device=device)
            diffe_deform_3 = 1.0/torch.sum(fixed_xyz_0 * moving_warp_phi_3d, 1).unsqueeze(1) * moving_warp_phi_3d - fixed_xyz_0            
            
            t1 = time.time()
            print((t1-t0)*1000, "ms")
            t.append((t1-t0)*1000)
            # save
            moved = {'vertices': moving_warp_phi_3d_0.detach().cpu().numpy()*100.0,
                    'faces': fixed_0['faces'],
                     regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_0.vtk')
            origin = {'vertices': fixed_0['vertices'],
                     'faces': fixed_0['faces'],
                     'velocity': phi_3d_0.detach().cpu().numpy() * 100.0,
                     'deformation': phi_3d_0_orig.detach().cpu().numpy() * 100.0,
                     'diffe_deform': diffe_deform_0.detach().cpu().numpy() * 100.0,
                     regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_0.vtk')
            
            moved = {'vertices': moving_warp_phi_3d_1.detach().cpu().numpy()*100.0,
                    'faces': fixed_1['faces'],
                    regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_1.vtk')
            origin = {'vertices': fixed_1['vertices'],
                     'faces': fixed_1['faces'],
                     'velocity': phi_3d_1.detach().cpu().numpy() * 100.0,
                     'deformation': phi_3d_1_orig.detach().cpu().numpy() * 100.0,
                     'diffe_deform': diffe_deform_1.detach().cpu().numpy() * 100.0,
                    regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_1.vtk')
            
            moved = {'vertices': moving_warp_phi_3d_2.detach().cpu().numpy()*100.0,
                    'faces': fixed_2['faces'],
                    regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_2.vtk')
            origin = {'vertices': fixed_2['vertices'],
                     'faces': fixed_2['faces'],
                     'velocity': phi_3d_2.detach().cpu().numpy() * 100.0,
                     'deformation': phi_3d_2_orig.detach().cpu().numpy() * 100.0,
                     'diffe_deform': diffe_deform_2.detach().cpu().numpy() * 100.0,
                    regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_2.vtk')
            
            # Combine all phi_3d together
            moved = {'vertices': moving_warp_phi_3d.detach().cpu().numpy()*100.0,
                    'faces': fixed_0['faces'],
                    regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_3.vtk')
            origin = {'vertices': fixed_0['vertices'],
                     'faces': fixed_0['faces'],
                     'velocity': phi_3d.detach().cpu().numpy() * 100.0,
                     'diffe_deform': diffe_deform_3.detach().cpu().numpy() * 100.0,
                    regis_feat: (moving.detach().cpu().numpy() + 1.) / 2.* (ma.item() - mi[0].item()) + mi[0].item()}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_3.vtk')
            
    return t
            
            
#            loss_l1 = torch.mean(torch.abs(fixed_inter_0 - moving) * z_weight_0.unsqueeze(1)) + \
#                      torch.mean(torch.abs(fixed_inter_1 - moving) * z_weight_1.unsqueeze(1)) + \
#                      torch.mean(torch.abs(fixed_inter_2 - moving) * z_weight_2.unsqueeze(1))
#                      
#            loss_corr = 1 - ((fixed_inter_0[index_0_0.squeeze()] - fixed_inter_0[index_0_0.squeeze()].mean()) * (moving[index_0_0.squeeze()] - moving[index_0_0.squeeze()].mean())).mean() / fixed_inter_0[index_0_0.squeeze()].std() / moving[index_0_0.squeeze()].std() + \
#                        1 - ((fixed_inter_1[index_1_0.squeeze()] - fixed_inter_1[index_1_0.squeeze()].mean()) * (moving[index_1_0.squeeze()] - moving[index_1_0.squeeze()].mean())).mean() / fixed_inter_1[index_1_0.squeeze()].std() / moving[index_1_0.squeeze()].std() + \
#                        1 - ((fixed_inter_2[index_2_0.squeeze()] - fixed_inter_2[index_2_0.squeeze()].mean()) * (moving[index_2_0.squeeze()] - moving[index_2_0.squeeze()].mean())).mean() / fixed_inter_2[index_2_0.squeeze()].std() / moving[index_2_0.squeeze()].std()
#                        
#            loss_l2 = torch.mean((fixed_inter_0 - moving)**2 * z_weight_0.unsqueeze(1)) + \
#                      torch.mean((fixed_inter_1 - moving)**2 * z_weight_1.unsqueeze(1)) + \
#                      torch.mean((fixed_inter_2 - moving)**2 * z_weight_2.unsqueeze(1))
#                      
#            tmp_0 = torch.abs(torch.mm(phi_3d_0_orig[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1) + \
#                    torch.abs(torch.mm(phi_3d_0_orig[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1) + \
#                    torch.abs(torch.mm(phi_3d_0_orig[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1)
#            tmp_1 = torch.abs(torch.mm(phi_3d_1_orig[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1) + \
#                    torch.abs(torch.mm(phi_3d_1_orig[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1) + \
#                    torch.abs(torch.mm(phi_3d_1_orig[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1)
#            tmp_2 = torch.abs(torch.mm(phi_3d_2_orig[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1) + \
#                    torch.abs(torch.mm(phi_3d_2_orig[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1) + \
#                    torch.abs(torch.mm(phi_3d_2_orig[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1)
#            loss_smooth = torch.mean(tmp_0) + torch.mean(tmp_1) + torch.mean(tmp_2)
#            
#            loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[index_01] - phi_3d_1[index_01])) + \
#                                   torch.mean(torch.abs(phi_3d_1_to_2[index_12] - phi_3d_2[index_12])) + \
#                                   torch.mean(torch.abs(phi_3d_0_to_2[index_02] - phi_3d_2[index_02]))
             

t = test(val_dataloader) 
print(np.asarray(t).mean())
t = test(train_dataloader) 
print(np.asarray(t).mean())
