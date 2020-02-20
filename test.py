#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:16:01 2019

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob

from utils_torch import resampleSphereSurf
from utils import Get_neighs_order, get_orthonormal_vectors, get_z_weight, get_vertex_dis
from utils_vtk import read_vtk, write_vtk

from model import Unet

###########################################################
""" hyper-parameters """

in_ch = 2   # one for sulc in fixed, one for sulc in moving
out_ch = 2  # two components for tangent plane deformation vector 
device = torch.device('cuda:0')
batch_size = 1
data_for_test = 0.3
model_name = 'regis_sulc_10242_3d_smooth0p8_phiconsis1_3model'
truncated = True

n_vertex = int(model_name.split('_')[2])

###########################################################
""" split files, only need 18 month now"""

#files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.'+str(n_vertex)+'.npy'))
#files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

# training 2562, interpolated from 642 for next level training
#files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_642_3d_smooth0p4_phiconsis0p6_3model/training_2562/*sucu_resampled.2562.npy'))
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_2562_3d_smooth0p33_phiconsis1_3model/training_10242/*sucu_resampled.10242.npy'))

test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, global parameter pre-defined """

fix_vertex_dis = get_vertex_dis(n_vertex)/100.0
max_disp = 0.5*fix_vertex_dis

fixed_0 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_0.vtk')
fixed_sulc = (fixed_0['sulc'] + 11.5)/(13.65+11.5)
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
index_double_computed = (counts == 2).nonzero().squeeze()
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

En_0 = get_orthonormal_vectors(n_vertex, rotated=0)
En_0 = torch.from_numpy(En_0.astype(np.float32)).cuda(device)
En_1 = get_orthonormal_vectors(n_vertex, rotated=1)
En_1 = torch.from_numpy(En_1.astype(np.float32)).cuda(device)
En_2 = get_orthonormal_vectors(n_vertex, rotated=2)
En_2 = torch.from_numpy(En_2.astype(np.float32)).cuda(device)

#############################################################
class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = files

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        sulc = (data[:,1]+11.5)/(13.65+11.5)
        
        return sulc.astype(np.float32), file

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
val_dataset = BrainSphere(test_files)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
model_0.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/' + model_name + '_0.mdl'))
model_0.cuda(device)

model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
model_1.cuda(device)
model_1.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/' + model_name + '_1.mdl'))

model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
model_2.cuda(device)
model_2.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/' + model_name + '_2.mdl'))


#    dataiter = iter(train_dataloader)
#    moving, file = dataiter.next()

def test(dataloader):
    
    model_0.eval()
    model_1.eval()
    model_2.eval()
    
    mae_0 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    smooth_0 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    mae_1 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    smooth_1 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    mae_2 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    smooth_2 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    phi_consis_01 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    phi_consis_12 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)
    phi_consis_02 = torch.zeros(len(dataloader), dtype=torch.float32, device=device, requires_grad=False)

    with torch.no_grad():
        for batch_idx, (moving, file) in enumerate(dataloader):
        
            moving = torch.transpose(moving, 0, 1).cuda(device)
            data = torch.cat((moving, fixed_sulc), 1)
        
            # registration field phi
            phi_2d_0 = model_0(data)
            phi_2d_1 = model_1(data)
            phi_2d_2 = model_2(data)
    #        phi_2d = torch.tanh(phi_2d) * fix_vertex_dis
            
            if truncated:
                tmp = torch.norm(phi_2d_0, dim=1) > max_disp
                phi_2d_0[tmp] = phi_2d_0[tmp] / (torch.norm(phi_2d_0[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
                tmp = torch.norm(phi_2d_1, dim=1) > max_disp
                phi_2d_1[tmp] = phi_2d_1[tmp] / (torch.norm(phi_2d_1[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
                tmp = torch.norm(phi_2d_2, dim=1) > max_disp
                phi_2d_2[tmp] = phi_2d_2[tmp] / (torch.norm(phi_2d_2[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp

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
            
            mae_0[batch_idx] = torch.mean(torch.abs(fixed_inter_0 - moving) * z_weight_0.unsqueeze(1))
            mae_1[batch_idx] = torch.mean(torch.abs(fixed_inter_1 - moving) * z_weight_1.unsqueeze(1))
            mae_2[batch_idx] = torch.mean(torch.abs(fixed_inter_2 - moving) * z_weight_2.unsqueeze(1))
            
            smooth_0[batch_idx] = torch.mean(torch.abs(torch.mm(phi_3d_0[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_0[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_0[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)))
            smooth_1[batch_idx] = torch.mean(torch.abs(torch.mm(phi_3d_1[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_1.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_1[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_1.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_1[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_1.unsqueeze(1)))
            smooth_2[batch_idx] = torch.mean(torch.abs(torch.mm(phi_3d_2[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_2.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_2[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_2.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_2[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_2.unsqueeze(1)))
            
            phi_consis_01[batch_idx] = torch.mean(torch.abs(phi_3d_0_to_1[index_01] - phi_3d_1[index_01]))
            phi_consis_12[batch_idx] = torch.mean(torch.abs(phi_3d_1_to_2[index_12] - phi_3d_2[index_12]))
            phi_consis_02[batch_idx] = torch.mean(torch.abs(phi_3d_0_to_2[index_02] - phi_3d_2[index_02]))
            
            moved = {'vertices': moving_warp_phi_3d_0.detach().cpu().numpy()*100.0,
                    'faces': fixed_0['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_0.vtk')
            origin = {'vertices': fixed_0['vertices'],
                     'faces': fixed_0['faces'],
                     'deformation': phi_3d_0.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_0.vtk')
            
            moved = {'vertices': moving_warp_phi_3d_1.detach().cpu().numpy()*100.0,
                    'faces': fixed_1['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_1.vtk')
            origin = {'vertices': fixed_1['vertices'],
                     'faces': fixed_1['faces'],
                     'deformation': phi_3d_1.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_1.vtk')
            
            moved = {'vertices': moving_warp_phi_3d_2.detach().cpu().numpy()*100.0,
                    'faces': fixed_2['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_2.vtk')
            origin = {'vertices': fixed_2['vertices'],
                     'faces': fixed_2['faces'],
                     'deformation': phi_3d_2.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_2.vtk')
            
            # Combine all phi_3d together
            phi_3d = torch.zeros(len(En_0), 3).cuda(device)
            phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2[index_double_02])/2.0
            phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2[index_double_12])/2.0
            tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1[index_double_01])/2.0
            phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp,0,1)), 0, 1)
            phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + phi_3d_2[index_triple_computed] + phi_3d_0_to_2[index_triple_computed])/3.0
            phi_3d = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)
            
            moving_warp_phi_3d = fixed_xyz_0 + phi_3d
            moving_warp_phi_3d = moving_warp_phi_3d/(torch.norm(moving_warp_phi_3d, dim=1, keepdim=True).repeat(1,3)) # normalize the deformed vertices onto the sphere
            
            moved = {'vertices': moving_warp_phi_3d.detach().cpu().numpy()*100.0,
                    'faces': fixed_0['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[-1][:-3] + 'DL.moved_3.vtk')
            origin = {'vertices': fixed_0['vertices'],
                     'faces': fixed_0['faces'],
                     'deformation': phi_3d.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[-1][:-3] + 'DL.origin_3.vtk')
            
        
    return mae_0, smooth_0, mae_1, smooth_1, mae_2, smooth_2, phi_consis_01, phi_consis_12, phi_consis_02


train_mae_0, train_smooth_0, train_mae_1, train_smooth_1, train_mae_2, train_smooth_2, train_phi_consis_01, train_phi_consis_12, train_phi_consis_02 = test(train_dataloader) 
val_mae_0, val_smooth_0, val_mae_1, val_smooth_1, val_mae_2, val_smooth_2, test_phi_consis_01, test_phi_consis_12, test_phi_consis_02 = test(val_dataloader) 

print("train_mae_0: mean: {:.4}, std: {:.4}".format(train_mae_0.mean().item(), train_mae_0.std().item()))
print("train_mae_1: mean: {:.4}, std: {:.4}".format(train_mae_1.mean().item(), train_mae_1.std().item()))
print("train_mae_2: mean: {:.4}, std: {:.4}".format(train_mae_2.mean().item(), train_mae_2.std().item()))

print("val_mae_0: mean: {:.4}, std: {:.4}".format(val_mae_0.mean().item(), val_mae_0.std().item()))
print("val_mae_1: mean: {:.4}, std: {:.4}".format(val_mae_1.mean().item(), val_mae_1.std().item()))
print("val_mae_2: mean: {:.4}, std: {:.4}".format(val_mae_2.mean().item(), val_mae_2.std().item()))

print("train_smooth_0: mean: {:.4}, std: {:.4}".format(train_smooth_0.mean().item(), train_smooth_0.std().item()))
print("train_smooth_1: mean: {:.4}, std: {:.4}".format(train_smooth_1.mean().item(), train_smooth_1.std().item()))
print("train_smooth_2: mean: {:.4}, std: {:.4}".format(train_smooth_2.mean().item(), train_smooth_2.std().item()))

print("val_smooth_0: mean: {:.4}, std: {:.4}".format(val_smooth_0.mean().item(), val_smooth_0.std().item()))
print("val_smooth_1: mean: {:.4}, std: {:.4}".format(val_smooth_1.mean().item(), val_smooth_1.std().item()))
print("val_smooth_2: mean: {:.4}, std: {:.4}".format(val_smooth_2.mean().item(), val_smooth_2.std().item()))

print("train_phi_consis_01: mean: {:.4}, std: {:.4}".format(train_phi_consis_01.mean().item(), train_phi_consis_01.std().item()))
print("train_phi_consis_12: mean: {:.4}, std: {:.4}".format(train_phi_consis_12.mean().item(), train_phi_consis_12.std().item()))
print("train_phi_consis_02: mean: {:.4}, std: {:.4}".format(train_phi_consis_02.mean().item(), train_phi_consis_02.std().item()))

print("test_phi_consis_01: mean: {:.4}, std: {:.4}".format(test_phi_consis_01.mean().item(), test_phi_consis_01.std().item()))
print("test_phi_consis_12: mean: {:.4}, std: {:.4}".format(test_phi_consis_12.mean().item(), test_phi_consis_12.std().item()))
print("test_phi_consis_02: mean: {:.4}, std: {:.4}".format(test_phi_consis_02.mean().item(), test_phi_consis_02.std().item()))