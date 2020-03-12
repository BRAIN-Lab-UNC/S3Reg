#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:41:30 2020

@author: fenqiang
"""


import torch

import torchvision
import numpy as np
import glob
import math

from utils import Get_neighs_order, get_z_weight, get_vertex_dis
from utils_vtk import read_vtk
from utils_torch import resampleSphereSurf, bilinearResampleSphereSurf, bilinearResampleSphereSurf_v2, getEn
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/M3')

from model import Unet

###########################################################
""" hyper-parameters """

device = torch.device('cuda:0') # torch.device('cpu'), or torch.device('cuda:0')
learning_rate = 1e-3
weight_corr = 0.6
weight_smooth = 10.0
weight_l2 = 6.0
weight_l1 = 1.0
weight_phi_consis = 100.0
regis_feat = 'sulc' # 'sulc' or 'curv'
num_composition = 7

#torch.autograd.set_detect_anomaly(True)

n_vertex = 40962
bi = True

###########################################################

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

    
def get_index(n_vertex, device):
    z_weight_0 = get_z_weight(n_vertex, 0)
    z_weight_0 = torch.from_numpy(z_weight_0.astype(np.float32)).to(device)
    index_0_0 = (z_weight_0 == 1).nonzero()
    index_0_1 = (z_weight_0 < 1).nonzero()
    assert len(index_0_0) + len(index_0_1) == n_vertex, "error!"
    z_weight_1 = get_z_weight(n_vertex, 1)
    z_weight_1 = torch.from_numpy(z_weight_1.astype(np.float32)).to(device)
    index_1_0 = (z_weight_1 == 1).nonzero()
    index_1_1 = (z_weight_1 < 1).nonzero()
    assert len(index_1_0) + len(index_1_1) == n_vertex, "error!"
    z_weight_2 = get_z_weight(n_vertex, 2)
    z_weight_2 = torch.from_numpy(z_weight_2.astype(np.float32)).to(device)
    index_2_0 = (z_weight_2 == 1).nonzero()
    index_2_1 = (z_weight_2 < 1).nonzero()
    assert len(index_2_0) + len(index_2_1) == n_vertex, "error!"
    
    index_01 = np.intersect1d(index_0_0.detach().cpu().numpy(), index_1_0.detach().cpu().numpy())
    index_02 = np.intersect1d(index_0_0.detach().cpu().numpy(), index_2_0.detach().cpu().numpy())
    index_12 = np.intersect1d(index_1_0.detach().cpu().numpy(), index_2_0.detach().cpu().numpy())
    index_01 = torch.from_numpy(index_01).to(device)
    index_02 = torch.from_numpy(index_02).to(device)
    index_12 = torch.from_numpy(index_12).to(device)
    rot_mat_01 = torch.tensor([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                               [0., 1., 0.],
                               [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]]).to(device)
    rot_mat_12 = torch.tensor([[1., 0., 0.],
                               [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                               [0, np.sin(np.pi/2), np.cos(np.pi/2)]]).to(device)
    rot_mat_02 = torch.mm(rot_mat_12, rot_mat_01)
    
    return rot_mat_01, rot_mat_12, rot_mat_02, z_weight_0, z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, index_1_0, index_2_0


fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_0.vtk')
if regis_feat == 'sulc':
    fixed_sulc = (fixed['sulc'] - fixed['sulc'].min())/(fixed['sulc'].max()-fixed['sulc'].min()) * 2. - 1.
elif regis_feat == 'curv':
    fixed_sulc = (fixed['curv'] - fixed['curv'].min())/(fixed['curv'].max()-fixed['curv'].min()) * 2. - 1.
else:
    raise NotImplementedError('feat should be curv or sulc.')
fixed_sulc = fixed_sulc[:, np.newaxis]
fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).to(device)

fixed_xyz_0 = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).to(device)
fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_1.vtk')
fixed_xyz_1 = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_1 = torch.from_numpy(fixed_xyz_1.astype(np.float32)).to(device)
fixed = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_2.vtk')
fixed_xyz_2 = fixed['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_2 = torch.from_numpy(fixed_xyz_2.astype(np.float32)).to(device)

grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
grad_filter[6] = -6 

ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
n_res = level-1 if level<6 else 5

neigh_orders = Get_neighs_order(0)[8-level]
neigh_orders = torch.from_numpy(neigh_orders).to(device)
assert len(neigh_orders) == n_vertex * 7, "neigh_orders wrong!"

En_0, En_1, En_2 = getEn(n_vertex, device)

rot_mat_01, rot_mat_12, rot_mat_02, z_weight_0, z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, index_1_0, index_2_0 = get_index(n_vertex, device)
bi_inter_0, bi_inter_1, bi_inter_2 = get_bi_inter(n_vertex, device)
img0 = get_latlon_img(bi_inter_0, fixed_sulc)
img1 = get_latlon_img(bi_inter_1, fixed_sulc)
img2 = get_latlon_img(bi_inter_2, fixed_sulc)

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
            sulc = (data[:,1]-data[:,1].min())/(data[:,1].max()-data[:,1].min()) * 2. - 1.
        else:
            sulc = (data[:,0]-data[:,0].min())/(data[:,0].max()-data[:,0].min()) * 2. - 1.
        
        return sulc.astype(np.float32)

    def __len__(self):
        return len(self.files)

train_dataset = BrainSphere(train_files, regis_feat)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#val_dataset = BrainSphere(test_files)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
model_0.to(device)
#print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
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


for epoch in range(80):
    lr = get_learning_rate(epoch)
    for optimizer in optimizers:
        optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(lr))
    
#    dataiter = iter(train_dataloader)
#    moving_0 = dataiter.next()
    
    for batch_idx, (moving_0) in enumerate(train_dataloader):
        
        model_0.train()
        model_1.train()
        model_2.train()
        
        moving = torch.transpose(moving_0, 0, 1).to(device)
        data = torch.cat((moving, fixed_sulc), 1)
        
        # registration field phi
        phi_2d_0_orig = model_0(data)/5.0
        phi_2d_1_orig = model_1(data)/5.0
        phi_2d_2_orig = model_2(data)/5.0
        
        phi_3d_0_orig = convert2DTo3D(phi_2d_0_orig, En_0)
        phi_3d_1_orig = convert2DTo3D(phi_2d_1_orig, En_1)
        phi_3d_2_orig = convert2DTo3D(phi_2d_2_orig, En_2)
        
        # divide to small veloctiy field
        phi_2d_0 = phi_2d_0_orig/math.pow(2,num_composition)
        phi_2d_1 = phi_2d_1_orig/math.pow(2,num_composition)
        phi_2d_2 = phi_2d_2_orig/math.pow(2,num_composition)
        
        print(torch.norm(phi_2d_0,dim=1).max())
        print(torch.norm(phi_2d_1,dim=1).max())
        print(torch.norm(phi_2d_2,dim=1).max())
        
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
        
        
        """ compute interpolation values on fixed surface """
        if bi:
            fixed_inter_0 = bilinearResampleSphereSurf(moving_warp_phi_3d_0, img0)
            fixed_inter_1 = bilinearResampleSphereSurf(moving_warp_phi_3d_1, img1)
            fixed_inter_2 = bilinearResampleSphereSurf(moving_warp_phi_3d_2, img2)
        else:
            fixed_inter_0 = resampleSphereSurf(fixed_xyz_0, moving_warp_phi_3d_0, fixed_sulc, neigh_orders, device)
            fixed_inter_1 = resampleSphereSurf(fixed_xyz_1, moving_warp_phi_3d_1, fixed_sulc, neigh_orders, device)
            fixed_inter_2 = resampleSphereSurf(fixed_xyz_2, moving_warp_phi_3d_2, fixed_sulc, neigh_orders, device)
        
        
        loss_l1 = torch.mean(torch.abs(fixed_inter_0 - moving) * z_weight_0.unsqueeze(1)) + \
                  torch.mean(torch.abs(fixed_inter_1 - moving) * z_weight_1.unsqueeze(1)) + \
                  torch.mean(torch.abs(fixed_inter_2 - moving) * z_weight_2.unsqueeze(1))
                  
        loss_corr = 1 - ((fixed_inter_0[index_0_0.squeeze()] - fixed_inter_0[index_0_0.squeeze()].mean()) * (moving[index_0_0.squeeze()] - moving[index_0_0.squeeze()].mean())).mean() / fixed_inter_0[index_0_0.squeeze()].std() / moving[index_0_0.squeeze()].std() + \
                    1 - ((fixed_inter_1[index_1_0.squeeze()] - fixed_inter_1[index_1_0.squeeze()].mean()) * (moving[index_1_0.squeeze()] - moving[index_1_0.squeeze()].mean())).mean() / fixed_inter_1[index_1_0.squeeze()].std() / moving[index_1_0.squeeze()].std() + \
                    1 - ((fixed_inter_2[index_2_0.squeeze()] - fixed_inter_2[index_2_0.squeeze()].mean()) * (moving[index_2_0.squeeze()] - moving[index_2_0.squeeze()].mean())).mean() / fixed_inter_2[index_2_0.squeeze()].std() / moving[index_2_0.squeeze()].std()
                    
        loss_l2 = torch.mean((fixed_inter_0 - moving)**2 * z_weight_0.unsqueeze(1)) + \
                  torch.mean((fixed_inter_1 - moving)**2 * z_weight_1.unsqueeze(1)) + \
                  torch.mean((fixed_inter_2 - moving)**2 * z_weight_2.unsqueeze(1))
                  
        tmp_0 = torch.abs(torch.mm(phi_3d_0_orig[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_0_orig[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_0_orig[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_0.unsqueeze(1)
        tmp_1 = torch.abs(torch.mm(phi_3d_1_orig[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_1_orig[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_1_orig[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_1.unsqueeze(1)
        tmp_2 = torch.abs(torch.mm(phi_3d_2_orig[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_2_orig[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1) + \
                torch.abs(torch.mm(phi_3d_2_orig[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter)) * z_weight_2.unsqueeze(1)
        loss_smooth = torch.mean(tmp_0) + torch.mean(tmp_1) + torch.mean(tmp_2)
        
        loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[index_01] - phi_3d_1[index_01])) + \
                               torch.mean(torch.abs(phi_3d_1_to_2[index_12] - phi_3d_2[index_12])) + \
                               torch.mean(torch.abs(phi_3d_0_to_2[index_02] - phi_3d_2[index_02]))
         
        loss = weight_l1 * loss_l1 + weight_smooth * loss_smooth + weight_l2 * loss_l2 + weight_phi_consis * loss_phi_consistency + weight_corr * loss_corr
    
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
       
        print("[Epoch {}] [Batch {}/{}] [loss_l1: {:5.4f}] [loss_l2: {:5.4f}] [loss_corr: {:5.4f}] [loss_smooth: {:5.4f}] [loss_phi_consistency: {:5.4f}]".format(epoch, batch_idx, len(train_dataloader),
                                                            loss_l1.item(), loss_l2.item(), loss_corr.item(), loss_smooth.item(), loss_phi_consistency.item()))
        writer.add_scalars('Train/loss', {'loss_l1': loss_l1.item()*weight_l1, 'loss_l2': loss_l2.item()*weight_l2, 'loss_corr': loss_corr.item()*weight_corr, 'loss_smooth': loss_smooth.item()*weight_smooth, 'loss_phi_consistency': loss_phi_consistency.item()*weight_phi_consis}, 
                                          epoch*len(train_dataloader) + batch_idx)
    
    torch.save(model_0.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/M3_regis_"+regis_feat+"_"+str(n_vertex)+"_3d_smooth10_phiconsis100_corr0p6_0.mdl")
    torch.save(model_1.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/M3_regis_"+regis_feat+"_"+str(n_vertex)+"_3d_smooth10_phiconsis100_corr0p6_1.mdl")
    torch.save(model_2.state_dict(), "/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/M3_regis_"+regis_feat+"_"+str(n_vertex)+"_3d_smooth10_phiconsis100_corr0p6_2.mdl")
    
    
    ### FOR DEBUG
    
#    
#            if torch.any(torch.isnan(phi_2d_0_orig)):
#            print("Detecting nan value from the model 0 at {}/{} ".format(batch_idx, epoch))
#            print(file)
#            continue
#        
#        if torch.any(torch.isnan(phi_2d_1_orig)):
#            print("Detecting nan value from the model 1 at {}/{} ".format(batch_idx, epoch))
#            print(file)
#            continue
#                
#        if torch.any(torch.isnan(phi_2d_2_orig)):
#            print("Detecting nan value from the model 2 at {}/{} ".format(batch_idx, epoch))
#            print(file)
#            continue
#        
    #        a = model_0.state_dict()
#        for key in a.keys():
#            print(a[key].grad)
#        
#        a = model_1.state_dict()
#        for key in a.keys():
#            print(a[key].grad)
#                    
#        a = model_2.state_dict()
#        for key in a.keys():
#            print(a[key].grad)
        
        
#        a = []
#        for param in model_0.parameters():
#            a.append(param)
#        for k in range(len(a)):
#            if torch.any(torch.isnan(a[k].data)):
#                print("Detecting nan value in the model 0 at {} layer /{}/{} ".format(k, batch_idx, epoch))
#                    
#        a = []
#        for param in model_1.parameters():
#            a.append(param)
#        for k in range(len(a)):
#            if torch.any(torch.isnan(a[k].data)):
#                print("Detecting nan value in the model 0 at {} layer /{}/{} ".format(k, batch_idx, epoch))
#                    
#            
#        a = []
#        for param in model_2.parameters():
#            a.append(param)
#        for k in range(len(a)):
#            if torch.any(torch.isnan(a[k].data)):
#                print("Detecting nan value in the model 0 at {} layer /{}/{} ".format(k, batch_idx, epoch))
#                    
#    
#        a = model_0.state_dict(keep_vars=True)
#        for key in a.keys():
##            print(key)
##            print(a[key].grad)
#            if a[key].grad is not None:
#                if torch.any(torch.isnan(a[key].grad)):
#                    print("Detecting nan value in grad at 433 model 0: ", key)
#        
#        a = model_1.state_dict(keep_vars=True)
#        for key in a.keys():
##            print(a[key].grad)
#            if a[key].grad is not None:
#                if torch.any(torch.isnan(a[key].grad)):
#                    print("Detecting nan value in grad at 433 model 1: ", key)
#                    
#        a = model_2.state_dict(keep_vars=True)
#        for key in a.keys():
##            print(a[key].grad)
#            if a[key].grad is not None:
#                if torch.any(torch.isnan(a[key].grad)):
#                    print("Detecting nan value in grad at 433 model 2: ", key)
#        





        