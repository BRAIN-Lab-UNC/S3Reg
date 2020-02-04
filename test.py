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
import itertools

from utils import Get_neighs_order, get_orthonormal_vectors, get_z_weight
from utils_vtk import read_vtk, write_vtk

from model import Unet

###########################################################
""" hyper-parameters """

in_ch = 2   # one for sulc in fixed, one for sulc in moving
out_ch = 2  # two components for tangent plane deformation vector 
device = torch.device('cuda:1')
batch_size = 1
data_for_test = 0.3
fix_vertex_dis = 0.3
model_name = 'regis_sulc_2562_3d_smooth1_phiconsis_3model'

n_vertex = 2562

###########################################################
""" split files, only need 18 month now"""

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.'+str(n_vertex)+'.npy'))
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]
test_files = [ files[x] for x in range(int(len(files)*data_for_test)) ]
train_files = [ files[x] for x in range(int(len(files)*data_for_test), len(files)) ]

###########################################################
""" load fixed/atlas surface, smooth filter, global parameter pre-defined """

fixed_0 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'_rotated_0.vtk')
fixed_faces = fixed_0['faces']
fixed_faces = torch.from_numpy(fixed_faces[:,[1,2,3]]).cuda(device)
fixed_faces = torch.sort(fixed_faces, axis=1)[0]
fixed_sulc = (fixed_0['sulc'] + 11.5)/(13.65+11.5)
fixed_sulc = fixed_sulc[:, np.newaxis]
fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)

fixed_xyz_0 = fixed_0['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).cuda(device)

fixed_1 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'_rotated_1.vtk')
fixed_xyz_1 = fixed_1['vertices']/100.0  # fixed spherical coordinate
fixed_xyz_1 = torch.from_numpy(fixed_xyz_1.astype(np.float32)).cuda(device)

fixed_2 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'_rotated_2.vtk')
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
z_weight_1 = get_z_weight(n_vertex, 1)
z_weight_1 = torch.from_numpy(z_weight_1.astype(np.float32)).cuda(device)
z_weight_2 = get_z_weight(n_vertex, 2)
z_weight_2 = torch.from_numpy(z_weight_2.astype(np.float32)).cuda(device)

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


def sphere_interpolation_7(moving_warp_phi_3d_i, distance, fixed_xyz):
    """
    Compute the three indices and weights for sphere interpolation at given position.
    
    moving_warp_phi_3d_i: torch.tensor, size: [3]
    distance: the distance from each fiexd vertices to the interpolation position
    """
    _, top7_near_vertex_index = torch.topk(distance, 7, largest = False, sorted = False)
    candi_faces = []
    for k in itertools.combinations(top7_near_vertex_index.detach().cpu().numpy(), 3):
        tmp1, _ = torch.sort(torch.tensor(k).cuda(device))  # get the indices of the potential candidate triangles
        tmp2 = ((fixed_faces - tmp1) == 0).all(1) # find the index "True" that is a face
        if tmp2.sum() == 1:
            candi_faces.append(tmp2.nonzero().squeeze().item())
            
    orig_vertex_1 = fixed_xyz[fixed_faces[candi_faces,0]]
    orig_vertex_2 = fixed_xyz[fixed_faces[candi_faces,1]]
    orig_vertex_3 = fixed_xyz[fixed_faces[candi_faces,2]]
    edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
    edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
    faces_normal = torch.cross(edge_12, edge_13, dim=1)    # normals of all the faces
    
    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
    upper = torch.sum(orig_vertex_1 * faces_normal, axis=1)
    lower = torch.sum(moving_warp_phi_3d_i * faces_normal, axis=1)
    ratio = upper/lower
    ratio = ratio.unsqueeze(1)
    moving_warp_phi_3d_i_proj = ratio * moving_warp_phi_3d_i  # intersection points
    
    # find the triangle face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole triangle
    area_BCP = 1/2 * torch.norm(torch.cross(orig_vertex_2 - moving_warp_phi_3d_i_proj, orig_vertex_3 - moving_warp_phi_3d_i_proj), 2, dim=1)
    area_ACP = 1/2 * torch.norm(torch.cross(orig_vertex_3 - moving_warp_phi_3d_i_proj, orig_vertex_1 - moving_warp_phi_3d_i_proj), 2, dim=1)
    area_ABP = 1/2 * torch.norm(torch.cross(orig_vertex_1 - moving_warp_phi_3d_i_proj, orig_vertex_2 - moving_warp_phi_3d_i_proj), 2, dim=1)
    area_ABC = 1/2 * torch.norm(faces_normal, 2, dim=1)
    
    min_area, index = torch.min(abs(area_BCP + area_ACP + area_ABP - area_ABC),0)
    assert abs(ratio[index] - 1) < 0.01, "projected vertex should be near the vertex!" 
    assert min_area < 5e-05, "Intersection should be in the triangle"
    w = torch.stack((area_BCP[index], area_ACP[index], area_ABP[index]))
    inter_weight = w / w.sum()
    
    return fixed_faces[candi_faces[index]], inter_weight
            
def sphere_interpolation(moving_warp_phi_3d, fixed_xyz, fixed_sulc):
    fixed_inter = torch.zeros((len(moving_warp_phi_3d),1), dtype=torch.float32, device = device)
    for i in range(len(moving_warp_phi_3d)):
        moving_warp_phi_3d_i = moving_warp_phi_3d[i]
        
        """ barycentric interpolation """
        distance = torch.norm((fixed_xyz - moving_warp_phi_3d_i), 2, dim=1)
        _, top3_near_vertex_index = torch.topk(distance, 3, largest = False, sorted = False)
        top3_near_vertex_index, _ = torch.sort(top3_near_vertex_index)     
        
        if ((fixed_faces - top3_near_vertex_index) == 0).all(1).sum() == 1:
            # if the 3 nearest indices compose a triangle:
            top3_near_vertex_0 = fixed_xyz[top3_near_vertex_index[0]]
            top3_near_vertex_1 = fixed_xyz[top3_near_vertex_index[1]]
            top3_near_vertex_2 = fixed_xyz[top3_near_vertex_index[2]]
            
            # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with the triangle face
            normal = torch.cross(top3_near_vertex_0-top3_near_vertex_2, top3_near_vertex_1-top3_near_vertex_2)
            ratio = torch.dot(top3_near_vertex_0, normal)/torch.dot(moving_warp_phi_3d_i, normal)
            moving_warp_phi_3d_i_proj = ratio * moving_warp_phi_3d_i  # intersection points
            
            # compute the small triangle area and check if the intersection point is in the triangle
            area_BCP = 1/2 * torch.norm(torch.cross(top3_near_vertex_1 - moving_warp_phi_3d_i_proj, top3_near_vertex_2 - moving_warp_phi_3d_i_proj), 2)
            area_ACP = 1/2 * torch.norm(torch.cross(top3_near_vertex_2 - moving_warp_phi_3d_i_proj, top3_near_vertex_0 - moving_warp_phi_3d_i_proj), 2)
            area_ABP = 1/2 * torch.norm(torch.cross(top3_near_vertex_0 - moving_warp_phi_3d_i_proj, top3_near_vertex_1 - moving_warp_phi_3d_i_proj), 2)
            area_ABC = 1/2 * torch.norm(normal, 2)
            
            if abs(area_BCP + area_ACP + area_ABP - area_ABC) > 5e-05:
                inter_indices, inter_weight = sphere_interpolation_7(moving_warp_phi_3d_i, distance, fixed_xyz) 
            else:
                inter_weight = torch.stack((area_BCP, area_ACP, area_ABP))
                inter_weight = inter_weight / inter_weight.sum()
                inter_indices = top3_near_vertex_index
        else:
            inter_indices, inter_weight = sphere_interpolation_7(moving_warp_phi_3d_i, distance, fixed_xyz)
            
        fixed_inter[i] = torch.mm(inter_weight.unsqueeze(0), fixed_sulc[inter_indices])
    
    return fixed_inter
            

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
    
   
    with torch.no_grad():
        for batch_idx, (moving, file) in enumerate(dataloader):
        
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
            fixed_inter_0 = sphere_interpolation(moving_warp_phi_3d_0, fixed_xyz_0, fixed_sulc)
            fixed_inter_1 = sphere_interpolation(moving_warp_phi_3d_1, fixed_xyz_1, fixed_sulc)
            fixed_inter_2 = sphere_interpolation(moving_warp_phi_3d_2, fixed_xyz_2, fixed_sulc)
                    
            
            mae_0[batch_idx] = torch.mean(torch.abs(fixed_inter_0 - moving) * z_weight_0.unsqueeze(1))
            mae_1[batch_idx] = torch.mean(torch.abs(fixed_inter_1 - moving) * z_weight_1.unsqueeze(1))
            mae_2[batch_idx] = torch.mean(torch.abs(fixed_inter_2 - moving) * z_weight_2.unsqueeze(1))
            
            smooth_0[batch_idx] = torch.mean(torch.abs(torch.mm(phi_3d_0[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_0[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_0[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)))
            smooth_1[batch_idx] = torch.mean(torch.abs(torch.mm(phi_3d_1[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_1[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_1[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)))
            smooth_2[batch_idx] = torch.mean(torch.abs(torch.mm(phi_3d_2[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_2[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)) + \
                                             torch.abs(torch.mm(phi_3d_2[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter) * z_weight_0.unsqueeze(1)))
            
            moved = {'vertices': moving_warp_phi_3d_0.detach().cpu().numpy()*100.0,
                    'faces': fixed_0['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[10][:-3] + 'DL.moved_0.vtk')
            origin = {'vertices': fixed_0['vertices'],
                     'faces': fixed_0['faces'],
                     'deformation': phi_3d_0.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[10][:-3] + 'DL.origin_0.vtk')
            
            moved = {'vertices': moving_warp_phi_3d_1.detach().cpu().numpy()*100.0,
                    'faces': fixed_1['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[10][:-3] + 'DL.moved_1.vtk')
            origin = {'vertices': fixed_1['vertices'],
                     'faces': fixed_1['faces'],
                     'deformation': phi_3d_1.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[10][:-3] + 'DL.origin_1.vtk')
            
            moved = {'vertices': moving_warp_phi_3d_2.detach().cpu().numpy()*100.0,
                    'faces': fixed_2['faces'],
                    'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(moved, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name +'/' + file[0].split('/')[10][:-3] + 'DL.moved_2.vtk')
            origin = {'vertices': fixed_2['vertices'],
                     'faces': fixed_2['faces'],
                     'deformation': phi_3d_2.detach().cpu().numpy() * 100.0,
                     'sulc': moving.detach().cpu().numpy() * (13.65+11.5) - 11.5}
            write_vtk(origin, '/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/' + file[0].split('/')[10][:-3] + 'DL.origin_2.vtk')
        
    return mae_0, smooth_0, mae_1, smooth_1, mae_2, smooth_2


train_mae_0, train_smooth_0, train_mae_1, train_smooth_1, train_mae_2, train_smooth_2 = test(train_dataloader) 
val_mae_0, val_smooth_0, val_mae_1, val_smooth_1, val_mae_2, val_smooth_2 = test(val_dataloader) 

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
