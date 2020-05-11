#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:01:40 2020

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob
import argparse
import math 

from utils_interpolation import resampleSphereSurf, bilinearResampleSphereSurf
from utils import get_z_weight, get_vertex_dis, get_upsample_order, get_neighs_order
from utils_vtk import read_vtk, write_vtk, resample_label
from utils_torch import getEn, convert2DTo3D

from model import Unet
import time


def get_bi_inter(n_vertex):
    inter_indices = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_0.npy')
    inter_weights = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_0.npy')
    
    return inter_indices, inter_weights

def get_trained_model(n_vertex, device, feat='sulc'):
    in_ch=2
    out_ch=2
    ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
    level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
    n_res = level-1 if level<6 else 5
    
    model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
    model_0.to(device)
    model_0.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/final_models/regis_'+ feat + '_' + str(n_vertex)+'_0.mdl'))
    
    model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
    model_1.to(device)
    model_1.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/final_models/regis_'+ feat + '_' + str(n_vertex)+'_1.mdl'))
    
    model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
    model_2.to(device)
    model_2.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/NAMIC/trained_models/final_models/regis_'+ feat + '_' + str(n_vertex)+'_2.mdl'))
    
    return model_0, model_1, model_2
    

def getOverlapIndex(n_vertex, device):
    """
    Compute the overlap indices' index for the 3 deforamtion field
    """
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
                               [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]], dtype=torch.float).to(device)
    rot_mat_12 = torch.tensor([[1., 0., 0.],
                               [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                               [0, np.sin(np.pi/2), np.cos(np.pi/2)]], dtype=torch.float).to(device)
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
    index_double_02 = torch.from_numpy(np.setdiff1d(index_02.cpu().numpy(), index_triple_computed.cpu().numpy())).to(device)
    tmp = np.intersect1d(index_12.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_12 = torch.from_numpy(np.setdiff1d(index_12.cpu().numpy(), index_triple_computed.cpu().numpy())).to(device)
    tmp = np.intersect1d(index_01.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_01 = torch.from_numpy(np.setdiff1d(index_01.cpu().numpy(), index_triple_computed.cpu().numpy())).to(device)
    assert len(index_double_01) + len(index_double_12) + len(index_double_02) + len(index_triple_computed) == n_vertex, "double computed and three computed error"

    return rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, index_double_02, index_double_12, index_double_01, index_triple_computed


def inferDeformation(model, En, merge_index, moving, fixed_sulc, device, truncated=False, diffe=False, num_composition=6):
    """
    infer the deformation from the 3 model framework
       
    """
    assert False in [truncated, diffe], "Need SD or MSM implementation for diffeomorphic, not both!"
    
    n_vertex = len(moving)
    
    model_0, model_1, model_2 = model
    model_0.eval()
    model_1.eval()
    model_2.eval()
    
    rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, index_double_02, index_double_12, index_double_01, index_triple_computed = merge_index
    En_0, En_1, En_2 = En
    
    moving = torch.from_numpy(moving.astype(np.float32)).to(device)
    fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).to(device)
    
    with torch.no_grad():
        data = torch.cat((moving.unsqueeze(1), fixed_sulc.unsqueeze(1)), 1)
    
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
        phi_3d = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)
        
        if truncated:
            max_disp = get_vertex_dis(n_vertex)/100.0*0.4
            tmp = torch.norm(phi_3d, dim=1) > max_disp
            phi_3d_tmp = phi_3d.clone()
            phi_3d_tmp[tmp] = phi_3d[tmp] / (torch.norm(phi_3d[tmp], dim=1, keepdim=True).repeat(1,3)) * max_disp
            phi_3d = phi_3d_tmp
                
        if diffe:
            """ diffeomorphic implementation, divied by 2^n_steps, to small veloctiy field  """
            phi_3d = phi_3d/math.pow(2,num_composition)

        return phi_3d.detach().cpu().numpy()
            
        
def diffeomorp(phi, fixed_xyz, num_composition=6, bi=False, bi_inter=None):
    if bi:
        assert bi_inter is not None, "bi_inter is None!"
    
    warped_vertices = fixed_xyz + phi
    warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
    # compute exp
    for i in range(num_composition):
        if bi:
            warped_vertices = bilinearResampleSphereSurf(warped_vertices, warped_vertices, bi_inter)
        else:
            warped_vertices = resampleSphereSurf(fixed_xyz, warped_vertices, warped_vertices)
        
        warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
    # get deform from warped_vertices 
    # tmp = 1.0/np.sum(np.multiply(fixed_xyz, warped_vertices), 1)[:,np.newaxis] * warped_vertices

    return warped_vertices

    
def projectOntoTangentPlane(curr_vertices, phi):
    # First find unit normal vectors of curr_vertices
    unit_normal = curr_vertices/np.linalg.norm(curr_vertices, axis=1)[:,np.newaxis]
    
    #Now, find projection of phi onto normal vectors
    projected_phi = unit_normal * np.sum(np.multiply(curr_vertices, phi), 1)[:,np.newaxis]
    phi = phi - projected_phi

    return phi


def inferTotalDeform(model, En, merge_index, n_vertex, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=False, upsample_neighbors=None, neigh_orders=None, diffe=False, num_composition=6, bi=False, bi_inter=None):
    if bi:
        assert bi_inter is not None, "bi_inter is None!"
        
    if len(total_deform) < n_vertex:
        total_deform = resampleSphereSurf(fixed_xyz[0:len(total_deform),:], fixed_xyz[0:n_vertex,:], total_deform, std=True, upsample_neighbors=upsample_neighbors)
        total_deform = total_deform/np.linalg.norm(total_deform, axis=1)[:,np.newaxis]
        # total_deform = projectOntoTangentPlane(fixed_xyz[0:n_vertex,:], total_deform)
        
    # moving_warp_phi = fixed_xyz[0:n_vertex,:] + total_deform
    # moving_warp_phi = moving_warp_phi/np.linalg.norm(moving_warp_phi, axis=1)[:,np.newaxis]
    moving_warp_phi = total_deform
    moving_warp_phi_resample = resampleSphereSurf(moving_warp_phi, fixed_xyz[0:n_vertex,:], moving_sulc[0:n_vertex], neigh_orders=neigh_orders)
    phi = inferDeformation(model, En, merge_index, moving_warp_phi_resample, fixed_sulc[0:n_vertex], device, truncated=truncated, diffe=diffe, num_composition=num_composition)
    
    if diffe:
        warped_vertices = diffeomorp(phi, fixed_xyz[0:n_vertex,:], num_composition, bi=bi, bi_inter=bi_inter)
    else:
        warped_vertices = fixed_xyz[0:n_vertex,:] + phi
        warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
    if bi:
        total_deform = bilinearResampleSphereSurf(moving_warp_phi, warped_vertices, bi_inter)
    else:
        total_deform = resampleSphereSurf(fixed_xyz[0:n_vertex,:], moving_warp_phi, warped_vertices, neigh_orders=neigh_orders)
    
    # phi_reinterpo = projectOntoTangentPlane(moving_warp_phi, phi_reinterpo)
    # total_deform = total_deform + phi_reinterpo
    # total_deform = projectOntoTangentPlane(fixed_xyz[0:n_vertex,:], total_deform)
    total_deform = total_deform/np.linalg.norm(total_deform, axis=1)[:,np.newaxis]
    
    return total_deform


#file_name = '/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP000178_494/MNBCP000178_494.lh.SphereSurf.Orig.sphere.resampled.163842.vtk'
#atlas_name = '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk'


device = torch.device('cuda:0')

model_642 = get_trained_model(642, device)
model_2562 = get_trained_model(2562, device)
model_10242 = get_trained_model(10242, device)
model_40962_sulc = get_trained_model(40962, device, feat='sulc')
model_40962_curv = get_trained_model(40962, device, feat='curv')


En_642 = getEn(642, device)
En_2562 = getEn(2562, device)
En_10242 = getEn(10242, device)
En_40962 = getEn(40962, device)

merge_index_642 = getOverlapIndex(642, device)
merge_index_2562 = getOverlapIndex(2562, device)
merge_index_10242 = getOverlapIndex(10242, device)
merge_index_40962 = getOverlapIndex(40962, device)

bi_inter_40962 = get_bi_inter(40962)
bi_inter_10242 = get_bi_inter(10242)
bi_inter_2562 = get_bi_inter(2562)

upsample_neighbors_2562 = get_upsample_order(2562)
upsample_neighbors_10242 = get_upsample_order(10242)
upsample_neighbors_40962 = get_upsample_order(40962)
upsample_neighbors_163842 = get_upsample_order(163842)

neigh_orders_642 = get_neighs_order('neigh_indices/adj_mat_order_642_rotated_0.mat')
neigh_orders_2562 = get_neighs_order('neigh_indices/adj_mat_order_2562_rotated_0.mat')
neigh_orders_10242 = get_neighs_order('neigh_indices/adj_mat_order_10242_rotated_0.mat')
neigh_orders_40962 = get_neighs_order('neigh_indices/adj_mat_order_40962_rotated_0.mat')
neigh_orders_163842 = get_neighs_order('neigh_indices/adj_mat_order_163842_rotated_0.mat')


def registerToAtlas(file_name, atlas_name):
    """
    atlas_name = '/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.163842.rotated_0.norm.vtk'
    """
    t0 = time.time()
    
    surf_163842 = read_vtk(file_name)
    moving_sulc = surf_163842['sulc']
    moving_curv = surf_163842['curv']
    moving_sulc = (moving_sulc+11.5)/(13.65+11.5)
    moving_curv = (moving_curv+ 1.)/(1.+1.)
    
#    t1 = time.time()
#    print("read surf: ", (t1-t0)*1000, "ms")
    
    atlas = read_vtk(atlas_name)
    fixed_sulc = atlas['sulc']
    fixed_curv = atlas['curv']
    fixed_sulc = (fixed_sulc+11.5)/(13.65+11.5)
    fixed_curv = (fixed_curv+ 1.)/(1.+1.)
    fixed_xyz = atlas['vertices']/100.0
    
#    t2 = time.time()
#    print("read atlas: ", (t2-t1)*1000, "ms")
    
    
    # coarse registration on 642 vertices
    total_deform = inferDeformation(model_642, En_642, merge_index_642, moving_sulc[0:642], fixed_sulc[0:642], device, truncated=False)
    total_deform = fixed_xyz[0:642] + total_deform
    total_deform = total_deform/np.linalg.norm(total_deform, axis=1)[:,np.newaxis]
    
#    t8 = time.time()
#    print("642: ", (t8-t2)*1000, "ms")
#    
    
    # reigs on 2562 vertices
    for j in range(2):
        total_deform = inferTotalDeform(model_2562, En_2562, merge_index_2562, 2562, 
                                    total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, 
                                    truncated=False, upsample_neighbors=upsample_neighbors_2562, neigh_orders=neigh_orders_2562,
                                    diffe=False, num_composition=6, bi=True, bi_inter=bi_inter_2562)
    
#    t9 = time.time()
#    print("2562: ", (t9-t8)*1000, "ms")
    
    
    # reigs on 10242 vertices
    for j in range(3):
        total_deform = inferTotalDeform(model_10242, En_10242, merge_index_10242, 10242, 
                                    total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, 
                                    truncated=False, upsample_neighbors=upsample_neighbors_10242, neigh_orders=neigh_orders_10242, 
                                    diffe=False, num_composition=6, bi=True, bi_inter=bi_inter_10242)
    
#    t10 = time.time()
#    print("10242: ", (t10-t9)*1000, "ms")

    # reigs on 40962 vertices
    for j in range(2):
        total_deform = inferTotalDeform(model_40962_sulc, En_40962, merge_index_40962, 40962, 
                                    total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, 
                                    truncated=True, upsample_neighbors=upsample_neighbors_40962, neigh_orders=neigh_orders_40962,
                                    diffe=False, num_composition=6, bi=True, bi_inter=bi_inter_40962)
    
    for j in range(2):
        total_deform = inferTotalDeform(model_40962_curv, En_40962, merge_index_40962, 40962, 
                                    total_deform, fixed_xyz, moving_curv, fixed_curv, device, 
                                    truncated=False, upsample_neighbors=upsample_neighbors_40962, neigh_orders=neigh_orders_40962,
                                    diffe=False, num_composition=6, bi=True, bi_inter=bi_inter_40962)
    
        
#    t3 = time.time()
#    print("40962: ", (t3-t10)*1000, "ms")

    
    total_deform = resampleSphereSurf(fixed_xyz[0:40962,:], fixed_xyz, total_deform, std=True, upsample_neighbors=upsample_neighbors_163842)
    # total_deform = projectOntoTangentPlane(fixed_xyz, total_deform)
    # moving_warp_phi_163842 = fixed_xyz + total_deform
    # moving_warp_phi_163842 = moving_warp_phi_163842/np.linalg.norm(moving_warp_phi_163842, axis=1)[:,np.newaxis]
    total_deform = total_deform/np.linalg.norm(total_deform, axis=1)[:,np.newaxis]
    moving_warp_phi_163842 = total_deform
    
#    t4 = time.time()
#    print("upsample to 163842: ", (t4-t3)*1000, "ms")
    
    moved = {'vertices': moving_warp_phi_163842 * 100.0,
             'faces': atlas['faces'],
             'curv': moving_curv * (1.+1.) - 1.,
             'sulc': moving_sulc * (13.65+11.5) - 11.5}
    if 'par_vec' in surf_163842.keys():
        moved['par_vec'] = surf_163842['par_vec']
    write_vtk(moved, file_name.replace('.vtk','.moved.vtk'))
#    write_vtk(moved, '/home/fenqiang/test.vtk')
    
    t5 = time.time()
    print("inference: ", (t5-t0), "s")
    
    # surf_163842['deformation'] = total_deform * 100.0
    # write_vtk(surf_163842, file_name.replace('.vtk','.deform.vtk'))
    
    moved_resample_sulc_curv = resampleSphereSurf(moving_warp_phi_163842, fixed_xyz, np.hstack((moved['sulc'][:,np.newaxis], moved['curv'][:,np.newaxis])), neigh_orders=neigh_orders_163842)
    moved_resample = {'vertices': atlas['vertices'],
                      'faces': atlas['faces'],
                      'sulc': moved_resample_sulc_curv[:,0],
                      'curv': moved_resample_sulc_curv[:,1]}
    if 'par_vec' in surf_163842.keys():
        resample_lbl = resample_label(moving_warp_phi_163842, fixed_xyz, surf_163842['par_vec'])
        moved_resample['par_vec'] = resample_lbl
    write_vtk(moved_resample, file_name.replace('.vtk', '.moved.resampled.163842.vtk'))
    
    t6 = time.time()
    print("resample: ", (t6-t5), "s")
    
    return (t5-t0)
  

#if __name__ == "__main__":    
#    parser = argparse.ArgumentParser(description='Rigister the surface to the atlas',
#                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument('--level', '-l', default='7',
#                        choices=['7', '8'],
#                        help="Specify the level of the surfaces. Generally, level 7 spherical surface is with 40962 vertices, 8 is with 163842 vertices.")
#    parser.add_argument('--input', '-i', metavar='INPUT', default="/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP000178_494/MNBCP000178_494.lh.SphereSurf.Orig.sphere.resampled.40962.vtk",
#                        help='filename of input surface')
#    parser.add_argument('--in_folder', '-in_folder',
#                        metavar='INPUT_FOLDER',
#                        help='folder path for input files. Will parcelalte all the files end in .vtk in this folder. Accept input or in_folder.')
#    parser.add_argument('--output', '-o',  default='[input].parc.vtk', metavar='OUTPUT',
#                        help='Filename of ouput surface.')
#    parser.add_argument('--out_folder', '-out_folder', default='[in_folder]', metavar='OUT_FOLDER',
#                        help='folder path for ouput surface. Accept output or out_folder.')


#files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/*_lh.SphereSurf.Resampled160K.vtk'))

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/sub*/sub*.lh.SphereSurf.Orig.resampled.163842.vtk'))

cost = np.zeros(len(files))
for i in range(30,39):
    print(i)
    file = files[i]
    print(file)
    cost[i] = registerToAtlas(file, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.163842.rotated_0.norm.vtk')
    # print(cost[i], "s")
print("cost mean: ", cost.mean())