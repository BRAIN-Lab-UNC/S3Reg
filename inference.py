#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:30:50 2020

@author: fenqiang
"""

import torch

import torchvision
import numpy as np
import glob
import argparse
import math 

from utils_interpolation import resampleSphereSurf, bilinearResampleSphereSurf
from utils import get_orthonormal_vectors, get_z_weight, get_vertex_dis
from utils_vtk import read_vtk, write_vtk, resample_label
from utils_torch import getEn

from model import Unet
import time


def get_bi_inter(n_vertex):
    inter_indices = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'.npy')
    inter_weights = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'.npy')
    
    return inter_indices, inter_weights

def get_trained_model(n_vertex, device):
    in_ch=2
    out_ch=2
    ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
    level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
    n_res = level-1 if level<6 else 5
    
    model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
    model_0.cuda(device)
    model_0.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/model/regis_'+str(n_vertex)+'_0.mdl'))
    
    model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
    model_1.cuda(device)
    model_1.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/model/regis_'+str(n_vertex)+'_1.mdl'))
    
    model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
    model_2.cuda(device)
    model_2.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/model/regis_'+str(n_vertex)+'_2.mdl'))
    
    return model_0, model_1, model_2
    

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


def inferDeformation(model, En, merge_index, moving, fixed_sulc, device, truncated=False, diffe=False, num_composition=3):
    """
    infer the deformation from the 3 model framework
    
    infer 3 phis  642 : 15.1519775390625 ms
    compute 3d phi from 2d phi  642 : 57.082176208496094 ms
    rotate 3d phi 642 : 0.19431114196777344 ms
    merge 3 3d phi 642 : 0.9906291961669922 ms
    infer 3 phis  2562 : 17.673492431640625 ms
    compute 3d phi from 2d phi  2562 : 230.2684783935547 ms
    rotate 3d phi 2562 : 0.5338191986083984 ms
    merge 3 3d phi 2562 : 0.8721351623535156 ms
    infer 3 phis  10242 : 37.55640983581543 ms
    truncated  10242 : 2.7837753295898438 ms
    compute 3d phi from 2d phi  10242 : 900.9714126586914 ms
    rotate 3d phi 10242 : 0.4401206970214844 ms
    merge 3 3d phi 10242 : 0.8537769317626953 ms
    
    """
    assert False in [truncated, diffe], "Need SD or MSM implementation for diffeomorphic, not both!"
    
    n_vertex = len(moving) 
    
    model_0, model_1, model_2 = model
    model_0.eval()
    model_1.eval()
    model_2.eval()
    
    rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, index_double_02, index_double_12, index_double_01, index_triple_computed = merge_index
    En_0, En_1, En_2 = En
    
    moving = torch.from_numpy(moving.astype(np.float32)).cuda(device)
    fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)
    
    with torch.no_grad():
        data = torch.cat((moving.unsqueeze(1), fixed_sulc.unsqueeze(1)), 1)
    
        # registration field phi
        phi_2d_0 = model_0(data)
        phi_2d_1 = model_1(data)
        phi_2d_2 = model_2(data)
        
        if truncated:
            max_disp = get_vertex_dis(n_vertex)/100.0*0.45
            tmp = torch.norm(phi_2d_0, dim=1) > max_disp
            phi_2d_0[tmp] = phi_2d_0[tmp] / (torch.norm(phi_2d_0[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            tmp = torch.norm(phi_2d_1, dim=1) > max_disp
            phi_2d_1[tmp] = phi_2d_1[tmp] / (torch.norm(phi_2d_1[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            tmp = torch.norm(phi_2d_2, dim=1) > max_disp
            phi_2d_2[tmp] = phi_2d_2[tmp] / (torch.norm(phi_2d_2[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            
        # diffeomorphic implementation, divied by 2^n_steps
        if diffe:
            phi_2d_0 = phi_2d_0/math.pow(2,num_composition)
            phi_2d_1 = phi_2d_1/math.pow(2,num_composition)
            phi_2d_2 = phi_2d_2/math.pow(2,num_composition)
            
        phi_3d_0 = torch.zeros(3, len(En_0)).cuda(device)
        for j in range(len(En_0)):
            phi_3d_0[:,j] = torch.squeeze(torch.mm(En_0[j,:,:], torch.unsqueeze(phi_2d_0[j,:],1)))
        phi_3d_1 = torch.zeros(3, len(En_1)).cuda(device)
        for j in range(len(En_1)):
            phi_3d_1[:,j] = torch.squeeze(torch.mm(En_1[j,:,:], torch.unsqueeze(phi_2d_1[j,:],1)))
        phi_3d_2 = torch.zeros(3, len(En_2)).cuda(device)
        for j in range(len(En_2)):
            phi_3d_2[:,j] = torch.squeeze(torch.mm(En_2[j,:,:], torch.unsqueeze(phi_2d_2[j,:],1)))
    
        phi_3d_0_to_1 = torch.mm(rot_mat_01, phi_3d_0)
        phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
        phi_3d_1_to_2 = torch.mm(rot_mat_12, phi_3d_1)
        phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
        phi_3d_0_to_2 = torch.mm(rot_mat_02, phi_3d_0)
        phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)
        
        phi_3d_0 = torch.transpose(phi_3d_0, 0, 1)
        phi_3d_1 = torch.transpose(phi_3d_1, 0, 1)
        phi_3d_2 = torch.transpose(phi_3d_2, 0, 1)
        
        phi_3d = torch.zeros(len(En_0), 3).cuda(device)
        phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2[index_double_02])/2.0
        phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2[index_double_12])/2.0
        tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1[index_double_01])/2.0
        phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp,0,1)), 0, 1)
        phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + phi_3d_2[index_triple_computed] + phi_3d_0_to_2[index_triple_computed])/3.0
        phi_3d = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)

        return phi_3d.detach().cpu().numpy()
            
        
def diffeomorp(phi, fixed_xyz, diffe_iter, bi=False, bi_inter=None):
    warped_vertices = fixed_xyz + phi
    warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
    # compute exp
    for i in range(diffe_iter):
        if bi == True:
            warped_vertices = bilinearResampleSphereSurf(fixed_xyz, warped_vertices, warped_vertices, bi_inter)
        else:
            warped_vertices = resampleSphereSurf(fixed_xyz, warped_vertices, warped_vertices)
        
        warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
    # get defrom from warped_vertices 
    tmp = 1.0/np.sum(np.multiply(fixed_xyz, warped_vertices), 1)[:,np.newaxis] * warped_vertices

    return tmp - fixed_xyz

    
def projectOntoTangentPlane(curr_vertices, phi):
    # First find unit normal vectors of curr_vertices
    unit_normal = curr_vertices/np.linalg.norm(curr_vertices, axis=1)[:,np.newaxis]
    
    #Now, find projection of phi onto normal vectors
    projected_phi = unit_normal * np.sum(np.multiply(curr_vertices, phi), 1)[:,np.newaxis]
    phi = phi - projected_phi

    return phi


def inferTotalDeform(model, En, merge_index, n_vertex, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=False, diffe=False, diffe_iter=2, bi=False, bi_inter=None):
    if len(total_deform) < n_vertex:
        total_deform = resampleSphereSurf(fixed_xyz[0:len(total_deform),:], fixed_xyz[0:n_vertex,:], total_deform)
        total_deform = projectOntoTangentPlane(fixed_xyz[0:n_vertex,:], total_deform)
        
    moving_warp_phi = fixed_xyz[0:n_vertex,:] + total_deform
    moving_warp_phi = moving_warp_phi/np.linalg.norm(moving_warp_phi, axis=1)[:,np.newaxis]
    moving_warp_phi_resample = resampleSphereSurf(moving_warp_phi, fixed_xyz[0:n_vertex,:], moving_sulc[0:n_vertex])
    phi = inferDeformation(model, En, merge_index, moving_warp_phi_resample, fixed_sulc[0:n_vertex], device, truncated=truncated, diffe=diffe)
    
    if diffe == True:
        phi = diffeomorp(phi, fixed_xyz[0:n_vertex,:], diffe_iter, bi=bi, bi_inter=bi_inter)
    
    phi_reinterpo = resampleSphereSurf(fixed_xyz[0:n_vertex,:], moving_warp_phi, phi)
    phi_reinterpo = projectOntoTangentPlane(moving_warp_phi, phi_reinterpo)
    
    total_deform = total_deform + phi_reinterpo
    total_deform = projectOntoTangentPlane(fixed_xyz[0:n_vertex,:], total_deform)
    
    return total_deform


#file_name = '/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP000178_494/MNBCP000178_494.lh.SphereSurf.Orig.sphere.resampled.163842.vtk'
#atlas_name = '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk'


device = torch.device('cuda:1')

model_642 = get_trained_model(642, device)
model_2562 = get_trained_model(2562, device)
model_10242 = get_trained_model(10242, device)
model_40962 = get_trained_model(40962, device)

En_642 = getEn(642, device)
En_2562 = getEn(2562, device)
En_10242 = getEn(10242, device)
En_40962 = getEn(40962, device)

merge_index_642 = getOverlapIndex(642, device)
merge_index_2562 = getOverlapIndex(2562, device)
merge_index_10242 = getOverlapIndex(10242, device)
merge_index_40962 = getOverlapIndex(40962, device)

bi_inter_40962 = get_bi_inter(40962)


def registerToAtlas(file_name, atlas_name):
    """
    load model:  15757.77006149292 ms
    read surface:  572.8440284729004 ms
    read atlas:  486.020565032959 ms
    
    total: 30s
    """
    
    t0 = time.time()
    
    surf_163842 = read_vtk(file_name)
    moving_sulc = surf_163842['sulc']
    moving_curv = surf_163842['curv']
    moving_sulc = (moving_sulc+11.5)/(13.65+11.5)
    moving_curv = (moving_curv+2.32)/(2.08+2.32)
    
    atlas = read_vtk(atlas_name)
    fixed_sulc = atlas['sulc']
    fixed_curv = atlas['curv']
    fixed_sulc = (fixed_sulc+11.5)/(13.65+11.5)
    fixed_curv = (fixed_curv+2.32)/(2.08+2.32)
    fixed_xyz = atlas['vertices']/100.0
    
    # coarse registration on 642 vertices
    total_deform = inferDeformation(model_642, En_642, merge_index_642, moving_sulc[0:642], fixed_sulc[0:642], device)
   
    # reigs on 2562 vertices
    total_deform = inferTotalDeform(model_2562, En_2562, merge_index_2562, 2562, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=False)
    
    # reigs on 10242 vertices
    total_deform = inferTotalDeform(model_10242, En_10242, merge_index_10242, 10242, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=True)
    total_deform = inferTotalDeform(model_10242, En_10242, merge_index_10242, 10242, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=True)
#    total_deform = inferTotalDeform(model_10242, En_10242, merge_index_10242, 10242, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=True)
    
    # reigs on 40962 vertices
    total_deform = inferTotalDeform(model_40962, En_40962, merge_index_40962, 40962, total_deform, fixed_xyz, moving_curv, fixed_curv, device, truncated=False, diffe=True, diffe_iter=3, bi=True, bi_inter=bi_inter_40962)
    
    total_deform = resampleSphereSurf(fixed_xyz[0:40962,:], fixed_xyz, total_deform)
    total_deform = projectOntoTangentPlane(fixed_xyz, total_deform)
    moving_warp_phi_163842 = fixed_xyz + total_deform
    moving_warp_phi_163842 = moving_warp_phi_163842/np.linalg.norm(moving_warp_phi_163842, axis=1)[:,np.newaxis]
    
    moved = {'vertices': moving_warp_phi_163842 * 100.0,
             'faces': surf_163842['faces'],
             'curv': moving_curv * (2.08+2.32) - 2.32,
             'sulc': moving_sulc * (13.65+11.5) - 11.5}
    if 'par_vec' in surf_163842.keys():
        moved['par_vec'] = surf_163842['par_vec']
    write_vtk(moved, file_name[:-3]+'moved.vtk')
    
    t1 = time.time()
    
    surf_163842['deformation'] = total_deform * 100.0
    write_vtk(surf_163842, file_name[:-3]+'deform.vtk')
     
    moved_resample_sulc_curv = resampleSphereSurf(moving_warp_phi_163842, fixed_xyz, np.hstack((moved['sulc'][:,np.newaxis], moved['curv'][:,np.newaxis])))
    moved_resample = {'vertices': surf_163842['vertices'],
                      'faces': surf_163842['faces'],
                      'sulc': moved_resample_sulc_curv[:,0],
                      'curv': moved_resample_sulc_curv[:,1]}
    if 'par_vec' in surf_163842.keys():
        resample_lbl = resample_label(moving_warp_phi_163842, fixed_xyz, surf_163842['par_vec'])
        moved_resample['par_vec'] = resample_lbl
    write_vtk(moved_resample, file_name[:-3]+'moved.resampled.163842.vtk')
    
    return (t1-t0)
    

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

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.163842.vtk'))
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

cost = np.zeros(len(files))
for i in range(len(files)):
    print(i)
    file = files[i]
    cost[i] = registerToAtlas(file, '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')
    print(cost[i], "s")
