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

from utils_interpolation import resampleSphereSurf
from utils import Get_neighs_order, get_orthonormal_vectors, get_z_weight, get_vertex_dis
from utils_vtk import read_vtk, write_vtk

from model import Unet
import time


def get_trained_model(n_vertex, device):
    in_ch=2
    out_ch=2
    ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
    level = 8 - np.nonzero(ns_vertex-n_vertex == 0)[0][0]
    n_res = level-1 if level<6 else 5
    
    model_0 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=0)
    model_0.cuda(device)
    model_0.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/model/regis_sulc_'+str(n_vertex)+'_0.mdl'))
    
    model_1 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=1)
    model_1.cuda(device)
    model_1.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/model/regis_sulc_'+str(n_vertex)+'_1.mdl'))
    
    model_2 = Unet(in_ch=in_ch, out_ch=out_ch, level=level, n_res=n_res, rotated=2)
    model_2.cuda(device)
    model_2.load_state_dict(torch.load('/media/fenqiang/DATA/unc/Data/registration/scripts/trained_model/model/regis_sulc_'+str(n_vertex)+'_2.mdl'))
    
    return model_0, model_1, model_2
    

def getEn(n_vertex, device):
    En_0 = get_orthonormal_vectors(n_vertex, rotated=0)
    En_0 = torch.from_numpy(En_0.astype(np.float32)).cuda(device)
    En_1 = get_orthonormal_vectors(n_vertex, rotated=1)
    En_1 = torch.from_numpy(En_1.astype(np.float32)).cuda(device)
    En_2 = get_orthonormal_vectors(n_vertex, rotated=2)
    En_2 = torch.from_numpy(En_2.astype(np.float32)).cuda(device)
    return En_0, En_1, En_2


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


def inferDeformation(model_0, model_1, model_2, moving, fixed_sulc, device, truncated=False):
    """
    infer the deformation from the 3 model framework
    
    load  642  3 models:  4628.545522689819 ms
    get overlap index  642 : 22.965192794799805 ms
    get En  642 : 9.379148483276367 ms
    move data to cuda:  0.6093978881835938 ms
    infer 3 phis  642 : 20.24078369140625 ms
    truncated  642 : 0.060558319091796875 ms
    compute 3d phi from 2d phi  642 : 58.59541893005371 ms
    rotate 3d phi 642 : 0.4603862762451172 ms
    merge 3 3d phi 642 : 0.7610321044921875 ms
    load  2562  3 models:  4687.880277633667 ms
    get overlap index  2562 : 49.8812198638916 ms
    get En  2562 : 27.04596519470215 ms
    move data to cuda:  0.5090236663818359 ms
    infer 3 phis  2562 : 23.22864532470703 ms
    truncated  2562 : 0.1430511474609375 ms
    compute 3d phi from 2d phi  2562 : 234.86733436584473 ms
    rotate 3d phi 2562 : 0.49304962158203125 ms
    merge 3 3d phi 2562 : 0.8366107940673828 ms
    load  10242  3 models:  4779.069900512695 ms
    get overlap index  10242 : 115.13924598693848 ms
    get En  10242 : 127.32338905334473 ms
    move data to cuda:  0.3612041473388672 ms
    infer 3 phis  10242 : 20.565509796142578 ms
    truncated  10242 : 1.8954277038574219 ms
    compute 3d phi from 2d phi  10242 : 934.7960948944092 ms
    rotate 3d phi 10242 : 0.6048679351806641 ms
    merge 3 3d phi 10242 : 0.9973049163818359 ms
    load  10242  3 models:  4732.393026351929 ms
    get overlap index  10242 : 114.3648624420166 ms
    get En  10242 : 126.28173828125 ms
    move data to cuda:  0.5986690521240234 ms
    infer 3 phis  10242 : 29.126405715942383 ms
    truncated  10242 : 2.161264419555664 ms
    compute 3d phi from 2d phi  10242 : 934.9527359008789 ms
    rotate 3d phi 10242 : 0.48804283142089844 ms
    merge 3 3d phi 10242 : 0.7569789886474609 ms
    load  10242  3 models:  4764.239311218262 ms
    get overlap index  10242 : 116.48797988891602 ms
    get En  10242 : 104.98881340026855 ms
    move data to cuda:  0.6291866302490234 ms
    infer 3 phis  10242 : 38.08140754699707 ms
    truncated  10242 : 2.8259754180908203 ms
    compute 3d phi from 2d phi  10242 : 927.3874759674072 ms
    rotate 3d phi 10242 : 0.5834102630615234 ms
    merge 3 3d phi 10242 : 1.2521743774414062 ms
    
    
    """
    n_vertex = len(moving) 
    
    t0 = time.time()

    

    model_0.eval()
    model_1.eval()
    model_2.eval()
    
    t1 = time.time()
    print("load ", n_vertex, " 3 models: ", (t1-t0)*1000, "ms")
    
    
    rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, index_double_02, index_double_12, index_double_01, index_triple_computed = getOverlapIndex(n_vertex, device)
    
    t2 = time.time()
    print("get overlap index ", n_vertex, ":", (t2-t1)*1000, "ms")
    
    
    En_0, En_1, En_2 = getEn(n_vertex, device)
    
    t3 = time.time()
    print("get En ", n_vertex, ":", (t3-t2)*1000, "ms")
    
    
    moving = torch.from_numpy(moving.astype(np.float32)).cuda(device)
    fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).cuda(device)
    
    t4 = time.time()
    print("move data to cuda: ", (t4-t3)*1000, "ms")
    
    
    with torch.no_grad():
        data = torch.cat((moving.unsqueeze(1), fixed_sulc.unsqueeze(1)), 1)
    
        # registration field phi
        phi_2d_0 = model_0(data)
        phi_2d_1 = model_1(data)
        phi_2d_2 = model_2(data)
        
        
        t5 = time.time()
        print("infer 3 phis ", n_vertex, ":",  (t5-t4)*1000, "ms")
    

        if truncated:
            max_disp = get_vertex_dis(n_vertex)/100.0*0.45
            tmp = torch.norm(phi_2d_0, dim=1) > max_disp
            phi_2d_0[tmp] = phi_2d_0[tmp] / (torch.norm(phi_2d_0[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            tmp = torch.norm(phi_2d_1, dim=1) > max_disp
            phi_2d_1[tmp] = phi_2d_1[tmp] / (torch.norm(phi_2d_1[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            tmp = torch.norm(phi_2d_2, dim=1) > max_disp
            phi_2d_2[tmp] = phi_2d_2[tmp] / (torch.norm(phi_2d_2[tmp], dim=1, keepdim=True).repeat(1,2)) * max_disp
            
            
        t6 = time.time()
        print("truncated ", n_vertex, ":",  (t6-t5)*1000, "ms")
    
        
        phi_3d_0 = torch.zeros(3, len(En_0)).cuda(device)
        for j in range(len(En_0)):
            phi_3d_0[:,j] = torch.squeeze(torch.mm(En_0[j,:,:], torch.unsqueeze(phi_2d_0[j,:],1)))
        phi_3d_1 = torch.zeros(3, len(En_1)).cuda(device)
        for j in range(len(En_1)):
            phi_3d_1[:,j] = torch.squeeze(torch.mm(En_1[j,:,:], torch.unsqueeze(phi_2d_1[j,:],1)))
        phi_3d_2 = torch.zeros(3, len(En_2)).cuda(device)
        for j in range(len(En_2)):
            phi_3d_2[:,j] = torch.squeeze(torch.mm(En_2[j,:,:], torch.unsqueeze(phi_2d_2[j,:],1)))
    
    
        t7 = time.time()
        print("compute 3d phi from 2d phi ", n_vertex, ":",  (t7-t6)*1000, "ms")
    
    
    
        phi_3d_0_to_1 = torch.mm(rot_mat_01, phi_3d_0)
        phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
        phi_3d_1_to_2 = torch.mm(rot_mat_12, phi_3d_1)
        phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
        phi_3d_0_to_2 = torch.mm(rot_mat_02, phi_3d_0)
        phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)
        
        phi_3d_0 = torch.transpose(phi_3d_0, 0, 1)
        phi_3d_1 = torch.transpose(phi_3d_1, 0, 1)
        phi_3d_2 = torch.transpose(phi_3d_2, 0, 1)
        
        
        t8 = time.time()
        print("rotate 3d phi", n_vertex, ":",  (t8-t7)*1000, "ms")
    
        
        phi_3d = torch.zeros(len(En_0), 3).cuda(device)
        phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2[index_double_02])/2.0
        phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2[index_double_12])/2.0
        tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1[index_double_01])/2.0
        phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp,0,1)), 0, 1)
        phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + phi_3d_2[index_triple_computed] + phi_3d_0_to_2[index_triple_computed])/3.0
        phi_3d = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)
        
        
        t9 = time.time()
        print("merge 3 3d phi", n_vertex, ":",  (t9-t8)*1000, "ms")
    
        return phi_3d.detach().cpu().numpy()
            
        

def inferTotalDeform(model_0, model_1, model_2, n_vertex, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=False):
    if len(total_deform) < n_vertex:
        total_deform = resampleSphereSurf(fixed_xyz[0:len(total_deform),:], fixed_xyz[0:n_vertex,:], total_deform)
    moving_warp_phi = fixed_xyz[0:n_vertex,:] + total_deform
    moving_warp_phi = moving_warp_phi/np.linalg.norm(moving_warp_phi, axis=1)[:,np.newaxis]
    moving_warp_phi_resample = resampleSphereSurf(moving_warp_phi, fixed_xyz[0:n_vertex,:], moving_sulc[0:n_vertex])
    phi = inferDeformation(model_0, model_1, model_2, moving_warp_phi_resample, fixed_sulc[0:n_vertex], device, truncated=truncated)
    phi_reinterpo = resampleSphereSurf(fixed_xyz[0:n_vertex,:], moving_warp_phi, phi)
    total_deform = total_deform + phi_reinterpo
    
    return total_deform



file_name = '/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP000178_494/MNBCP000178_494.lh.SphereSurf.Orig.sphere.resampled.163842.vtk'
atlas_name = '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk'


def registerToAtlas(file_name, atlas_name):
    """
    read surface:  572.8440284729004 ms
    read atlas:  486.020565032959 ms
    infer on 642 :  4811.462163925171 ms
    resample to 2562:  1799.1602420806885 ms
    infer on 2562 :  5178.599834442139 ms
    get total deform on 2562:  883.124589920044 ms
    resample to 10242:  3291.548728942871 ms
    infer on 10242 :  6145.487070083618 ms
    get total deform on 10242:  1297.8949546813965 ms
    resample infer get total deform on 10242 again:  8875.661134719849 ms
    resample infer get total deform on 10242 again:  9059.664249420166 ms
    final resample to 163842:  14184.407472610474 ms
    
    total: 53s
    """
    
    device = torch.device('cuda:0')
    model_0_642, model_1_642, model_2_642 = get_trained_model(642, device)
    model_0_2562, model_1_2562, model_2_2562 = get_trained_model(2562, device)
    model_0_10242, model_1_10242, model_2_10242 = get_trained_model(10242, device)
#    model_0_40962, model_1_40962, model_2_40962 = get_trained_model(40962, device)
        
    
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
    total_deform = inferDeformation( model_0_642, model_1_642, model_2_642, moving_sulc[0:642], fixed_sulc[0:642], device)
   
    # reigs on 2562 vertices
    total_deform = inferTotalDeform(model_0_2562, model_1_2562, model_2_2562, 2562, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=False)
    
    # reigs on 10242 vertices
    total_deform = inferTotalDeform(model_0_10242, model_1_10242, model_2_10242, 10242, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=True)
    
    # reigs on 10242 vertices again
    total_deform = inferTotalDeform(model_0_10242, model_1_10242, model_2_10242, 10242, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=True)
    
    # reigs on 10242 vertices  
    total_deform = inferTotalDeform(model_0_10242, model_1_10242, model_2_10242, 10242, total_deform, fixed_xyz, moving_sulc, fixed_sulc, device, truncated=True)
    

    total_deform = resampleSphereSurf(fixed_xyz[0:10242,:], fixed_xyz, total_deform)
    moving_warp_phi_163842 = fixed_xyz + total_deform
    moving_warp_phi_163842 = moving_warp_phi_163842/np.linalg.norm(moving_warp_phi_163842, axis=1)[:,np.newaxis]
    
    t1 = time.time()
    
    moved = {'vertices': moving_warp_phi_163842 * 100.0,
             'faces': surf_163842['faces'],
             'curv': moving_curv * (2.08+2.32) - 2.32,
             'sulc': moving_sulc * (13.65+11.5) - 11.5}
    write_vtk(moved, file_name[:-3]+'moved.vtk')
    origin = {'vertices': surf_163842['vertices'],
              'faces': surf_163842['faces'],
              'deformation': total_deform * 100.0,
              'curv': moving_curv * (2.08+2.32) - 2.32,
              'sulc': moving_sulc * (13.65+11.5) - 11.5}
    write_vtk(origin, file_name[:-3]+'deform.vtk')
     
    moved_resample_sulc_curv = resampleSphereSurf(moving_warp_phi_163842, fixed_xyz, np.hstack((moved['sulc'][:,np.newaxis], moved['curv'][:,np.newaxis])))
    moved_resample = {'vertices': surf_163842['vertices'],
                      'faces': surf_163842['faces'],
                      'sulc': moved_resample_sulc_curv[:,0],
                      'curv': moved_resample_sulc_curv[:,1]}
    write_vtk(moved_resample, file_name[:-3]+'moved.resampled.163842.vtk')
    
    return (t1-t0)
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Rigister the surface to the atlas',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

  
    files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/*_lh.SphereSurf.Resampled160K.vtk'))
#    files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]
    cost = np.zeros(len(files))
    for i in range(len(files)):
        print(i)
        file = files[i]
        cost[i] = registerToAtlas(file, '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')
        print(cost[i], "s")
