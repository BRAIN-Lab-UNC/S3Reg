#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 02:50:26 2020

@author: fenqiang
"""

import numpy as np 
import glob
import math, multiprocessing

from utils_vtk import read_vtk, write_vtk
from utils import get_par_36_to_fs_vec


DL_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.163842.moved.resampled.163842.vtk'))
SD_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/*/surf/lh.NewResampledAlignedToBCPAtlas.sphereresampled.163842.vtk'))
MSM_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/MSM/*/surf/Curv.L.sphere.reg.sucu.resampled.vtk'))

#DL_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/sub*_lh.SphereSurf.Resampled160K.moved.resampled.163842.vtk'))
#SD_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/SD_reg/sub*/surf/lh.sphere.AlignedToBCPAtlas.sphere.resampled.sucu.vtk'))

#atlas = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')
#fixed_sulc = atlas['sulc']
#fixed_curv = atlas['curv']

par_36_to_fs_vec = get_par_36_to_fs_vec()


def compute_dice(lbl1, lbl2):
    dice = np.zeros(36)
    for i in range(36):
        lbl1_indices = np.where(lbl1 == i)[0]
        lbl2_indices = np.where(lbl2 == i)[0]
        dice[i] = 2 * len(np.intersect1d(lbl1_indices, lbl2_indices))/(len(lbl1_indices) + len(lbl2_indices))
    return dice.mean()


def evaluate(files):
    sulc = np.zeros((len(files), 163842))
    curv = np.zeros((len(files), 163842))
    label = []
    
    for i in range(len(files)):
        print(i)
        
        data = read_vtk(files[i])
        sulc[i,:] = data['sulc']
        curv[i,:] = data['curv']
        
        if 'par_vec' in data.keys():
            lbl_vec = data['par_vec']
            lbl_36 = np.zeros(len(lbl_vec))
            for j in range(len(lbl_vec)):
                lbl_36[j] = np.where(np.all(lbl_vec[j] == par_36_to_fs_vec, axis=1))[0][0]
            label.append(lbl_36)
    
    label = np.asarray(label).astype(np.int32)
    assert len(label) == 72, "error at len(label)"
    
    corr_sulc = np.zeros((len(files), len(files)))
    corr_curv = np.zeros((len(files), len(files)))
    for i in range(len(SD_files)):
        for j in range(len(SD_files)):
            corr_sulc[i,j] = ((sulc[i,:] - sulc[i,:].mean()) * (sulc[j,:] - sulc[j,:].mean())).mean() / sulc[i,:].std() / sulc[j,:].std()  
            corr_curv[i,j] = ((curv[i,:] - curv[i,:].mean()) * (curv[j,:] - curv[j,:].mean())).mean() / curv[i,:].std() / curv[j,:].std()  
    
    print('corr_sulc mean, std:', corr_sulc.mean(), corr_sulc.std())
    print('corr_curv mean, std:', corr_curv.mean(), corr_curv.std())
    
    mae_sulc = np.zeros((len(files), len(files)))
    mae_curv = np.zeros((len(files), len(files)))
    for i in range(len(files)):
        for j in range(len(files)):
            mae_sulc[i,j] = abs(sulc[i,:] - sulc[j,:]).mean()
            mae_curv[i,j] = abs(curv[i,:] - curv[j,:]).mean()
        
    print('mae_sulc mean, std:', mae_sulc.mean(), mae_sulc.std())
    print('mae_curv mean, std:', mae_curv.mean(), mae_curv.std())
    
    dice = np.zeros((len(label), len(label)))
    for i in range(len(label)):
        print(i)
        for j in range(len(label)): 
            dice[i,j] = compute_dice(label[i,:], label[j,:])
            
    print('dice mean, std:', dice.mean(), dice.std())

print("evaluate DL:")
evaluate(DL_files)

print("evaluate SD:")
evaluate(SD_files)

print("evaluate MSM:")
evaluate(MSM_files)


###############################################################################

sphere_template = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')
inner_template = read_vtk('/media/fenqiang/DATA/unc/Data/Template/UNC-Infant-Cortical-Surface-Atlas/18/18_lh.InnerSurf.vtk')
inflate_templat = read_vtk('/media/fenqiang/DATA/unc/Data/Template/UNC-Infant-Cortical-Surface-Atlas/18/18_lh.InnerSurf.inflated.vtk')

a = sulc.mean(0)
b = curv.mean(0)

sphere_template['sulc'] = a
sphere_template['curv'] = b
write_vtk(sphere_template,  '/media/fenqiang/DATA/unc/Data/registration/ForFigure/MSM_atlas.sphere.vtk')

inner_template['sulc'] = a
inner_template['curv'] = b
write_vtk(inner_template,  '/media/fenqiang/DATA/unc/Data/registration/ForFigure/MSM_atlas.inner.vtk')


inflate_templat['sulc'] = a
inflate_templat['curv'] = b
write_vtk(inflate_templat,  '/media/fenqiang/DATA/unc/Data/registration/ForFigure/DL_atlas.inflate.vtk')

###############################################################################

template = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')
faces = template['faces']
faces = faces[:,1:]
    
vertex_faces = list(range(163842))
for i in range(len(vertex_faces)):
    vertex_faces[i] = []
    
for i in range(len(faces)):
#    print(i)
    vertex_faces[faces[i,0]].append(i)
    vertex_faces[faces[i,1]].append(i)
    vertex_faces[faces[i,2]].append(i)
    
for i in range(163842):
    if(len(vertex_faces[i]) == 5):
        vertex_faces[i].append(vertex_faces[i][0])
        
vertex_faces = np.asarray(vertex_faces)



def compute_triangles_area(a, b, c):
    """
    a, b, c: N*3 numpy array, represents three vertices 
    """
    return np.linalg.norm(np.cross(a-b, c-b), axis=1)/2.0

def compute_area_dis(old_file, new_file):
    old_ver_surf = read_vtk(old_file)
    old_ver = old_ver_surf['vertices'].astype(np.float64)
    faces_1 = old_ver_surf['faces']
    faces_1 = faces_1[:,1:]

    new_ver_surf = read_vtk(new_file)
    new_ver = new_ver_surf['vertices'].astype(np.float64)
    faces_2 = new_ver_surf['faces']
    faces_2 = faces_2[:,1:]
     
    assert (faces_1 == faces_2).sum() == faces_2.shape[0] * faces_2.shape[1] 
    
    old_area = compute_triangles_area(old_ver[faces_1[:,0],:], old_ver[faces_1[:,1],:], old_ver[faces_1[:,2],:])
    new_area = compute_triangles_area(new_ver[faces_1[:,0],:], new_ver[faces_1[:,1],:], new_ver[faces_1[:,2],:])

    if np.any(old_area == 0) or np.any(new_area == 0):
        return 0 
    
    area_dis = np.mean(np.abs(np.mean(np.log2(old_area[vertex_faces] / new_area[vertex_faces]), axis=1)))

    return area_dis

DL_area_dis = np.zeros(len(DL_files))
for i in range(len(DL_files)):
    print(i)
    DL_area_dis[i] = compute_area_dis(DL_files[i].replace('.moved.resampled.163842.vtk','.vtk'), DL_files[i].replace('.moved.resampled.163842.vtk','.moved.vtk'))
    
DL_area_dis[(DL_area_dis==0).nonzero()] = DL_area_dis[(DL_area_dis != 0).nonzero()].mean()
print("DL_area_dis, mean, std:", DL_area_dis.mean(), DL_area_dis.std())
    

SD_area_dis = np.zeros(len(SD_files))
for i in range(len(SD_files)):
    print(i)
    SD_area_dis[i] = compute_area_dis(SD_files[i].replace('lh.NewResampledAlignedToBCPAtlas.sphereresampled.163842.vtk','lh.sphere.resampled.vtk'), SD_files[i].replace('lh.NewResampledAlignedToBCPAtlas.sphereresampled.163842.vtk','lh.NewResampledAlignedToBCPAtlas.sphere.vtk'))
    
SD_area_dis[(SD_area_dis==0).nonzero()] = SD_area_dis[(SD_area_dis != 0).nonzero()].mean()
print("SD_area_dis, mean, std:", SD_area_dis.mean(), SD_area_dis.std())    


MSM_area_dis = np.zeros(len(MSM_files))
for i in range(len(MSM_files)):
    print(i)
    MSM_area_dis[i] = compute_area_dis(MSM_files[i].replace('Curv.L.sphere.reg.sucu.resampled.vtk','lh.sphere.resampled.vtk'), MSM_files[i].replace('Curv.L.sphere.reg.sucu.resampled.vtk','Curv.L.sphere.reg.vtk'))

MSM_area_dis[(MSM_area_dis==0).nonzero()] = MSM_area_dis[(MSM_area_dis != 0).nonzero()].mean()
print("MSM_area_dis, mean, std:", MSM_area_dis.mean(), MSM_area_dis.std())
    

