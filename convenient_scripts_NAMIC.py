#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:10:20 2020

@author: fenqiang
"""

import numpy as np
import glob
import os
from utils_vtk import read_vtk, remove_field, write_vtk, resample_label
from utils import get_orthonormal_vectors, get_neighs_order
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import scipy.io as sio 
from utils_interpolation import resampleSphereSurf



#####################################################################
""" Preprocessing after resample using ResampleFeature on longleaf """
""" Combine lh.curv.resampled.txt, lh.sulc.resampled.txt into the vtk file """

files = sorted(glob.glob(os.path.join('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/sub*', 'lh.sulc.resampled.txt'))) 
for file in files:
    data = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
    data = remove_field(data, 'par_fs')
    data = remove_field(data, 'par_fs_vec')
    data = remove_field(data, 'thickness')
    data['sulc'] = np.squeeze(np.loadtxt(file))
    data['curv'] = np.squeeze(np.loadtxt(file.replace('sulc', 'curv')))
    write_vtk(data, file.replace('lh.sulc.resampled.txt', file.split('/')[9]+'.lh.SphereSurf.Orig.resampled.163842.vtk'))
    
    
#####################################################################
""" Preprocess npy and downsample """

template_40k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_40962.vtk')
template_10k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_10242.vtk')
template_2562 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_2562.vtk')
template_642 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_642.vtk')
template_162 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_162.vtk')
template_42 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_42.vtk')
    
files = sorted(glob.glob(os.path.join('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/sub*', '*.lh.SphereSurf.Orig.resampled.163842.vtk')))
for file in files:
    data = read_vtk(file)
    
    """ convert vtk to .npy data """
    curv = data['curv']
    sulc = data['sulc']
    sucu = np.concatenate((curv[:,np.newaxis], sulc[:,np.newaxis]), axis=1)
    np.save(file.replace('.163842.vtk', '.163842.npy'), sucu)
    np.save(file.replace('.163842.vtk', '.40962.npy'), sucu[0:40962])
    np.save(file.replace('.163842.vtk', '.10242.npy'), sucu[0:10242])
    np.save(file.replace('.163842.vtk', '.2562.npy'), sucu[0:2562])
    np.save(file.replace('.163842.vtk', '.642.npy'), sucu[0:642])
    np.save(file.replace('.163842.vtk', '.162.npy'), sucu[0:162])
    np.save(file.replace('.163842.vtk', '.42.npy'), sucu[0:42])


    """ downsample 160k to 40k 10k 2562 642 162... """ 
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:40962]
        else:
            data['faces'] = template_40k['faces']
    write_vtk(data, file.replace('.163842.vtk','.40962.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:10242]
        else:
            data['faces'] = template_10k['faces']
    write_vtk(data, file.replace('.163842.vtk','.10242.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:2562]
        else:
            data['faces'] = template_2562['faces']
    write_vtk(data, file.replace('.163842.vtk','.2562.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:642]
        else:
            data['faces'] = template_642['faces']
    write_vtk(data, file.replace('.163842.vtk','.642.vtk')) 
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:162]
        else:
            data['faces'] = template_162['faces']
    write_vtk(data, file.replace('.163842.vtk','.162.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:42]
        else:
            data['faces'] = template_42['faces']
    write_vtk(data, file.replace('.163842.vtk','.42.vtk'))


    


#####################################################################
""" downsample atlas """ 
data = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.163842.rotated_0.vtk')

file = '/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.163842.rotated_0.vtk'
for key, value in data.items():
    if key != 'faces':
        data[key] = data[key][0:40962]
    else:
        data['faces'] = template_40k['faces']
write_vtk(data, file.replace('.163842','.40962'))
for key, value in data.items():
    if key != 'faces':
        data[key] = data[key][0:10242]
    else:
        data['faces'] = template_10k['faces']
write_vtk(data, file.replace('.163842','.10242'))      
for key, value in data.items():
    if key != 'faces':
        data[key] = data[key][0:2562]
    else:
        data['faces'] = template_2562['faces']
write_vtk(data, file.replace('.163842','.2562'))
for key, value in data.items():
    if key != 'faces':
        data[key] = data[key][0:642]
    else:
        data['faces'] = template_642['faces']
write_vtk(data, file.replace('.163842','.642'))        
for key, value in data.items():
    if key != 'faces':
        data[key] = data[key][0:162]
    else:
        data['faces'] = template_162['faces']
write_vtk(data, file.replace('.163842','.162'))
for key, value in data.items():
    if key != 'faces':
        data[key] = data[key][0:42]
    else:
        data['faces'] = template_42['faces']
write_vtk(data, file.replace('.163842','.42'))


#####################################################################
""" rotate atlas """ 

ns_vertex = [42, 162, 642, 2562, 10242, 40962, 163842]
for n_vertex in ns_vertex:
    atlas = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+ str(n_vertex) +'.rotated_0.vtk')    
    vertices = atlas['vertices']
    rotate_mat = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                           [0, 1, 0],
                           [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])
    rotated_ver_1 = np.transpose(np.dot(rotate_mat, np.transpose(vertices)))
    atlas['vertices'] = rotated_ver_1
    write_vtk(atlas, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+str(n_vertex)+'.rotated_1.vtk')

    rotate_mat = np.array([[1, 0, 0],
                           [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                           [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    rotated_ver_2 = np.transpose(np.dot(rotate_mat, np.transpose(rotated_ver_1)))
    atlas['vertices'] = rotated_ver_2
    write_vtk(atlas, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+str(n_vertex)+'.rotated_2.vtk')


#####################################################################
"""  Normalize atlas to better visualize atlas """ 

ns_vertex = [42, 162, 642, 2562, 10242, 40962, 163842]
for n_vertex in ns_vertex:
    name = '/media/fenqiang/DATA/unc/Data/registration/NAMIC/atlas_FS/lh.sphere.'+ str(n_vertex) +'.rotated_0.vtk'
    
    atlas = read_vtk(name)
    sulc = atlas['sulc']
    curv = atlas['curv']
    atlas['sulc'] = (sulc + 1.8818) / (1.779+1.8818) * (16.1992+12.5661) - 12.5661
    atlas['curv'] = (curv + 0.67445) / (0.54047+0.67445) * (0.6 + 0.7) - 0.7
    write_vtk(atlas, name.replace('.vtk', '.norm.vtk'))
    
    atlas = read_vtk(name.replace('_0', '_1'))
    sulc = atlas['sulc']
    curv = atlas['curv']
    atlas['sulc'] = (sulc + 1.8818) / (1.779+1.8818) * (16.1992+12.5661) - 12.5661
    atlas['curv'] = (curv + 0.67445) / (0.54047+0.67445) * (0.6 + 0.7) - 0.7
    write_vtk(atlas, name.replace('_0', '_1').replace('.vtk', '.norm.vtk'))
    
    atlas = read_vtk(name.replace('_0', '_2'))
    sulc = atlas['sulc']
    curv = atlas['curv']
    atlas['sulc'] = (sulc + 1.8818) / (1.779+1.8818) * (16.1992+12.5661) - 12.5661
    atlas['curv'] = (curv + 0.67445) / (0.54047+0.67445) * (0.6 + 0.7) - 0.7
    write_vtk(atlas, name.replace('_0', '_2').replace('.vtk', '.norm.vtk'))
    
    
#####################################################################
""" convert vtk to npy """ 

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/M3_regis_sulc_2562_nondiffe_smooth10_phiconsis2_corr1/training_10242/*.lh.SphereSurf.Orig.resampled.642.DL.moved_3.upto2562.resampled.2562.DL.moved_3.upto10242.resampled.10242.vtk'))
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/results/M3_regis_sulc_10242_nondiffe_smooth8_phiconsis5_corr1/training_40962/*.lh.SphereSurf.Orig.resampled.642.DL.moved_3.upto2562.resampled.2562.DL.moved_3.upto10242.resampled.10242.DL.moved_3.upto40962.resampled.40962.vtk'))

for file in files:
    data = read_vtk(file)
    
    curv = data['curv']
    sulc = data['sulc']
    sucu = np.concatenate((curv[:,np.newaxis], sulc[:,np.newaxis]), axis=1)
    np.save(file.replace('.vtk', '.npy'), sucu)
    
    
#####################################################################
""" add label from NAMIC dataset """ 
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/sub*/sub*.lh.SphereSurf.Orig.resampled.163842.vtk'))

for file in files:
    data = read_vtk(file)
    par_vec = read_vtk('/media/fenqiang/Seagate/NAMIC/RigidAlignAndResampledData/SphericalMappingWithNewCurvSulc/resampledVTK/'+ file.split('/')[9] +'_lh.SphereSurf.Resampled160K.vtk')
    par_vec = par_vec['par_vec']
    data['par_vec'] = par_vec
    write_vtk(data, file)
    
   
#####################################################################
""" resample infaltedH for SD resgistration """ 
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SD_reg/sub*/surf/lh.inflated.H.mat'))
tem = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk');

for file in files:
    data = sio.loadmat(file)
    inflatedH = data['inflatedH']
    surf = read_vtk(file.replace('lh.inflated.H.mat', 'lh.sphere.vtk'))
    assert len(surf['vertices']) == len(inflatedH)
    resampled = resampleSphereSurf(surf['vertices'], tem['vertices'], inflatedH)
    
   
#####################################################################
""" Convert MSM registered results  """
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/MSM_reg/sub*/surf/Curv.L.sphere.reg.vtk'))

for file in files:
    surf = read_vtk(file)
    curv = np.loadtxt(file.replace('Curv.L.sphere.reg.vtk', 'lh.curv.resampled.txt'))
    sulc = np.loadtxt(file.replace('Curv.L.sphere.reg.vtk', 'lh.sulc.resampled.txt'))
    lbl = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/'+ file.split('/')[9] +'/' + file.split('/')[9] +'.lh.SphereSurf.Orig.resampled.163842.vtk')
    lbl = lbl['par_vec']
    surf['curv'] = curv
    surf['sulc'] = sulc
    surf['par_vec'] = lbl
    write_vtk(surf, file.replace('Curv.L.sphere.reg.vtk', 'Curv.L.sphere.reg.sucu.vtk'))
    
""" resample MSM registered results  """
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/MSM_reg/sub*/surf/Curv.L.sphere.reg.sucu.vtk'))
template_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')

for file in files:
    surf = read_vtk(file)
    sucu = np.hstack((surf['sulc'][:,np.newaxis], surf['curv'][:,np.newaxis]))
    resample_feat = resampleSphereSurf(surf['vertices'], template_163842['vertices'], sucu)
    resample_lbl = resample_label(surf['vertices'], template_163842['vertices'], surf['par_vec'])
    
    sphere_surf_163842 = {'vertices': template_163842['vertices'],
                          'faces': template_163842['faces'],
                          'curv': resample_feat[:,1],
                          'sulc': resample_feat[:,0],
                          'par_vec': resample_lbl}
    write_vtk(sphere_surf_163842, file.replace('Curv.L.sphere.reg.sucu.vtk', 'Curv.L.sphere.reg.sucu.resampled.163842.vtk'))



   
#####################################################################
""" Copy lbl to SD registered surfacea and resample  """

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SD_reg/sub*/surf/lh.sphere.resampled.AlignToFSAtlas.sphere.sucu.vtk'))

for file in files:
    surf = read_vtk(file)
    sub = file.split('/')[9]
    
    lbl = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/DL_reg/'+ sub +'/'+ sub +'.lh.SphereSurf.Orig.resampled.163842.vtk')
    surf['par_vec'] = lbl['par_vec']
    write_vtk(surf, file)

    
""" resample SD registered results  """
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SD_reg/sub*/surf/lh.sphere.resampled.AlignToFSAtlas.sphere.sucu.vtk'))

template_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
for file in files:
    print(file)
    surf = read_vtk(file)
        
    resample_sulc_curv = resampleSphereSurf(surf['vertices'], template_163842['vertices'], np.hstack((surf['sulc'][:,np.newaxis], surf['curv'][:,np.newaxis])))
    resample_surf = {'vertices': template_163842['vertices'],
                      'faces': template_163842['faces'],
                      'sulc': resample_sulc_curv[:,0],
                      'curv': resample_sulc_curv[:,1]}
    if 'par_vec' in surf.keys():
        resample_lbl = resample_label(surf['vertices'], template_163842['vertices'], surf['par_vec'])
        resample_surf['par_vec'] = resample_lbl
    write_vtk(resample_surf, file.replace('.vtk', '.resampled.163842.vtk'))
    