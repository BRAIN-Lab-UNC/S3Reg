#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:28:13 2020

@author: fenqiang
"""

import numpy as np
import glob
from utils_vtk import read_vtk, write_vtk
from utils_interpolation import resampleSphereSurf
from utils import check_intersect_vertices
import os

###########################################################
""" hyper-parameters """

model_name = 'regis_sulc_10242_3d_smooth0p8_phiconsis1_3model_one_step_truncated'

###########################################################
""" split files, only need 18 month now"""
n_vertex = int(model_name.split('_')[2])

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/presentation/' + model_name + '/*.origin_3.vtk'))
#assert len(files) == len(files_orig), "len(orig_files) == len(files), error"


###########################################################
""" check intersection """
#passed = []
#
#for file in files:
#    print(file)
#    moved = read_vtk(file.replace('origin_3.vtk', 'moved_3.vtk'))
#    passed.append(check_intersect_vertices(moved['vertices'], moved['faces'][:,1:]))
#
#if False in passed:
#    indices = [i for i, x in enumerate(passed) if x == False]
#    print("Found intersecion in", len(indices) , "files, they are: ")
#    print([files[index] for index in indices])
#else:
#    print("Check intersection passed.")

    
###########################################################
""" generate high resolution surface for next level training """
template_next = read_vtk('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/sphere_'+ str(n_vertex*4-6) +'_rotated_0.vtk')
vertices_next = template_next['vertices']
faces_next = template_next['faces']

for file in files:
    print(file)
    origin_curr = read_vtk(file)
    
    # TODO
    # Project new deformation to old one, simple to do, convert 3d to 2d, then project 2d to 3d;
    
        
    sucu = read_vtk(os.path.dirname(file).replace('presentation/'+model_name, 'data/preprocessed_npy/') + \
                   os.path.basename(file).split('.')[0] + '/' + \
                   os.path.basename(file).split('.')[0] + '.lh.SphereSurf.Orig.sphere.resampled.'+ str(n_vertex*4-6) +'.vtk')
    
    
    deform_prev = read_vtk('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_2562_3d_smooth0p33_phiconsis1_3model/training_10242/'+file.split('/')[-1].split('.')[0]+'.lh.SphereSurf.Orig.sphere.resampled.642.DL.origin_3.phi_resampled.2562.moved.sucu_resampled.2562.DL.origin_3.phi_resampled.10242.origin.vtk')
    deform_prev = deform_prev['deformation']
    moved_10242 =  read_vtk('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_2562_3d_smooth0p33_phiconsis1_3model/training_10242/'+file.split('/')[-1].split('.')[0]+'.lh.SphereSurf.Orig.sphere.resampled.642.DL.origin_3.phi_resampled.2562.moved.sucu_resampled.2562.DL.origin_3.phi_resampled.10242.moved.vtk')
    moved_10242 = moved_10242['vertices']
    deform_curre = resampleSphereSurf(origin_curr['vertices'], moved_10242, origin_curr['deformation'])
    deform = deform_prev + deform_curre
    deform_40962 = resampleSphereSurf(origin_curr['vertices'], vertices_next, deform)
    
    orig_40962 = {'vertices': vertices_next,
                  'faces': faces_next,
                  'deformation': deform_40962,
                  'sulc': sucu['sulc'],
                  'curv': sucu['curv']}
    write_vtk(orig_40962, file.replace(model_name, model_name+'/training_'+str(n_vertex*4-6)).replace('.vtk', '.' + str(n_vertex*4-6) +'.origin.vtk'))
    
    moved_40962 = orig_40962['vertices'] + orig_40962['deformation']
    moved_40962 = moved_40962 / np.linalg.norm(moved_40962, axis=1)[:,np.newaxis] * 100
    
    moved_next = {'vertices': moved_40962,
                  'faces': faces_next,
                  'sulc': sucu['sulc'],
                  'curv': sucu['curv']}
    write_vtk(moved_next, file.replace(model_name, model_name+'/training_'+str(n_vertex*4-6)).replace('.vtk','.' + str(n_vertex*4-6) +'.moved.vtk'))
    
    moved_next_resample_sulc_curv = resampleSphereSurf(moved_40962, vertices_next, np.hstack((moved_next['sulc'][:,np.newaxis], moved_next['curv'][:,np.newaxis])))
    moved_next_resample = {'vertices': vertices_next,
                           'faces': faces_next,
                           'sulc': moved_next_resample_sulc_curv[:,0],
                           'curv': moved_next_resample_sulc_curv[:,1]}
    write_vtk(moved_next_resample, file.replace(model_name, model_name+'/training_'+str(n_vertex*4-6)).replace('.vtk','.'+ str(n_vertex*4-6) +'.moved.'+ str(n_vertex*4-6) +'.vtk'))
    