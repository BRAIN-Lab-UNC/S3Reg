#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:44:25 2020

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

model_name = 'regis_sulc_2562_3d_smooth0p33_phiconsis1_3model'

###########################################################
""" split files, only need 18 month now"""
n_vertex = int(model_name.split('_')[2])

files_orig = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.'+ str(n_vertex) +'.npy'))
files_orig = [x for x in files_orig if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

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
    origin_cur = read_vtk(file)
    
    # TODO
    # Project deformation onto the tangent plane of the vertex and then interpolate the deformation
    moved_next_phi = resample_sphere_surf(origin_cur['vertices'], vertices_next, origin_cur['deformation'])
    
    tmp = read_vtk(os.path.dirname(file).replace('presentation/'+model_name, 'data/preprocessed_npy/') + \
                   os.path.basename(file).split('.')[0] + '/' + \
                   os.path.basename(file).split('.')[0] + '.lh.SphereSurf.Orig.sphere.resampled.'+ str(n_vertex*4-6) +'.vtk')
    moved_next_orig = {'vertices': vertices_next,
                  'faces': faces_next,
                  'deformation': moved_next_phi,
                  'sulc': tmp['sulc'],
                  'curv': tmp['curv']}
    write_vtk(moved_next_orig, file.replace(model_name, model_name+'/training_'+str(n_vertex*4-6)).replace('.vtk','.phi_resampled.'+ str(n_vertex*4-6) +'.origin.vtk'))
    
    moved_vertices = vertices_next + moved_next_phi
    moved_vertices = moved_vertices / np.linalg.norm(moved_vertices, axis=1)[:,np.newaxis] * 100
    moved_next = {'vertices': moved_vertices,
                  'faces': faces_next,
                  'sulc': tmp['sulc'],
                  'curv': tmp['curv']}
    write_vtk(moved_next, file.replace(model_name, model_name+'/training_'+str(n_vertex*4-6)).replace('.vtk','.phi_resampled.'+ str(n_vertex*4-6) +'.moved.vtk'))
    
    moved_next_resample_sulc_curv = resample_sphere_surf(moved_vertices, vertices_next, np.hstack((moved_next['sulc'][:,np.newaxis], moved_next['curv'][:,np.newaxis])))
    moved_next_resample = {'vertices': vertices_next,
                           'faces': faces_next,
                           'sulc': moved_next_resample_sulc_curv[:,0],
                           'curv': moved_next_resample_sulc_curv[:,1]}
    write_vtk(moved_next_resample, file.replace(model_name, model_name+'/training_'+str(n_vertex*4-6)).replace('.vtk','.phi_resampled.'+ str(n_vertex*4-6) +'.moved.sucu_resampled.'+ str(n_vertex*4-6) +'.vtk'))
    