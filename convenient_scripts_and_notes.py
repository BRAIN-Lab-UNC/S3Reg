#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:13:58 2020

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
""" compute the number of subjects of each month """

files = sorted(glob.glob(os.path.join('/media/fenqiang/DATA/unc/Data/registration/data/orig_data', '*lh*.Orig.vtk'))) 
age = [float(x.split('/')[-1].split('_')[1].split('.')[0]) for x in files ]

n_subjects_per_month = {'0':0, 
                        '3':0,
                        '6':0,
                        '9':0,
                        '12':0,
                        '18':0,
                        '24':0,
                        '36':0,
                        '48':0,
                        '60':0,
                        '72':0}

for a in age:
    if a >= 0 and a < 45:
        n_subjects_per_month['0'] = n_subjects_per_month['0']+1
    elif a >= 45 and a < 135:
        n_subjects_per_month['3'] = n_subjects_per_month['3']+1
    elif a >= 135 and a < 225:
        n_subjects_per_month['6'] = n_subjects_per_month['6']+1
    elif a >= 225 and a < 315:
        n_subjects_per_month['9'] = n_subjects_per_month['9']+1
    elif a >= 315 and a < 450:
        n_subjects_per_month['12'] = n_subjects_per_month['12']+1
    elif a >= 450 and a < 630:
        n_subjects_per_month['18'] = n_subjects_per_month['18']+1
    elif a >= 630 and a < 900:
        n_subjects_per_month['24'] = n_subjects_per_month['24']+1
    elif a >= 900 and a < 1260:
        n_subjects_per_month['36'] = n_subjects_per_month['36']+1
    elif a >= 1260 and a < 1620:
        n_subjects_per_month['48'] = n_subjects_per_month['48']+1
    elif a >= 1620 and a < 1980:
        n_subjects_per_month['60'] = n_subjects_per_month['60']+1
    elif a >= 1980:
        n_subjects_per_month['72'] = n_subjects_per_month['72']+1
    else:
        raise NotImplementedError('age is out of range')
    
num = 0    
for value in n_subjects_per_month.values():
    num += value
assert num == len(files), "error!"


#####################################################################
""" Preprocessing after resample using ResampleFeature on longleaf """
""" Combine lh.curv.resampled.txt, lh.sulc.resampled.txt into the vtk file """

files = sorted(glob.glob(os.path.join('/media/fenqiang/DATA/unc/Data/registration/data/orig_data', '*.resampled.vtk'))) 
for file in files:
    data = read_vtk(file)
    data = remove_field(data, 'vertexArea')
    data['curv'] = np.squeeze(np.loadtxt(file.replace('resampled.vtk', 'curv.resampled.txt')))
    data['sulc'] = np.squeeze(np.loadtxt(file.replace('resampled.vtk', 'sulc.resampled.txt')))
    write_vtk(data, file)
    
    
#####################################################################
""" Preprocess npy and downsample """

template_40k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_40962.vtk')
template_10k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_10242.vtk')
template_2562 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_2562.vtk')
template_642 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_642.vtk')
template_162 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_162.vtk')
template_42 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_42.vtk')
    
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/orig_data/*lh*.resampled.vtk'))
for file in files:
    
    data = read_vtk(file)
    
    """ convert vtk to .npy data """
    # mkdir for different subject
    subject_id = file.split('/')[-1].split('.')[0]
#    os.system("mkdir -p /media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/" + subject_id)
    
#    curv = data['curv']
#    sulc = data['sulc']
#    data = np.concatenate((curv[:,np.newaxis], sulc[:,np.newaxis]), axis=1)
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.163842.npy'), data)
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.40962.npy'), data[0:40962])
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.10242.npy'), data[0:10242])
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.2562.npy'), data[0:2562])
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.642.npy'), data[0:642])
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.162.npy'), data[0:162])
#    np.save(file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk', '.42.npy'), data[0:42])


    """ downsample 160k to 40k 10k 2562 642 162... """ 
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.163842.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:40962]
        else:
            data['faces'] = template_40k['faces']
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.40962.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:10242]
        else:
            data['faces'] = template_10k['faces']
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.10242.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:2562]
        else:
            data['faces'] = template_2562['faces']
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.2562.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:642]
        else:
            data['faces'] = template_642['faces']
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.642.vtk')) 
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:162]
        else:
            data['faces'] = template_162['faces']
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.162.vtk'))
    for key, value in data.items():
        if key != 'faces':
            data[key] = data[key][0:42]
        else:
            data['faces'] = template_42['faces']
    write_vtk(data, file.replace('orig_data','preprocessed_npy/'+subject_id).replace('.vtk','.42.vtk'))


    


#####################################################################
""" Compute atlas of bcp """ 

""" downsample template 18 month from 160k to 40k """ 
template_160k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
template_40k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_40962.vtk')
template_10k = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_10242.vtk')
template_2562 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_2562.vtk')
template_642 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_642.vtk')
template_162 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_162.vtk')
template_42 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_42.vtk')

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.AlignedToAtlas.sphere.resampled.163842.npy'))
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

data = np.zeros((len(files), 163842, 2))
for i in range(len(files)):
    data[i,:,:] = np.load(files[i])
mean_atlas = np.mean(data, axis=0)

data = {'vertices': template_160k['vertices'],
        'faces': template_160k['faces'],
        'curv': np.squeeze(mean_atlas[:,0]),
        'sulc': np.squeeze(mean_atlas[:,1])}
write_vtk(data, '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')

file = '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk'
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

""" save npy atlas """
data = read_vtk(file)
curv = data['curv']
sulc = data['sulc']
data = np.concatenate((curv[:, np.newaxis], sulc[:, np.newaxis]), axis=1)
np.save(file.replace('vtk','npy'), data)
np.save(file.replace('vtk','npy').replace('163842','40962'), data[0:40962])
np.save(file.replace('vtk','npy').replace('163842','10242'), data[0:10242])
np.save(file.replace('vtk','npy').replace('163842','2562'), data[0:2562])
np.save(file.replace('vtk','npy').replace('163842','642'), data[0:642])
np.save(file.replace('vtk','npy').replace('163842','162'), data[0:162])
np.save(file.replace('vtk','npy').replace('163842','42'), data[0:42])


#####################################################################
""" rotate atlas """ 

ns_vertex = [42, 162, 642, 2562, 10242, 40962, 163842]
for n_vertex in ns_vertex:
    atlas = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_0.vtk')    
    vertices = atlas['vertices']
    rotate_mat = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                           [0, 1, 0],
                           [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])
    rotated_ver = np.transpose(np.dot(rotate_mat, np.transpose(vertices)))
    atlas['vertices'] = rotated_ver
    write_vtk(atlas, '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_1.vtk')

    rotate_mat = np.array([[1, 0, 0],
                           [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                           [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    rotated_ver_2 = np.transpose(np.dot(rotate_mat, np.transpose(rotated_ver)))
    atlas['vertices'] = rotated_ver_2
    write_vtk(atlas, '/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+str(n_vertex)+'.rotated_2.vtk')


#####################################################################
""" max min of sulc curv """ 

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.AlignedToAtlas.sphere.resampled.163842.npy'))
data = np.zeros((len(files), 163842, 2))
for i in range(len(files)):
    data[i,:,:] = np.load(files[i])
a = np.max(data, axis = 1)
np.mean(a, axis = 0)    
a = np.min(data, axis = 1)
np.mean(a, axis = 0) 


#####################################################################
""" check npy and vtk data is consistent """ 

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.*.sphere.resampled.*.vtk')) 
for file in files:
    print(files.index(file))
    data = read_vtk(file)
    curv = data['curv']
    sulc = data['sulc']
    data = np.concatenate((curv[:,np.newaxis], sulc[:, np.newaxis]),1)
    npy = np.load(file.replace('vtk','npy'))
    if (data == npy).sum() != len(data) *2:
        print(file)
        print('this file not consistent!')
        raise NotImplementedError('error')
        

#####################################################################
""" check deformation from 2d to 3d """ 
import matplotlib.pyplot as plt

data = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_642.vtk')
vertices = data['vertices']
faces = data['faces']

En = get_orthonormal_vectors(642)
neigh_orders_642 = Get_neighs_order()[4]

z_n = np.array([[10],[0]])
gamma_n = np.zeros((3, len(En)))
for i in range(len(En)):
    gamma_n[:,i] = np.squeeze(np.dot(En[i,:,:], z_n))

gamma_n = np.transpose(gamma_n)
data = {'vertices': vertices,
        'faces': faces,
        'deformation': gamma_n}
write_vtk(data, '/media/fenqiang/DATA/unc/Data/registration/presentation/test_642.vtk')

origin = [0], [0]
a = gamma_n[neigh_orders_642[0:5]]
a = a[:,0:2]
plt.quiver(*origin, a[:,0], a[:,1])
plt.show()

#####################################################################
"""compute vertex distance, the edge length of spheres """
 
def compute_vertex_distance(vertices, faces):
    dis = []
    for i in range(len(faces)):
        for j in range(3):
            edge = [1,2,3]
            edge.remove(j+1)
            dis.append(np.sqrt(np.sum((vertices[faces[i, edge[0]]] - vertices[faces[i, edge[1]]])**2)))
    assert len(dis) == faces.shape[0] * 3
    dis = np.asarray(dis)
    print(len(vertices) , "vertices: mean:", dis.mean(), "std: ", dis.std())
    print(len(vertices) , "vertices: max:", dis.max(), "min: ", dis.min())
    
sphere_42 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_42.vtk')
compute_vertex_distance(sphere_42['vertices'], sphere_42['faces'])
sphere_162 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_162.vtk')
compute_vertex_distance(sphere_162['vertices'], sphere_162['faces'])
sphere_642 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_642.vtk')
compute_vertex_distance(sphere_642['vertices'], sphere_642['faces'])
sphere_2562 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_2562.vtk')
compute_vertex_distance(sphere_2562['vertices'], sphere_2562['faces'])
sphere_10242 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_10242.vtk')
compute_vertex_distance(sphere_10242['vertices'], sphere_10242['faces'])
sphere_40962 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_40962.vtk')
compute_vertex_distance(sphere_40962['vertices'], sphere_40962['faces'])
sphere_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
compute_vertex_distance(sphere_163842['vertices'], sphere_163842['faces'])


#####################################################################
""" check image patches from sphere """

i = 1
shape = [65,65]
pixel_length = 1.2  #np.sqrt(4*np.pi*10000/len(vertices))

sphere_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842_rotated_0.vtk') 
En_163842 = get_orthonormal_vectors(163842,0)
vertices = sphere_163842['vertices'].astype(np.float64)
tree = KDTree(vertices, leaf_size=10)  # build kdtree
faces = sphere_163842['faces']
faces = faces[:,[1,2,3]]
faces = np.sort(faces, axis=1)
sulc = sphere_163842['sulc']
neigh_orders_163842 = Get_neighs_order()[0]
En_i = En_163842[i]

assert shape[0] == shape[1], "Shape should be square."
assert shape[0]%2 == 1 and shape[1]%2 == 1, "Shape[0] and shape[1] should be odd number."

inter_indices = np.zeros((shape[0]*shape[1],3)).astype(np.int64)
inter_weights = np.zeros((shape[0]*shape[1],3))

for p in range(shape[0]):
#    print(p)
    for q in range(shape[1]):
#        print(q)
        if p == int(shape[0]/2) and q == int(shape[0]/2):
             inter_indices[p*shape[0]+q,:], inter_weights[p*shape[0]+q,:] = np.array([i,i,i]), np.array([1,0,0])
        else:
            loc_tangent = En_i.dot(np.array([(q-int(shape[0]/2))*pixel_length, (int(shape[0]/2)-p)*pixel_length]))
            t = np.linalg.norm(loc_tangent)/100.0   # theta to rotate
            n = np.cross(loc_tangent, vertices[i])  # normal of the great circle, the rotation axis
            n = n/np.linalg.norm(n)                # normalize normal
            rot_mat = np.array([[n[0]*n[0]*(1-np.cos(t))+np.cos(t),      n[0]*n[1]*(1-np.cos(t))+n[2]*np.sin(t), n[0]*n[2]*(1-np.cos(t))-n[1]*np.sin(t)],
                                [n[0]*n[1]*(1-np.cos(t))-n[2]*np.sin(t), n[1]*n[1]*(1-np.cos(t))+np.cos(t),      n[1]*n[2]*(1-np.cos(t))+n[0]*np.sin(t)],
                                [n[0]*n[2]*(1-np.cos(t))+n[1]*np.sin(t), n[1]*n[2]*(1-np.cos(t))-n[0]*np.sin(t), n[2]*n[2]*(1-np.cos(t))+np.cos(t)]])
            assert abs(np.linalg.det(rot_mat) - 1.0) < 1e-5, "rotation matrix's det should be 1!"
            sphere_loc = rot_mat.dot(vertices[i])
            assert abs(np.linalg.norm(sphere_loc) - 100) < 0.01
            
            inter_indices[p*shape[0]+q,:], inter_weights[p*shape[0]+q,:] = singleVertexInterpo(sphere_loc, vertices, tree, neigh_orders_163842)
    
patch = np.sum(np.multiply((sulc[inter_indices.flatten()]).reshape((inter_indices.shape[0], inter_indices.shape[1])), inter_weights), axis=1)
patch = patch.reshape((shape[0],shape[1]))
plt.imshow(patch)
            
            
#####################################################################
""" check lonlgeaf visited vertices index and image patches from sphere """       

indices_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/*_indices.npy'))
weights_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/*_weights.npy'))

assert len(indices_files) == len(weights_files), "indices files should have the same number as weights number"

indices = [x.split('/')[-1].split('_')[0] for x in indices_files]
weights = [x.split('/')[-1].split('_')[0] for x in weights_files]

for x in indices:
    if x not in weights:
        print(x, "only in indices, not in weights")
for x in weights:
    if x not in indices:
        print(x, "only in weights, not in indices")
assert indices == weights, "indices are not consistent with weights!"

indices = [int(x) for x in indices]
weights = [int(x) for x in weights]
assert indices == weights, "indices are not consistent with weights!"

indices = np.sort(np.asarray(indices))
np.savetxt('/media/fenqiang/DATA/unc/Data/registration/scripts/visited_indices_for_patch_on_longleaf.txt', indices, fmt='%d')


import random

shape = [65,65]
atlas_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.vtk') 
vertices = atlas_163842['vertices'].astype(np.float64)
atlas_sulc = atlas_163842['sulc']

moving_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP000178_494/MNBCP000178_494.lh.SphereSurf.Orig.sphere.resampled.163842.vtk')
moving_sulc = moving_163842['sulc']

i = indices[random.randint(0,len(indices))]
print(vertices[i])
inter_indices = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/'+str(i)+'_indices.npy')
inter_weights = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/'+str(i)+'_weights.npy')

patch = np.sum(np.multiply((atlas_sulc[inter_indices.flatten()]).reshape((inter_indices.shape[0], inter_indices.shape[1])), inter_weights), axis=1)
patch = patch.reshape((shape[0],shape[1]))
plt.figure(1)
plt.imshow(patch)

patch = np.sum(np.multiply((moving_sulc[inter_indices.flatten()]).reshape((inter_indices.shape[0], inter_indices.shape[1])), inter_weights), axis=1)
patch = patch.reshape((shape[0],shape[1]))
plt.figure(2)
plt.imshow(patch)
            


#####################################################################
""" check bilinear interpolation """  

import torch

a = torch.tensor([[0,0.5,1],[1,1.5,2], [2,2.5,3]]).float()
a = a.unsqueeze(0)
tgt = a.unsqueeze(0)
tgt = tgt.repeat(4, 1, 1, 1)
flow = torch.tensor(list(range(64)))
grid_x, grid_y = torch.meshgrid(flow, flow)
grid_x_2 = grid_y - 32
grid_y_2 = 32 - grid_x
flow = torch.cat((grid_x_2.unsqueeze(2), grid_y_2.unsqueeze(2)), 2)
flow = flow.float()/32.0
img = torch.zeros(4, 64,64)
for i in range(64):
    for j in range(64):
        flow_ij = flow[i,j,:]
        flow_ij = flow_ij.unsqueeze(0)
        flow_ij = flow_ij.repeat(4,1)
        img[:,i,j] = bilinear_inter(tgt, flow_ij)

plt.figure(3)
plt.imshow(img[3,:,:])        
plt.colorbar()

def bilinear_inter(tgt, flow):
    """
     Push the src and flow through the spatial transform block
        tgt: N*1*65*65
        flow: N*2
    """
    assert flow.shape[0] == tgt.shape[0], "Flow size shoulbe be consistent with images"
    assert flow.shape == (flow.shape[0], 2), "Error!"
    
    side = 1.0
    
    new_locs = flow*side
    right_top = torch.ceil(new_locs).long()
    right_top[right_top <= -side] = -side+1  # rights top should beyond the left bottom boundary
    left_bottom = right_top - 1
    
    left_top = torch.cat((left_bottom[:,0].unsqueeze(1), right_top[:,1].unsqueeze(1)), dim=1)
    right_bottom = torch.cat((right_top[:,0].unsqueeze(1), left_bottom[:,1].unsqueeze(1)), dim=1)
    
    weight_mat = torch.cat((((right_top[:,0].float()-new_locs[:,0])*(new_locs[:,1]-right_bottom[:,1].float())).unsqueeze(1),
                            ((right_top[:,0].float()-new_locs[:,0])*(right_top[:,1].float()-new_locs[:,1])).unsqueeze(1),
                            ((new_locs[:,0]-left_bottom[:,0].float()) *(right_top[:,1].float()-new_locs[:,1])).unsqueeze(1),
                            ((new_locs[:,0]-left_bottom[:,0].float()) *(new_locs[:,1]-right_bottom[:,1].float())).unsqueeze(1)), dim=1)
    
    right_top_glo = torch.zeros_like(right_top)
    right_top_glo[:,1] = right_top[:,0] + side
    right_top_glo[:,0] = side - right_top[:,1]
    right_top = right_top_glo
    left_bottom = torch.zeros_like(right_top)
    left_bottom[:,0] = right_top[:,0] + 1
    left_bottom[:,1] = right_top[:,1] - 1
     
    left_top = torch.cat((right_top[:,0].unsqueeze(1), left_bottom[:,1].unsqueeze(1)), dim=1)
    right_bottom = torch.cat((left_bottom[:,0].unsqueeze(1), right_top[:,1].unsqueeze(1)), dim=1)
   
    left_top_value = tgt[list(range(tgt.shape[0])), [0], left_top[:,0], left_top[:,1]]
    left_bottom_value = tgt[list(range(tgt.shape[0])), [0], left_bottom[:,0], left_bottom[:,1]]
    right_bottom_value = tgt[list(range(tgt.shape[0])), [0], right_bottom[:,0], right_bottom[:,1]]
    right_top_value = tgt[list(range(tgt.shape[0])), [0], right_top[:,0], right_top[:,1]]
    
    value = torch.cat((left_top_value.unsqueeze(1),left_bottom_value.unsqueeze(1),right_bottom_value.unsqueeze(1),right_top_value.unsqueeze(1)), dim=1)
   
    y = torch.sum(value*weight_mat, dim=1)
    
    return y


#####################################################################
""" check rotated neighbors orders """ 

#def get_neighs_order(order_path):
#    adj_mat_order = sio.loadmat(order_path)
#    adj_mat_order = adj_mat_order['adj_mat_order']
#    neigh_orders = np.zeros((len(adj_mat_order), 7))
#    neigh_orders[:,0:6] = adj_mat_order-1
#    neigh_orders[:,6] = np.arange(len(adj_mat_order))
#    
#    return neigh_orders

ns_vertex = np.array([163842,40962,10242,2562,642,162,42,12])
for n_vertex in ns_vertex:
    neigh_orders_0 = get_neighs_order('neigh_indices/adj_mat_order_' + str(n_vertex) + '_rotated_0.mat')
    neigh_orders_1 = get_neighs_order('neigh_indices/adj_mat_order_' + str(n_vertex) + '_rotated_1.mat')
    neigh_orders_2 = get_neighs_order('neigh_indices/adj_mat_order_' + str(n_vertex) + '_rotated_2.mat')
    
    neigh_orders_0 = np.sort(neigh_orders_0, axis=1)        
    neigh_orders_1 = np.sort(neigh_orders_1, axis=1)     
    neigh_orders_2 = np.sort(neigh_orders_2, axis=1)        

    assert (neigh_orders_0 == neigh_orders_1).sum() == neigh_orders_0.shape[0] * neigh_orders_0.shape[1], "error!"
    assert (neigh_orders_1 == neigh_orders_2).sum() == neigh_orders_1.shape[0] * neigh_orders_1.shape[1], "error!"



#####################################################################
""" check rotated vectors """

surf = read_vtk('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP107842_593/MNBCP107842_593.lh.SphereSurf.Orig.sphere.resampled.2562.vtk')
En = get_orthonormal_vectors(2562,0)
phi_2d = np.random.uniform(5,12,2562*2)
phi_2d = phi_2d.reshape((2562,2))
phi_3d = np.zeros((2562,3))
for i in range(2562):
    phi_3d[i,:] = np.dot(En[i,:,:], phi_2d[i,:][:,np.newaxis]).squeeze()
surf['deformation'] = phi_3d
write_vtk(surf, '/media/fenqiang/DATA/unc/Data/registration/delete_this/test_origin.vtk')

rotate_mat_01 = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                           [0, 1, 0],
                           [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])
rotate_mat_12 = np.array([[1, 0, 0],
                           [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                           [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
rotate_mat_02 = np.dot(rotate_mat_12, rotate_mat_01)
rotated_ver = np.transpose(np.dot(rotate_mat_02, np.transpose(surf['vertices'])))
rotated_phi_3d = np.transpose(np.dot(rotate_mat_02, np.transpose(phi_3d)))
rotated_surf = {'vertices': rotated_ver,
                'faces': surf['faces'],
                'sulc': surf['sulc'],
                'deformation': rotated_phi_3d}
write_vtk(rotated_surf, '/media/fenqiang/DATA/unc/Data/registration/delete_this/test_1.vtk')



#####################################################################
""" check intersection """

from utils import check_intersect_vertices


files = sorted(glob.glob(os.path.join('/media/fenqiang/DATA/unc/Data/registration/data/orig_data', '*lh*.Orig.vtk'))) 
age = [float(x.split('/')[-1].split('_')[1].split('.')[0]) for x in files ]



#####################################################################
""" Convert vtk to npy """
files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_10242_3d_smooth0p8_phiconsis1_3model_one_step_truncated/training_40962_113_truncated/*.lh.SphereSurf.Orig.sphere.resampled.163842.moved.resampled.163842.vtk'))
for file in files:
    data = read_vtk(file)
    data = np.concatenate((data['curv'][0:40962][:,np.newaxis], data['sulc'][0:40962][:,np.newaxis]), axis=1)
    np.save(file.replace('.163842.vtk','.40962.npy'), data)
    
    
    
####################################################################
""" save new atlas's lh.curv lh.sulc for SD registration """
surf = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')
sulc = surf['sulc']
curv = surf['curv']
np.savetxt('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/freesurfer_format/lh.sulc.txt', sulc)
np.savetxt('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/freesurfer_format/lh.curv.txt', curv)


####################################################################
""" Resample label and write it into the Deep Learning registered sphere on NAMIC dataset """

from utils import get_par_35_to_fs_vec
from utils_vtk import resample_label

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/*_lh.SphereSurf.Resampled160K.moved.vtk'))
for i in range(len(files)):
    deformed_ver = read_vtk(files[i])
    deformed_ver = deformed_ver['vertices']
    orig_lbl = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/' + files[i].split('/')[-1].split('.')[0] + '.SphereSurf.Resampled160K.vtk')
    orig_lbl = orig_lbl['par_1_to_35']
    resampled_surf = read_vtk('/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/' + files[i].split('/')[-1].split('.')[0] + '.SphereSurf.Resampled160K.moved.resampled.163842.vtk')
    resampled_lbl = resample_label(deformed_ver, resampled_surf['vertices'], orig_lbl)
    resampled_surf['par_1_to_35'] = resampled_lbl
    
    lbl_35_to_vec = get_par_35_to_fs_vec()
    resampled_surf['par_vec'] = lbl_35_to_vec[resampled_lbl-1]

    write_vtk(resampled_surf, '/media/fenqiang/DATA/unc/Data/registration/NAMIC/SphericalMappingWithNewCurvSulc/DL_reg/' + files[i].split('/')[-1].split('.')[0] + '.SphereSurf.Resampled160K.moved.resampled.163842.vtk')
    
    
    
####################################################################
""" compare old bcp data for training nitrc model and new bcp data for registration task, for copying the par_fs in old data to new data """

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.163842.vtk'))
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

for i in range(len(files)):
    surf1 = read_vtk(files[i])
    
    if os.path.exists('/media/fenqiang/DATA/unc/Data/NITRC/data/left/train/'+ files[i].split('/')[9] +'.lh.SphereSurf.Orig.Resample.vtk'):
        surf2 = read_vtk('/media/fenqiang/DATA/unc/Data/NITRC/data/left/train/'+ files[i].split('/')[9] +'.lh.SphereSurf.Orig.Resample.vtk')
        surf1['par_vec'] = surf2['par_fs_vec']
        write_vtk(surf1, files[i])
    


####################################################################
""" Resample label and write it into the Deep Learning registered sphere on BCP dataset """

from utils_vtk import resample_label

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.163842.moved.vtk'))
for i in range(len(files)):
    
    deformed_ver = read_vtk(files[i])
    deformed_ver = deformed_ver['vertices']
    orig_lbl_vec = read_vtk(files[i].replace('.moved.vtk', '.vtk'))
    
    if 'par_vec' in orig_lbl_vec.keys():
        orig_lbl_vec = orig_lbl_vec['par_vec']
        
        resampled_surf = read_vtk(files[i].replace('.moved.vtk','.moved.resampled.163842.vtk'))
        resampled_lbl = resample_label(deformed_ver, resampled_surf['vertices'], orig_lbl_vec)
        resampled_surf['par_vec'] = resampled_lbl
        
        write_vtk(resampled_surf, files[i].replace('.moved.vtk','.moved.resampled.163842.vtk'))
        
    
####################################################################
""" Interpolate par_fs from original resampled surface to original surface """

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/*/*.lh.SphereSurf.Orig.sphere.resampled.163842.vtk'))
files = [x for x in files if float(x.split('/')[-1].split('_')[1].split('.')[0]) >=450 and float(x.split('/')[-1].split('_')[1].split('.')[0]) <= 630]

for i in range(len(files)):
    surf1 = read_vtk(files[i])
    
    if 'par_vec' in surf1.keys():
        orig_surf = read_vtk('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/'+ files[i].split('/')[9] +'/surf/lh.sphere.vtk')
        orig_surf_lbl = resample_label(surf1['vertices'], orig_surf['vertices'], surf1['par_vec'])
        SD_reg_surf = read_vtk('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/'+ files[i].split('/')[9] +'/surf/lh.AlignedToBCPAtlas.sphere.vtk')
        resampled_surf = read_vtk('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/'+ files[i].split('/')[9] +'/surf/lh.AlignedToBCPAtlas.sphere.resampled.sucu.vtk')
        resampled_surf_lbl = resample_label(SD_reg_surf['vertices'], resampled_surf['vertices'], orig_surf_lbl)
        resampled_surf['par_vec'] = resampled_surf_lbl
        write_vtk(resampled_surf, '/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/'+ files[i].split('/')[9] +'/surf/lh.AlignedToBCPAtlas.sphere.resampled.sucu.vtk')
        
        
####################################################################
""" compute ineterpolated indices and weights for fixed 40962 sphere to alpha beta image """

shape = [256,256]
n_vertex = 2562
rotateds = [0,1,2]

for rotated in rotateds:
    sphere = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.'+ str(n_vertex)+'.rotated_'+ str(rotated) +'.vtk') 
    vertices = sphere['vertices'].astype(np.float64)
    tree = KDTree(vertices, leaf_size=10)  # build kdtree
    faces = sphere['faces']
    faces = faces[:,1:]
    neigh_orders = get_neighs_order('neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_'+ str(rotated) +'.mat')
    
    assert shape[0] == shape[1], "Shape should be square."
    
    inter_indices = np.zeros((shape[0]*shape[1],3)).astype(np.int64)
    inter_weights = np.zeros((shape[0]*shape[1],3))
    
    for p in range(shape[0]):
        print(p)
        beta = np.pi/(shape[1]-1)*p
        for q in range(shape[1]):
    #        print(q)
            alpha = 2*np.pi/(shape[0]-1)*q
            x = 100 * np.sin(beta) * np.cos(alpha)
            y = 100 * np.sin(beta) * np.sin(alpha)
            z = 100 * np.cos(beta)
            sphere_loc = np.array([x, y, z])
            inter_indices[p*shape[0]+q,:], inter_weights[p*shape[0]+q,:] = singleVertexInterpo(sphere_loc, vertices, tree, neigh_orders)
     
    np.save('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_'+ str(rotated) +'.npy', inter_indices)
    np.save('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_'+ str(rotated) +'.npy', inter_weights)


####################################################################
""" Write complete sphere with sulc and curv for NEW SD reg results using resampled sphere """

files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/*/surf/lh.NewResampledAlignedToBCPAtlas.sphere.vtk'))

s_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.163842.rotated_0.vtk')

for i in range(len(files)):
    s = read_vtk(files[i])
    s['sulc'] = np.loadtxt('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/'+ files[i].split('/')[9] +'/surf/lh.sphere.sulc.resampled.txt')
    s['curv'] = np.loadtxt('/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/'+ files[i].split('/')[9] +'/surf/lh.sphere.curv.resampled.txt')
    
    s2 = read_vtk('/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/'+ files[i].split('/')[9] +'/'+ files[i].split('/')[9] +'.lh.SphereSurf.Orig.sphere.resampled.163842.vtk')
    
    if 'par_vec' in s2.keys():
        s['par_vec'] = s2['par_vec']
    write_vtk(s, files[i])
    
    resample_sulc_curv = resampleSphereSurf(s['vertices'], s_163842['vertices'], np.hstack((s['sulc'][:,np.newaxis], s['curv'][:,np.newaxis])))
    
    moved_resample = {'vertices': s_163842['vertices'],
                      'faces': s_163842['faces'],
                      'sulc': resample_sulc_curv[:,0],
                      'curv': resample_sulc_curv[:,1]}
        
    if 'par_vec' in s2.keys():
        moved_resample['par_vec'] = resample_label(s['vertices'], moved_resample['vertices'], s['par_vec'])
    write_vtk(moved_resample, files[i].replace('.vtk','resampled.163842.vtk'))


##########################################################################
def maunal_label(in_dic, field, ind, label):
    label = np.asarray(label)
    orig_lbl = in_dic[field]
    orig_lbl[ind] = label
    in_dic[field] = orig_lbl
    
    return in_dic

files = ['/media/fenqiang/DATA/unc/Data/registration/data/MSM/MNBCP266437_574/surf/Curv.L.sphere.reg.sucu.vtk',
        '/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP266437_574/MNBCP266437_574.lh.SphereSurf.Orig.sphere.resampled.163842.moved.vtk',
        '/media/fenqiang/DATA/unc/Data/registration/data/preprocessed_npy/MNBCP266437_574/MNBCP266437_574.lh.SphereSurf.Orig.sphere.resampled.163842.vtk',
        '/media/fenqiang/DATA/unc/Data/registration/data/SD_registration/MNBCP266437_574/surf/lh.NewResampledAlignedToBCPAtlas.sphere.vtk']


for file in files:
    a = read_vtk(file)
    a = maunal_label(a, 'par_vec', [10301,
72513,
93990,
129464,
150229
], [20,180,140])
    write_vtk(a, file)









