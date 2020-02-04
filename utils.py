#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:55:33 2018

@author: zfq
"""

import scipy.io as sio 
import numpy as np
import glob
import os
from numpy import median
from utils_vtk import read_vtk


def Get_neighs_order(rotated=0):
    neigh_orders_163842 = get_neighs_order('neigh_indices/adj_mat_order_163842_rotated_' + str(rotated) + '.mat')
    neigh_orders_40962 = get_neighs_order('neigh_indices/adj_mat_order_40962_rotated_' + str(rotated) + '.mat')
    neigh_orders_10242 = get_neighs_order('neigh_indices/adj_mat_order_10242_rotated_' + str(rotated) + '.mat')
    neigh_orders_2562 = get_neighs_order('neigh_indices/adj_mat_order_2562_rotated_' + str(rotated) + '.mat')
    neigh_orders_642 = get_neighs_order('neigh_indices/adj_mat_order_642_rotated_' + str(rotated) + '.mat')
    neigh_orders_162 = get_neighs_order('neigh_indices/adj_mat_order_162_rotated_' + str(rotated) + '.mat')
    neigh_orders_42 = get_neighs_order('neigh_indices/adj_mat_order_42_rotated_' + str(rotated) + '.mat')
    neigh_orders_12 = get_neighs_order('neigh_indices/adj_mat_order_12_rotated_' + str(rotated) + '.mat')
    
    return neigh_orders_163842, neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders


def Get_upconv_index(rotated=0):
    
    upconv_top_index_163842, upconv_down_index_163842 = get_upconv_index('neigh_indices/adj_mat_order_163842_rotated_' + str(rotated) + '.mat')
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index('neigh_indices/adj_mat_order_40962_rotated_' + str(rotated) + '.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index('neigh_indices/adj_mat_order_10242_rotated_' + str(rotated) + '.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index('neigh_indices/adj_mat_order_2562_rotated_' + str(rotated) + '.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index('neigh_indices/adj_mat_order_642_rotated_' + str(rotated) + '.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index('neigh_indices/adj_mat_order_162_rotated_' + str(rotated) + '.mat')
    
    return upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 


def get_upconv_index(order_path):  
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index


def compute_weight():
    folder = 'neigh_indices/90/raw'
    files = sorted(glob.glob(os.path.join(folder, '*.label')))
    
    labels = np.zeros((len(files),10242))
    for i in range(len(files)):
        file = files[i]
        label = sio.loadmat(file)
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        label = label.astype(np.float64)
        labels[i,:] = label
        
    num = np.zeros(36)
    for i in range(36):
        num[i] = len(np.where(labels == i)[0])
       
    num = num/sum(num) 
    num = median(num)/num
    print(num)

    return num


def get_par_fs_to_36():
    """ Preprocessing for parcellatiion label """
    file = '/media/fenqiang/DATA/unc/Data/NITRC/data/left/train/MNBCP107842_809.lh.SphereSurf.Orig.Resample.vtk'
    data = read_vtk(file)
    par_fs = data['par_fs']
    par_fs_label = np.sort(np.unique(par_fs))
    par_dic = {}
    for i in range(len(par_fs_label)):
        par_dic[par_fs_label[i]] = i
    return par_dic


def get_par_36_to_fs_vec():
    """ Preprocessing for parcellatiion label """
    file = '/media/fenqiang/DATA/unc/Data/NITRC/data/left/train/MNBCP107842_809.lh.SphereSurf.Orig.Resample.vtk'
    data = read_vtk(file)
    par_fs = data['par_fs']
    par_fs_vec = data['par_fs_vec']
    par_fs_to_36 = get_par_fs_to_36()
    par_36_to_fs = dict(zip(par_fs_to_36.values(), par_fs_to_36.keys()))
    par_36_to_fs_vec = {}
    for i in range(len(par_fs_to_36)):
        par_36_to_fs_vec[i] = par_fs_vec[np.where(par_fs == par_36_to_fs[i])[0][0]]
    return par_36_to_fs_vec


def get_orthonormal_vectors(n_ver, rotated=0):
    """
    get the orthonormal vectors
    
    n_vec: int, number of vertices, 42,162,642,2562,10242,...
    rotated: 0: original, 1: rotate 90 degrees along y axis, 2: then rotate 90 degrees along x axis
    return orthonormal matrix, shape: n_vec * 3 * 2
    """
    assert type(n_ver) is int, "n_ver, the number of vertices should be int type"
    assert n_ver in [42,162,642,2562,10242,40962,163842], "n_ver, the number of vertices should be the one of [42,162,642,2562,10242,40962,163842]"
    assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
   
    template = read_vtk('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/sphere_'+str(n_ver)+'_rotated_'+str(rotated)+'.vtk')
    vertices = template['vertices'].astype(np.float64)
    
    x_0 = np.argwhere(vertices[:,0]==0)
    y_0 = np.argwhere(vertices[:,1]==0)
    inter_ind = np.intersect1d(x_0, y_0)
    
    En_1 = np.cross(np.array([0,0,1]), vertices)
    En_1[inter_ind] = np.array([1,0,0])
    En_2 = np.cross(vertices, En_1)
    
    En_1 = En_1/np.repeat(np.sqrt(np.sum(En_1**2, axis=1))[:,np.newaxis], 3, axis=1)  # normalize to unit orthonormal vector
    En_2 = En_2/np.repeat(np.sqrt(np.sum(En_2**2, axis=1))[:,np.newaxis], 3, axis=1)  # normalize to unit orthonormal vector
    En = np.transpose(np.concatenate((En_1[np.newaxis,:], En_2[np.newaxis,:]), 0), (1,2,0))
    
    return En

def get_patch_indices(n_vertex):
    """
    return all the patch indices and weights
    """
    indices_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/*_indices.npy'))
    weights_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/*_weights.npy'))
    
    assert len(indices_files) == len(weights_files), "indices files should have the same number as weights number"
    assert len(indices_files) == 163842, "Indices should have dimension 163842 "
    
    indices = [x.split('/')[-1].split('_')[0] for x in indices_files]
    weights = [x.split('/')[-1].split('_')[0] for x in weights_files]
    assert indices == weights, "indices are not consistent with weights!"
    
    indices = [int(x) for x in indices]
    weights = [int(x) for x in weights]
    assert indices == weights, "indices are not consistent with weights!"
    
    indices = np.zeros((n_vertex, 4225, 3)).astype(np.int32)
    weights = np.zeros((n_vertex, 4225, 3))
    
    for i in range(n_vertex):
        indices_file = '/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/'+ str(i) + '_indices.npy'
        weights_file = '/media/fenqiang/DATA/unc/Data/registration/scripts/patch_inter/'+ str(i) + '_weights.npy'
        indices[i,:,:] = np.load(indices_file)
        weights[i,:,:] = np.load(weights_file)
    
    return indices, weights
        

def get_z_weight(n_vertex, rotated=0):
    sphere = read_vtk('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/sphere_'+str(n_vertex)+'_rotated_'+str(rotated)+'.vtk')
    fixed_xyz = sphere['vertices']/100.0
    z_weight = np.abs(fixed_xyz[:,2])
    index_1 = (z_weight <= 1/np.sqrt(2)).nonzero()[0]
    index_2 = (z_weight > 1/np.sqrt(2)).nonzero()[0]
    assert len(index_1) + len(index_2) == n_vertex, "error"
    z_weight[index_1] = 1.0
    z_weight[index_2] = z_weight[index_2] * (-1./(1.-1./np.sqrt(2))) + 1./(1.-1./np.sqrt(2))
    
    return z_weight


def check_intersect_edges(vertices, faces):
    """
    vertices: N * 3, numpy array, float 64
    faces: (N*2-4) * 3, numpy array, float 64
    """
    assert 2*len(vertices)-4 == len(faces), "vertices are not consistent with faces."

        
    
    
    