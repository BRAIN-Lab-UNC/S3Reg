#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:01:52 2020

@author: fenqiang
"""

import itertools
import torch
from sklearn.neighbors import KDTree
import numpy as np
import math

from utils import get_orthonormal_vectors, get_z_weight


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

    return rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, z_weight_0, z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, index_1_0, index_2_0, index_double_02, index_double_12, index_double_01, index_triple_computed


def convert2DTo3D(phi_2d, En, device):
    """
    phi_2d: N*2
    En: N*6
    """
    phi_3d = torch.zeros(len(En), 3).to(device)
    tmp = En * phi_2d.repeat(1,3)
    phi_3d[:,0] = tmp[:,0] + tmp[:,1]
    phi_3d[:,1] = tmp[:,2] + tmp[:,3]
    phi_3d[:,2] = tmp[:,4] + tmp[:,5]
    return phi_3d


def get_bi_inter(n_vertex, device):
    inter_indices_0 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_0.npy')
    inter_indices_0 = torch.from_numpy(inter_indices_0.astype(np.int64)).to(device)
    inter_weights_0 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_0.npy')
    inter_weights_0 = torch.from_numpy(inter_weights_0.astype(np.float32)).to(device)
    
    inter_indices_1 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_1.npy')
    inter_indices_1 = torch.from_numpy(inter_indices_1.astype(np.int64)).to(device)
    inter_weights_1 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_1.npy')
    inter_weights_1 = torch.from_numpy(inter_weights_1.astype(np.float32)).to(device)
    
    inter_indices_2 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_indices_'+ str(n_vertex) +'_2.npy')
    inter_indices_2 = torch.from_numpy(inter_indices_2.astype(np.int64)).to(device)
    inter_weights_2 = np.load('/media/fenqiang/DATA/unc/Data/registration/scripts/neigh_indices/img_weights_'+ str(n_vertex) +'_2.npy')
    inter_weights_2 = torch.from_numpy(inter_weights_2.astype(np.float32)).to(device)
    
    return (inter_indices_0, inter_weights_0), (inter_indices_1, inter_weights_1), (inter_indices_2, inter_weights_2)


def get_latlon_img(bi_inter, feat):
    inter_indices, inter_weights = bi_inter
    width = int(np.sqrt(len(inter_indices)))
    img = torch.sum(((feat[inter_indices.flatten()]).reshape(inter_indices.shape[0], inter_indices.shape[1], feat.shape[1])) * ((inter_weights.unsqueeze(2)).repeat(1,1,feat.shape[1])), 1)
    img = img.reshape(width, width, feat.shape[1])
    
    return img


def getEn(n_vertex, device):
    En_0 = get_orthonormal_vectors(n_vertex, rotated=0)
    En_0 = torch.from_numpy(En_0.astype(np.float32)).to(device)
    En_1 = get_orthonormal_vectors(n_vertex, rotated=1)
    En_1 = torch.from_numpy(En_1.astype(np.float32)).to(device)
    En_2 = get_orthonormal_vectors(n_vertex, rotated=2)
    En_2 = torch.from_numpy(En_2.astype(np.float32)).to(device)
    
    En_0 = En_0.reshape(n_vertex, 6)
    En_1 = En_1.reshape(n_vertex, 6)
    En_2 = En_2.reshape(n_vertex, 6)
    
    return En_0, En_1, En_2


def isATriangle(neigh_orders, face):
    """
    neigh_orders_163842: int, (N*7) x 1
    face: int, 3 x 1
    """
    neighs = neigh_orders[face[0]*7:face[0]*7+6]
    if face[1] not in neighs or face[2] not in neighs:
        return False
    neighs = neigh_orders[face[1]*7:face[1]*7+6]
    if face[2] not in neighs:
        return False
    return True


def singleVertexInterpo_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders, k=7):
    """
    Compute the three indices and weights for sphere interpolation at given position.
    
    moving_warp_phi_3d_i: torch.tensor, size: [3]
    distance: the distance from each fiexd vertices to the interpolation position
    """

    if k > 25:
        top1_near_vertex_index = tree.query(moving_warp_phi_3d_i_cpu, k=1)[1].squeeze(0)
        inter_weight = torch.tensor([1.0, 0.0, 0.0])
        inter_indices = np.asarray([top1_near_vertex_index[0], top1_near_vertex_index[0], top1_near_vertex_index[0]])
        return inter_indices, inter_weight

    top7_near_vertex_index = tree.query(moving_warp_phi_3d_i_cpu, k=k)[1].squeeze()
    candi_faces = []
    for t in itertools.combinations(top7_near_vertex_index, 3):
        tmp = np.asarray(t)  # get the indices of the potential candidate triangles
        if isATriangle(neigh_orders, tmp):
            candi_faces.append(tmp)
    if candi_faces:
        candi_faces = np.asarray(candi_faces)
    else:
        print("cannot find candidate faces, top k shoulb be larger, function recursion, current k =", k)
        return singleVertexInterpo_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders, k=k+3)
            
    orig_vertex_0 = fixed_xyz[candi_faces[:,0]]
    orig_vertex_1 = fixed_xyz[candi_faces[:,1]]
    orig_vertex_2 = fixed_xyz[candi_faces[:,2]]
    faces_normal = torch.cross(orig_vertex_1 - orig_vertex_0, orig_vertex_2 - orig_vertex_0, dim=1)    # normals of all the faces
    
    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
    ratio = torch.sum(orig_vertex_0 * faces_normal, axis=1)/torch.sum(moving_warp_phi_3d_i * faces_normal, axis=1)
    ratio = ratio.unsqueeze(1)
    moving_warp_phi_3d_i_proj = ratio * moving_warp_phi_3d_i  # intersection points
    
    # find the triangle face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole triangle
    area_BCP = torch.norm(torch.cross(orig_vertex_1 - moving_warp_phi_3d_i_proj, orig_vertex_2 - moving_warp_phi_3d_i_proj), 2, dim=1)/2.0
    area_ACP = torch.norm(torch.cross(orig_vertex_2 - moving_warp_phi_3d_i_proj, orig_vertex_0 - moving_warp_phi_3d_i_proj), 2, dim=1)/2.0
    area_ABP = torch.norm(torch.cross(orig_vertex_0 - moving_warp_phi_3d_i_proj, orig_vertex_1 - moving_warp_phi_3d_i_proj), 2, dim=1)/2.0
    area_ABC = torch.norm(faces_normal, 2, dim=1)/2.0
    
    min_area, index = torch.min(area_BCP + area_ACP + area_ABP - area_ABC, 0)
    if min_area > 1e-08:
        print("top k shoulb be larger, function recursion, current k =", k)
        return singleVertexInterpo_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders, k=k+3)
    
    assert abs(ratio[index] - 1) < 0.01, "projected vertex should be near the vertex!" 
    w = torch.stack((area_BCP[index], area_ACP[index], area_ABP[index]))
    inter_weight = w / w.sum()
    
    return candi_faces[index], inter_weight

            
def singleVertexInterpo(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders, fixed_sulc):
    """
    TODO
    """
    # using kdtree find top 3 and check if is a triangle: 0.13ms on cpu
    top3_near_vertex_index = tree.query(moving_warp_phi_3d_i_cpu, k=3)[1].squeeze()
    
    if isATriangle(neigh_orders, top3_near_vertex_index):
        # if the 3 nearest indices compose a triangle:
        top3_near_vertex_0 = fixed_xyz[top3_near_vertex_index[0]]
        top3_near_vertex_1 = fixed_xyz[top3_near_vertex_index[1]]
        top3_near_vertex_2 = fixed_xyz[top3_near_vertex_index[2]]
        
        # use formula p(x) = <p1,n>/<x,n> * x to calculate the intersection with the triangle face
        normal = torch.cross(top3_near_vertex_0-top3_near_vertex_2, top3_near_vertex_1-top3_near_vertex_2)
        moving_warp_phi_3d_i_proj = torch.dot(top3_near_vertex_0, normal)/torch.dot(moving_warp_phi_3d_i, normal) * moving_warp_phi_3d_i  # intersection points

        # compute the small triangle area and check if the intersection point is in the triangle
        area_BCP = torch.norm(torch.cross(top3_near_vertex_1 - moving_warp_phi_3d_i_proj, top3_near_vertex_2 - moving_warp_phi_3d_i_proj), 2)/2.0
        area_ACP = torch.norm(torch.cross(top3_near_vertex_2 - moving_warp_phi_3d_i_proj, top3_near_vertex_0 - moving_warp_phi_3d_i_proj), 2)/2.0
        area_ABP = torch.norm(torch.cross(top3_near_vertex_0 - moving_warp_phi_3d_i_proj, top3_near_vertex_1 - moving_warp_phi_3d_i_proj), 2)/2.0
        area_ABC = torch.norm(normal, 2)/2.0
        
        if area_BCP + area_ACP + area_ABP - area_ABC > 1e-08:
            inter_indices, inter_weight = singleVertexInterpo_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders) 
        else:
            inter_weight = torch.stack((area_BCP, area_ACP, area_ABP))
            inter_weight = inter_weight / inter_weight.sum()
            inter_indices = top3_near_vertex_index
    else:
        inter_indices, inter_weight = singleVertexInterpo_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders)
    
    return torch.mm(inter_weight.unsqueeze(0), fixed_sulc[inter_indices])


def resampleSphereSurf(fixed_xyz, moving_xyz, fixed_sulc, neigh_orders, device):
    """
    Interpolate moving points using fixed points and its feature
    
    moving_xyz, N*3, torch cuda tensor, points to be interpolated
    fixed_xyz:          N*3, torch cuda tensor, known fixed sphere points
    fixed_sulc:         N*3, torch cuda tensor, known feature corresponding to fixed points
    device:             'torch.device('cpu')', or torch.device('cuda:0'), or ,torch.device('cuda:1')
    
    """
#    if len(fixed_sulc.shape) == 1
    fixed_inter = torch.zeros((len(moving_xyz),fixed_sulc.shape[1]), dtype=torch.float32, device = device)
    
    # detach().cpu() cost ~0.2ms
    moving_warp_phi_3d_cpu = moving_xyz.detach().cpu().numpy()
    fixed_xyz_cpu = fixed_xyz.detach().cpu().numpy()
    neigh_orders = neigh_orders.detach().cpu().numpy()
    
    tree = KDTree(fixed_xyz_cpu, leaf_size=10)  # build kdtree
    
    """ Single process, single thread: 163842:  s, 40962:  s, 10242: 7.6s, 2562:  s """
    for i in range(len(moving_xyz)):
        fixed_inter[i] = singleVertexInterpo(moving_xyz[i], moving_warp_phi_3d_cpu[i,:][np.newaxis,:], tree, fixed_xyz, neigh_orders, fixed_sulc)
        
    return fixed_inter
    

def bilinear_interpolate(im, x, y):
    """
    im: 512*512*C
    """
    x = torch.clamp(x, 0.0001, im.shape[1]-1.0001)
    y = torch.clamp(y, 0.0001, im.shape[1]-1.0001)
    
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    Ia = im[ y0.long(), x0.long() ]
    Ib = im[ y1.long(), x0.long() ]
    Ic = im[ y0.long(), x1.long() ]
    Id = im[ y1.long(), x1.long() ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa.unsqueeze(1)*Ia + wb.unsqueeze(1)*Ib + wc.unsqueeze(1)*Ic + wd.unsqueeze(1)*Id

        
def bilinearResampleSphereSurfImg(vertices_inter, img, radius=1.0):
    """
    assume vertices_fix are on the standard icosahedron discretized spheres
    
    """
    
    width = img.shape[0]

    vertices_inter[:,2] = torch.clamp(vertices_inter[:,2].clone(), -0.9999999, 0.9999999)
    beta = torch.acos(vertices_inter[:,2]/radius)
    row = beta/(math.pi/(width-1))

    alpha = torch.zeros_like(beta)
    # prevent divide by 0
    tmp1 = (vertices_inter[:,0] == 0).nonzero(as_tuple=True)[0]
    vertices_inter[tmp1, 0] = 1e-15
    
    tmp1 = (vertices_inter[:,0] > 0).nonzero(as_tuple=True)[0]
#    print("len(tmp1): ", len(tmp1))
    alpha[tmp1] = torch.atan(vertices_inter[tmp1, 1]/vertices_inter[tmp1, 0])
    
    tmp2 = (vertices_inter[:,0] < 0).nonzero(as_tuple=True)[0]
#    print("len(tmp2): ", len(tmp2))
    alpha[tmp2] = torch.atan(vertices_inter[tmp2, 1]/vertices_inter[tmp2, 0]) + math.pi
    
    alpha = alpha + math.pi * 2
    alpha = torch.remainder(alpha, math.pi * 2)
    
    if len(tmp1) + len(tmp2) != len(vertices_inter):
        print("len(tmp1) + len(tmp2) != len(vertices_inter), subtraction is: ", len(tmp1) + len(tmp2) - len(vertices_inter))
    
    col = alpha/(2*math.pi/(width-1))
    
    feat_inter = bilinear_interpolate(img, col, row)
    
    return feat_inter 
        

def bilinearResampleSphereSurf(vertices_inter, feat, bi_inter, radius=1.0):
    """
    ONLY!! assume vertices_fix are on the standard icosahedron discretized spheres!!
    
    """
    inter_indices, inter_weights = bi_inter
    width = int(np.sqrt(len(inter_indices)))
    img = torch.sum(((feat[inter_indices.flatten()]).reshape(inter_indices.shape[0], inter_indices.shape[1], feat.shape[1])) * ((inter_weights.unsqueeze(2)).repeat(1,1,feat.shape[1])), 1)
    img = img.reshape(width, width, feat.shape[1])
    
    return bilinearResampleSphereSurfImg(vertices_inter, img)
    
    
    
    
    

#    """ multiple processes method: 163842: 9.6s, 40962: 2.8s, 10242: 1.0s, 2562: 0.28s """
#    pool = torch.multiprocessing.Pool()
#    num_processes = 4      # torch.multiprocessing.cpu_count()
#    vertexs_num_per_proc = math.ceil(len(moving_warp_phi_3d)/num_processes)
#    results = []
#    
#    for i in range(num_processes):
#        result = pool.apply_async(multiVertexInterpo, args=(moving_warp_phi_3d[i*vertexs_num_per_proc:(i+1)*vertexs_num_per_proc,:], moving_warp_phi_3d_cpu[i*vertexs_num_per_proc:(i+1)*vertexs_num_per_proc,:], tree, fixed_xyz, neigh_orders, fixed_sulc, device))
#        results.append(result)
#
#    pool.close()
#    pool.join()
#
#    for i in range(num_processes):
#        fixed_inter[i*vertexs_num_per_proc:(i+1)*vertexs_num_per_proc,:] = results[i].get()
#        
#        
#    et = time.time()
#    print((et-st)*1000, "ms")
        