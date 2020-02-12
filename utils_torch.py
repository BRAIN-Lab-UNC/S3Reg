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



def sphere_interpolation_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders, k=7):
    """
    Compute the three indices and weights for sphere interpolation at given position.
    
    moving_warp_phi_3d_i: torch.tensor, size: [3]
    distance: the distance from each fiexd vertices to the interpolation position
    """

    top7_near_vertex_index = tree.query(moving_warp_phi_3d_i_cpu, k=k)[1].squeeze()
    candi_faces = []
    for t in itertools.combinations(top7_near_vertex_index, 3):
        tmp = np.asarray(t)  # get the indices of the potential candidate triangles
        if isATriangle(neigh_orders, tmp):
            candi_faces.append(tmp)
    candi_faces = np.asarray(candi_faces)     
            
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
    
    min_area, index = torch.min(area_BCP + area_ACP + area_ABP - area_ABC,0)
    assert abs(ratio[index] - 1) < 0.01, "projected vertex should be near the vertex!" 
    if min_area > 1e-06:
        print("top k shoulb be larger, function recursion, current k =", k)
        return sphere_interpolation_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders, k=k+2)
    
    w = torch.stack((area_BCP[index], area_ACP[index], area_ABP[index]))
    inter_weight = w / w.sum()
    
    return candi_faces[index], inter_weight

            
def sphere_interpolation(moving_warp_phi_3d, fixed_xyz, fixed_sulc, neigh_orders, device):
    """
    Interpolate moving points using fixed points and its feature
    
    moving_warp_phi_3d: points to be interpolated
    fixed_xyz:          known fixed sphere points
    fixed_sulc:         known feature corresponding to fixed points
    device:             'torch.device('cpu')', or torch.device('cuda:0'), or ,torch.device('cuda:1')
    
    """
    fixed_inter = torch.zeros((len(moving_warp_phi_3d),1), dtype=torch.float32, device = device)
    
    # detach().cpu() cost ~0.2ms
    moving_warp_phi_3d_cpu = moving_warp_phi_3d.detach().cpu().numpy()
    fixed_xyz_cpu = fixed_xyz.detach().cpu().numpy()
    neigh_orders = neigh_orders.detach().cpu().numpy()
    
    tree = KDTree(fixed_xyz_cpu, leaf_size=10)  # build kdtree
    
    for i in range(len(moving_warp_phi_3d)):
        
        moving_warp_phi_3d_i = moving_warp_phi_3d[i]
        moving_warp_phi_3d_i_cpu = moving_warp_phi_3d_cpu[i,:][np.newaxis,:]

        # using kdtree find top 3 and check if is a triangle: 0.13ms on cpu
        top3_near_vertex_index = tree.query(moving_warp_phi_3d_i_cpu, k=3)[1].squeeze()
        
        if isATriangle(neigh_orders, top3_near_vertex_index):
            # if the 3 nearest indices compose a triangle:
            top3_near_vertex_0 = fixed_xyz[top3_near_vertex_index[0]]
            top3_near_vertex_1 = fixed_xyz[top3_near_vertex_index[1]]
            top3_near_vertex_2 = fixed_xyz[top3_near_vertex_index[2]]
            
            # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with the triangle face
            normal = torch.cross(top3_near_vertex_0-top3_near_vertex_2, top3_near_vertex_1-top3_near_vertex_2)
            moving_warp_phi_3d_i_proj = torch.dot(top3_near_vertex_0, normal)/torch.dot(moving_warp_phi_3d_i, normal) * moving_warp_phi_3d_i  # intersection points
    
            # compute the small triangle area and check if the intersection point is in the triangle
            area_BCP = torch.norm(torch.cross(top3_near_vertex_1 - moving_warp_phi_3d_i_proj, top3_near_vertex_2 - moving_warp_phi_3d_i_proj), 2)/2.0
            area_ACP = torch.norm(torch.cross(top3_near_vertex_2 - moving_warp_phi_3d_i_proj, top3_near_vertex_0 - moving_warp_phi_3d_i_proj), 2)/2.0
            area_ABP = torch.norm(torch.cross(top3_near_vertex_0 - moving_warp_phi_3d_i_proj, top3_near_vertex_1 - moving_warp_phi_3d_i_proj), 2)/2.0
            area_ABC = torch.norm(normal, 2)/2.0
            
            if abs(area_BCP + area_ACP + area_ABP - area_ABC) > 1e-06:
                inter_indices, inter_weight = sphere_interpolation_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders) 
            else:
                inter_weight = torch.stack((area_BCP, area_ACP, area_ABP))
                inter_weight = inter_weight / inter_weight.sum()
                inter_indices = top3_near_vertex_index
        else:
            inter_indices, inter_weight = sphere_interpolation_7(moving_warp_phi_3d_i, moving_warp_phi_3d_i_cpu, tree, fixed_xyz, neigh_orders)
        
        fixed_inter[i] = torch.mm(inter_weight.unsqueeze(0), fixed_sulc[inter_indices])
        
    return fixed_inter
