#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:36:21 2020

@author: fenqiang
"""
import numpy as np
import itertools
from sklearn.neighbors import KDTree
from utils import get_neighs_order
#import time

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


def projectVertex(vertex, v0, v1, v2):
    normal = np.cross(v0 - v2, v1 - v2)
    if np.linalg.norm(normal) == 0:
        normal = v0
    ratio = v0.dot(normal)/vertex.dot(normal)
    vertex_proj = ratio * vertex
    return vertex_proj


def isOnSameSide(P, v0 , v1, v2):
    """
    Check if P and v0 is on the same side
    """
    edge_12 = v2 - v1
    tmp0 = P - v1
    tmp1 = v0 - v1
    
    edge_12 = edge_12 / np.linalg.norm(edge_12)
    tmp0 = tmp0 / np.linalg.norm(tmp0)
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    
    vec1 = np.cross(edge_12, tmp0)
    vec2 = np.cross(edge_12, tmp1)
    
    return vec1.dot(vec2) >= 0


def isInTriangle(vertex, v0, v1, v2):
    """
    Check if the vertices is in the triangle composed by v0 v1 v2
    vertex: N*3, check N vertices at the same time
    v0: (3,)
    v1: (3,)
    v2: (3,)
    """
    # Project point onto the triangle plane
    P = projectVertex(vertex, v0, v1, v2)
          
    return isOnSameSide(P, v0, v1, v2) and isOnSameSide(P, v1, v2, v0) and isOnSameSide(P, v2, v0, v1)



def sphere_interpolation_7(vertex, vertices, tree, neigh_orders, k=7):
    
    _, top7_near_vertex_index = tree.query(vertex[np.newaxis,:], k=k) 
    candi_faces = []
    for t in itertools.combinations(np.squeeze(top7_near_vertex_index), 3):
        tmp = np.asarray(t)  # get the indices of the potential candidate triangles
        if isATriangle(neigh_orders, tmp):
             candi_faces.append(tmp)
    candi_faces = np.asarray(candi_faces)

    orig_vertex_1 = vertices[candi_faces[:,0]]
    orig_vertex_2 = vertices[candi_faces[:,1]]
    orig_vertex_3 = vertices[candi_faces[:,2]]
    edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
    edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
    faces_normal = np.cross(edge_12, edge_13)    # normals of all the faces
    faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:,np.newaxis]

    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
    temp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
    ratio = temp[:, np.newaxis]
    P = ratio * vertex  # intersection points

    # find the triangle face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole one
    area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
    area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
    area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
    area_ABC = np.linalg.norm(faces_normal, axis=1)/2.0
    
    tmp = area_BCP + area_ACP + area_ABP - area_ABC
    index = np.argmin(tmp)
    assert abs(ratio[index] - 1) < 0.005, "projected vertex should be near the vertex!" 
    if tmp[index] > 1e-08: 
        print("tmp[index] = ", tmp[index])
        if isInTriangle(vertex, vertices[candi_faces[index][0]], vertices[candi_faces[index][1]], vertices[candi_faces[index][2]]):
            assert False, "threshold should be smaller"
        else:
            print("top k shoulb be larger, function recursion, current k =", k)
            return sphere_interpolation_7(vertex, vertices, tree, neigh_orders, k=k+2)
    
    w = np.array([area_BCP[index], area_ACP[index], area_ABP[index]])
    inter_weight = w / w.sum()
    
    return candi_faces[index], inter_weight


def sphere_interpolation(vertex, vertices, tree, neigh_orders):
    """
    Compute the three indices and weights for sphere interpolation at given position.
    
    moving_warp_phi_3d_i: torch.tensor, size: [3]
    distance: the distance from each fiexd vertices to the interpolation position
    """
    _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3) 
    top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
    if isATriangle(neigh_orders, top3_near_vertex_index):
        v0 = vertices[top3_near_vertex_index[0]]
        v1 = vertices[top3_near_vertex_index[1]]
        v2 = vertices[top3_near_vertex_index[2]]
        normal = np.cross(v1-v2, v0-v2)
        
        vertex_proj = v0.dot(normal)/vertex.dot(normal) * vertex
        
        area_BCP = np.linalg.norm(np.cross(v2-vertex_proj, v1-vertex_proj))/2.0
        area_ACP = np.linalg.norm(np.cross(v2-vertex_proj, v0-vertex_proj))/2.0
        area_ABP = np.linalg.norm(np.cross(v1-vertex_proj, v0-vertex_proj))/2.0
        area_ABC = np.linalg.norm(normal)/2.0
        
        if area_BCP + area_ACP + area_ABP - area_ABC > 1e-08:
            return sphere_interpolation_7(vertex, vertices, tree, neigh_orders)
             
        else:
            inter_weight = np.array([area_BCP, area_ACP, area_ABP])
            inter_weight = inter_weight / inter_weight.sum()
            return top3_near_vertex_index, inter_weight
       
    else:
        return sphere_interpolation_7(vertex, vertices, tree, neigh_orders)

    
def resample_sphere_surf(vertices, vertices_inter, feat):
    """
    vertices: N*3, numpy array
    faces: N*3, numpy array,
    template_vertices: unknown*3, numpy array, fixed sphere to be interpolated
    feat: N*D, features to be interpolated
    """
#    template = read_vtk('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_2562_3d_smooth1_3model/MNBCP124529_588.lh.SphereSurf.Orig.sphere.resampled.2562.DL.moved_2.vtk')
#    vertices = template['vertices']
#    feat = template['sulc']
#    vertices_inter = read_vtk('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_10242_3d_smooth1p4_phiconsis_3model/MNBCP000178_494.lh.SphereSurf.Orig.sphere.resampled.10242.DL.moved_3.vtk')
#    vertices_inter = vertices_inter['vertices']
    
    assert vertices.shape[0] == feat.shape[0], "vertices.shape[0] == feat.shape[0], error"
    assert vertices.shape[1] == 3, "vertices size not right"
    
    vertices = vertices.astype(np.float64)
    vertices_inter = vertices_inter.astype(np.float64)
    feat = feat.astype(np.float64)
    
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1
    
    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]
        
    n_vertex = vertices.shape[0]
    neigh_orders = get_neighs_order('neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_0.mat')
    feat_inter = np.zeros((vertices_inter.shape[0], feat.shape[1]))
    
    tree = KDTree(vertices, leaf_size=10)  # build kdtree
    for i in range(vertices_inter.shape[0]):
#        print(i)
        inter_indices, inter_weight = sphere_interpolation(vertices_inter[i,:], vertices, tree, neigh_orders)
        feat_inter[i,:] = np.sum(np.multiply(feat[inter_indices], np.repeat(inter_weight[:,np.newaxis], feat.shape[1], axis=1)), axis=0)
        
    return np.squeeze(feat_inter)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        