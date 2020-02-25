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
import math, multiprocessing


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



def singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=7):
    
    _, top7_near_vertex_index = tree.query(vertex[np.newaxis,:], k=k) 
    candi_faces = []
    for t in itertools.combinations(np.squeeze(top7_near_vertex_index), 3):
        tmp = np.asarray(t)  # get the indices of the potential candidate triangles
        if isATriangle(neigh_orders, tmp):
             candi_faces.append(tmp)
    if candi_faces:
        candi_faces = np.asarray(candi_faces)
    else:
        if k > 20:
            print("cannot find candidate faces, top k shoulb be larger, function recursion, current k =", k)
        return singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=k+3)

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
    
    if tmp[index] > 1e-10: 
#        print("tmp[index] = ", tmp[index])
        if isInTriangle(vertex, vertices[candi_faces[index][0]], vertices[candi_faces[index][1]], vertices[candi_faces[index][2]]):
            assert False, "threshold should be smaller"
        else:
            if k > 20:
                print("candidate faces don't contain the correct one, top k shoulb be larger, function recursion, current k =", k)
            return singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=k+3)
    
    assert abs(ratio[index] - 1) < 0.005, "projected vertex should be near the vertex!" 
    w = np.array([area_BCP[index], area_ACP[index], area_ABP[index]])
    inter_weight = w / w.sum()
    
    return candi_faces[index], inter_weight


def singleVertexInterpo(vertex, vertices, tree, neigh_orders, feat):
    """
    Compute the three indices and weights for sphere interpolation at given position.
    
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
        
        if area_BCP + area_ACP + area_ABP - area_ABC > 1e-10:
             inter_indices, inter_weight = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders)
             
        else:
            inter_weight = np.array([area_BCP, area_ACP, area_ABP])
            inter_weight = inter_weight / inter_weight.sum()
            inter_indices = top3_near_vertex_index
       
    else:
        inter_indices, inter_weight = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders)
        
    return np.sum(np.multiply(feat[inter_indices], np.repeat(inter_weight[:,np.newaxis], feat.shape[1], axis=1)), axis=0)


def multiVertexInterpo(vertexs, vertices, tree, neigh_orders, feat):
    feat_inter = np.zeros((vertexs.shape[0], feat.shape[1]))
    for i in range(vertexs.shape[0]):
        feat_inter[i,:] = singleVertexInterpo(vertexs[i,:], vertices, tree, neigh_orders, feat)
    return feat_inter
    

def resampleSphereSurf(vertices_fix, vertices_inter, feat, neigh_orders=None):
    """
    vertices_fix: N*3, numpy array
    vertices_inter: unknown*3, numpy array, sphere to be interpolated
    feat: N*D, features to be interpolated
    """
#    template = read_vtk('/media/fenqiang/DATA/unc/Data/registration/presentation/regis_sulc_2562_3d_smooth0p33_phiconsis1_3model/training_10242/MNBCP107842_593.lh.SphereSurf.Orig.sphere.resampled.642.DL.origin_3.phi_resampled.2562.moved.sucu_resampled.2562.DL.origin_3.phi_resampled.10242.moved.vtk')
#    vertices_fix = template['vertices']
#    feat = template['sulc']
#    vertices_inter = read_vtk('/media/fenqiang/DATA/unc/Data/Template/Atlas-20200107-newsulc/18/18.lh.SphereSurf.10242.rotated_2.vtk')
#    vertices_inter = vertices_inter['vertices']
    
    assert vertices_fix.shape[0] == feat.shape[0], "vertices.shape[0] == feat.shape[0], error"
    assert vertices_fix.shape[1] == 3, "vertices size not right"
    
    vertices_fix = vertices_fix.astype(np.float64)
    vertices_inter = vertices_inter.astype(np.float64)
    feat = feat.astype(np.float64)
    
    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1
    
    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]
        
    if neigh_orders == None:
        neigh_orders = get_neighs_order('neigh_indices/adj_mat_order_'+ str(vertices_fix.shape[0]) +'_rotated_0.mat')
    
    feat_inter = np.zeros((vertices_inter.shape[0], feat.shape[1]))
    tree = KDTree(vertices_fix, leaf_size=10)  # build kdtree
    
    
    """ Single process, single thread: 163842: 54.5s, 40962: 12.7s, 10242: 3.2s, 2562: 0.8s """
#    for i in range(vertices_inter.shape[0]):
#        print(i)
#        feat_inter[i,:] = singleVertexInterpo(vertices_inter[i,:], vertices_fix, tree, neigh_orders, feat)
       

    """ multiple processes method: 163842: 9.6s, 40962: 2.8s, 10242: 1.0s, 2562: 0.28s """
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    vertexs_num_per_cpu = math.ceil(vertices_inter.shape[0]/cpus)
    results = []
    
    for i in range(cpus):
        results.append(pool.apply_async(multiVertexInterpo, args=(vertices_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:], vertices_fix, tree, neigh_orders, feat,)))

    pool.close()
    pool.join()

    for i in range(cpus):
        feat_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:] = results[i].get()
        
    return np.squeeze(feat_inter)
        
        
        
        
        
        
        
        
        