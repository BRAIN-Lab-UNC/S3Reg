#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: fenqiang
"""

import numpy as np
import pyvista
import glob
import os
import copy
from functools import reduce

def read_vtk(in_file):
    """
    Read .vtk POLYDATA file
    
    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'curv', 'sulc', ...
    """

    polydata = pyvista.read(in_file)
 
    n_faces = polydata.n_faces
    vertices = np.array(polydata.points)  # get vertices coordinate
    
    # only for triangles polygons data
    faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
    assert len(faces)/4 == n_faces, "faces number is not consistent!"
    faces = np.reshape(faces, (n_faces,4))
    
    data = {'vertices': vertices,
            'faces': faces
            }
    
    point_arrays = polydata.point_arrays
    for key, value in point_arrays.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif  value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data
    

def write_vtk(in_dic, file):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    assert 'vertices' in in_dic, "output vtk data does not have vertices!"
    assert 'faces' in in_dic, "output vtk data does not have faces!"
    
    data = copy.deepcopy(in_dic)
    
    vertices = data['vertices']
    faces = data['faces']
    surf = pyvista.PolyData(vertices, faces)
    
    del data['vertices']
    del data['faces']
    for key, value in data.items():
        surf.point_arrays[key] = value

    surf.save(file, binary=False)  
    
    
def remove_field(data, *fields):
    """
    remove the field attribute in data
    
    fileds: list, strings to remove
    data: dic, vtk dictionary
    """
    for field in fields:
        if field in data.keys():
            del data[field]
    
    return data

def interpolation_sphere(fix):
    return 1
    



def resample_surf(data, template):
    """
    template = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
    
    resample data onto template,
    This is for resampling original surface to standard surface with 2562, 10242,
    40962, 163842 vertices. 
    If just resampling from 163842 to 40962, please use downsample_surf
    
    data: vtk dict, contains 'vertices': , 'faces': , and features: , ...
    template: resample template, standard 2562, 10242, or 40962, ... sphere
    """
    data_resample = {'vertices': template['vertices'],
            'faces': template['faces']
            }
    
    template_vertices = template['vertices'].astype(np.float64)
    orig_vertices = data['vertices'].astype(np.float64)
    orig_faces = data['faces']
    
    orig_vertex_1 = orig_vertices[orig_faces[:,1]]
    orig_vertex_2 = orig_vertices[orig_faces[:,2]]
    orig_vertex_3 = orig_vertices[orig_faces[:,3]]
    edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
    edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
    faces_normal = np.cross(edge_12, edge_13) * 100    # normals of all the faces
    
    resample_indices = np.zeros((len(template['vertices']),3)).astype(np.int32)
    resample_weights = np.zeros((len(template['vertices']),3)).astype(np.float64)
    
    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
    upper = np.sum(orig_vertex_1 * faces_normal, axis=1)
    for i in range(len(template_vertices)):
        print(i)
        X = template_vertices[i]
        lower = np.sum(X * faces_normal, axis=1)
        temp = upper/lower
        ratio = temp[:, np.newaxis]
        P = ratio * X  # intersection points
        
        # find the triangle face that the inersection is in, 
        # refer to https://blackpawn.com/texts/pointinpoly/default.html
        edge_1i = P - orig_vertex_1
        dot00 = np.sum(edge_12**2, axis=1)
        dot01 = np.sum(edge_12*edge_13, axis=1)
        dot02 = np.sum(edge_12*edge_1i, axis=1)
        dot11 = np.sum(edge_13**2, axis=1)
        dot12 = np.sum(edge_13*edge_1i, axis=1)
        temp = dot00 * dot11 - dot01 * dot01
        temp[np.argwhere(temp==0)] = 0.00000001
        invDenom = 1 / temp
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        w = u + v
        indices = reduce(np.intersect1d, (np.argwhere(u>=0), np.argwhere(v>=0), np.argwhere(w<=1)))
        
        assert (abs(abs(ratio[indices])-1) < 0.01).all(), "Intersection point should be inside the sphere, and near the sphere"
        if len(indices) == 2:
            assert ratio[indices[0]] * ratio[indices[1]] < 0, "Intersection points should be on both side!"
            index_0 = indices[0] if ratio[indices[0]] > 0 else indices[1]
        else:
            length = len(indices)
            flag = 0
            if ratio[indices[0]] > 0:
                for k in range(1, length):
                    if ratio[indices[k]] < 0:
                        flag = 1
                        index_0 = indices[0]
                        break
            else:
                for k in range(1, length):
                    if ratio[indices[k]] > 0:
                        flag = 1
                        index_0 = indices[k]
                        break
            if flag == 0:
                raise NotImplementedError('error') 
        
        resample_indices[i,:] = orig_faces[index_0,1:]
        A = orig_vertex_1[index_0]
        B = orig_vertex_2[index_0]
        C = orig_vertex_3[index_0]
        P = P[index_0]
        assert abs(np.dot(A-P, faces_normal[index_0])) < 1e-05, "Intersection should be on the triangle plane"
        area_ABP = np.sqrt(np.sum(np.cross(B-P, A-P)**2))/2
        area_ACP = np.sqrt(np.sum(np.cross(C-P, A-P)**2))/2
        area_BCP = np.sqrt(np.sum(np.cross(C-P, B-P)**2))/2
        area_ABC = np.sqrt(np.sum(np.cross(C-A, B-A)**2))/2
        assert abs(area_ABC - (area_ABP + area_ACP + area_BCP)) < 1e-06 , "Triangle area is not equal to 3 small triangles."
        
        resample_weights[i,:] = np.array([area_BCP/area_ABC, area_ACP/area_ABC, area_ABP/area_ABC])
        
        
        
    for feat_name in data:
        if feat_name != 'vertices' and feat_name != 'faces':
            feat_data = data[feat_name]
          
            
   
def resample_surf_v2(data, template):
    """
    template = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
    
    resample data onto template,
    This is for resampling original surface to standard surface with 2562, 10242,
    40962, 163842 vertices. 
    If just resampling from 163842 to 40962, please use downsample_surf
    
    data: vtk dict, contains 'vertices': , 'faces': , and features: , ...
    template: resample template, standard 2562, 10242, or 40962, ... sphere
    """
    data_resample = {'vertices': template['vertices'],
            'faces': template['faces']
            }
    
    template_vertices = template['vertices'].astype(np.float64)
    orig_vertices = data['vertices'].astype(np.float64)
    orig_faces = data['faces']
    
    resample_indices = np.zeros((len(template['vertices']),3)).astype(np.int32)
    resample_weights = np.zeros((len(template['vertices']),3)).astype(np.float64)
        
    for i in range(len(template_vertices)):
        print(i)
        
        X = template_vertices[i]
        
        candi_faces = []
        for j in range(len(orig_faces)):
            if abs(orig_vertices[orig_faces[j,1]][0] - X[0]) < 5 and abs(orig_vertices[orig_faces[j,1]][1] - X[1]) < 5 and abs(orig_vertices[orig_faces[j,1]][2] - X[2]) < 5:
                candi_faces.append(j)
        candi_faces = np.array(candi_faces)
        
        orig_vertex_1 = orig_vertices[orig_faces[candi_faces,1]]
        orig_vertex_2 = orig_vertices[orig_faces[candi_faces,2]]
        orig_vertex_3 = orig_vertices[orig_faces[candi_faces,3]]
        edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
        edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
        faces_normal = np.cross(edge_12, edge_13) * 100    # normals of all the faces
        
        # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
        upper = np.sum(orig_vertex_1 * faces_normal, axis=1)
        lower = np.sum(X * faces_normal, axis=1)
        temp = upper/lower
        ratio = temp[:, np.newaxis]
        P = ratio * X  # intersection points
        
        # find the triangle face that the inersection is in, 
        # refer to https://blackpawn.com/texts/pointinpoly/default.html
        edge_1p = P - orig_vertex_1
        dot00 = np.sum(edge_12**2, axis=1)
        dot01 = np.sum(edge_12*edge_13, axis=1)
        dot02 = np.sum(edge_12*edge_1p, axis=1)
        dot11 = np.sum(edge_13**2, axis=1)
        dot12 = np.sum(edge_13*edge_1p, axis=1)
        temp = dot00 * dot11 - dot01 * dot01
        temp[np.argwhere(temp==0)] = 0.00000001
        invDenom = 1 / temp
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        w = u + v
        index = reduce(np.intersect1d, (np.argwhere(u>=0), np.argwhere(v>=0), np.argwhere(w<=1)))
        
#        assert len(index) == 1, "Intersection should be in only one triangle."
        if len(index) != 1:
            print(i)
            print(index)
            raise NotImplementedError('error, TODO')
        assert abs(ratio[index] - 1) < 0.001, "Intersection point should be inside the sphere, and near the sphere"
        
        resample_indices[i,:] = orig_faces[candi_faces[index],1:]
        
        A = orig_vertex_1[index]
        B = orig_vertex_2[index]
        C = orig_vertex_3[index]
        P = P[index]
        assert abs(np.sum((A-P) * faces_normal[index])) < 1e-08, "Intersection should be on the triangle plane"
        area_ABP = np.sqrt(np.sum(np.cross(B-P, A-P)**2))/2
        area_ACP = np.sqrt(np.sum(np.cross(C-P, A-P)**2))/2
        area_BCP = np.sqrt(np.sum(np.cross(C-P, B-P)**2))/2
        area_ABC = np.sqrt(np.sum(np.cross(C-A, B-A)**2))/2
        assert abs(area_ABC - (area_ABP + area_ACP + area_BCP)) < 1e-06 , "Triangle area is not equal to 3 small triangles."
        
        resample_weights[i,:] = np.array([area_BCP/area_ABC, area_ACP/area_ABC, area_ABP/area_ABC])    
    

        
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
## Make data
#u = np.linspace(0, 2 * np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#x = 100 * np.outer(np.cos(u), np.sin(v))
#y = 100 * np.outer(np.sin(u), np.sin(v))
#z = 100 * np.outer(np.ones(np.size(u)), np.cos(v))
#
## Plot the surface
#ax.plot_wireframe(x, y, z, color='b')
#
#ax.set_xlim3d(-200, 200)
#ax.set_ylim3d(-200, 200)
#ax.set_zlim3d(-200, 200)
##        a = np.array([1,0,0])
##        b = np.array([0,2,0])
##        c = np.array([0,0,3])
#ax.scatter(a[0], a[1], a[2], s=50, c='r', marker='o')
#ax.scatter(b[0], b[1], b[2], s=50, c='r', marker='o')
#ax.scatter(c[0], c[1], c[2], s=50, c='r', marker='o')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#ab = b - a
#ac = c - a
#normal = np.cross(ab,ac)
#x = np.linspace(-200,200,100)
#y = np.linspace(-200,200,100)
#X,Y = np.meshgrid(x,y)
#Z = (np.dot(normal, c) - normal[0]*X - normal[1]*Y)/normal[2]
#ax.plot_wireframe(X, Y, Z)
#
#a_1 = a + (a-b) * 100
#b_1 = b + (b-a) * 100
#c_1 = c + (c-b) * 100
#ax.scatter(a_1[0], a_1[1], a_1[2], s=50, c='r', marker='o')
#ax.scatter(b_1[0], b_1[1], b_1[2], s=50, c='r', marker='o')
#ax.scatter(c_1[0], c_1[1], c_1[2], s=50, c='r', marker='o')
#
#
#plt.show()
        