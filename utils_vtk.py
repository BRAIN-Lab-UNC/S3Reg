#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: fenqiang
"""

import numpy as np
import pyvista
import copy


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
        