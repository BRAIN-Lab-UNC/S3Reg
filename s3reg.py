#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:01:40 2020

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""
import torch
import numpy as np
import glob
import argparse
import time
import os

# from s3pipe.utils.vtk import write_vtk, read_vtk
from s3pipe.surface.s3reg import readRegConfig, createRegConfig, regOnSingleLevel
from s3pipe.utils.interp_numpy import resampleSphereSurf
from s3pipe.utils.utils import S3_normalize

# abspath = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Cortical surface registration based on deep learning.\n @author: Fenqiang Zhao, Contact: zhaofenqiang0221@gmail.com',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hemisphere', '-hemi', default='lh',
                        choices=['lh', 'rh'], 
                        help="Specify the hemisphere for registration, lh or rh.")
    parser.add_argument('--config', '-c', default=None,  required=True,
                        help="Specify the config file for registration. " + \
                            "An example config file can be found at https://github.com/BRAIN-Lab-UNC/S3Reg/blob/master/config/regConfig_3level_sucu.yaml")
    parser.add_argument('--moving', '-i',  required=True, 
                        help='Path to input vtk surface(s). This could be a list of '+\
                            'multiple files that can be interpreted by glob, e.g., /media/data/sub-*/sub-*.lh.sphere.Resp163842.vtk'+\
                            'or just a single file, e.g., /path/to/your/file.vtk')
    parser.add_argument('--atlas', '-a', default=None, help='The atlas file to be registered to. '+\
                        'Either atlas or age should be given. Atlas has higher priority.')
    parser.add_argument('--age', '-age', default=None, help='Age in month used to find the UNC atlas if you do not have an atlas.'+\
                        ' If you have age in days or years, please convert it to months.'+\
                            ' It will choose the closest model and atlas based '+\
                                'on the input age otherwise the 72 months atlas and model will be chosen.')
    parser.add_argument('--moved', '-o',  default='[input].AlignToUNCAtlas.vtk',
                        help='Path to ouput surface(s).')
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='The device for running the model.')
    parser.add_argument('--smooth', '-s',  default='large', choices=['small', 'medium', 'large'],
                        help='expert option, smoothness for the deformation field')

    args =  parser.parse_args()
    hemi = args.hemisphere
    moving = args.moving
    atlas = args.atlas
    age = args.age
    moved = args.moved
    config = args.config
    device = args.device
    smooth = args.smooth
    print('----------------------------------------------------------------------------')
    print('Surface registration using S3Reg...')
    print('Hemi:', hemi)
    
    # check config file
    if config == None:
        raise NotImplementedError('Need to specify config file.')
    print("Config file: ", config)
    # load configuration
    config = readRegConfig(config)
    
    # check device
    if device == 'GPU':
        device = torch.device('cuda:0')
    elif device =='CPU':
        device = torch.device('cpu')
    else:
        raise NotImplementedError('Only support GPU or CPU device')
    config['device'] = device

    # check atlas
    if atlas == None and age == None:
        raise NotImplementedError('Need to specify an atlas or age to be the reference that the surface will be aligned to.')
    if atlas != None:
        print('Atlas file: ', atlas)
    else:
        age = float(age)
        if age < 2:
            age = '01'
        elif age < 4.5:
            age = '03'
        elif age < 7.5:
            age = '06'
        elif age < 10.5:
            age = '09'
        elif age < 15:
            age = '12'
        elif age < 21:
            age = '18'
        elif age < 30:
            age = '24'
        elif age < 42:
            age = '36'
        elif age < 54:
            age = '48'
        elif age < 66:
            age = '60'
        else:
            age = '72'
        atlas = '/media/ychenp/DATA/unc/Data/Template/UNC-Infant-Cortical-Surface-Atlas/'+ age +'/'+ age +'_'+ config['hemi'] +'.SphereSurf.vtk'
        print('Atlas file: ', atlas)
    config['atlas_file'] = atlas

    # check moving files and out_file_names
    try:
        moving_surf = read_vtk(moving)
        files = [moving]
    except:
        files = sorted(glob.glob(moving))
    print('Moving surface(s): ')
    for tmp in files:
        print(tmp)
    if moved == '[input].AlignToUNCAtlas.vtk':
        moved_files = [x.replace('.vtk', '.AlignToUNCAtlas.vtk') for x in files]
    else:
        moved_files = [moved]
    print('Moved surface(s) will be: ')
    for tmp in moved_files:
        print(tmp)
        
    if smooth == 'small':
        config['weight_smooth'] = [5]
    elif smooth == 'medium':
        config['weight_smooth'] = [10]
    elif smooth == 'large':
        config['weight_smooth'] = [15]
    else:
        raise NotImplementedError('cannot find correct model corresponding to the given smoothness parameter')
    
    # create configuration
    config['val'] = True
    config['model_dir'] = abspath + '/' + 'pretrained_models'
    config = createRegConfig(config)

    print('Pretrained models have: ')
    for models in config['modelss']:
        for model in models:
            print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for models in config['modelss']:
        for model in models:
            model.eval()
    
    
    template_surf = read_vtk('/media/ychenp/fq/S3_pipeline/SphericalUNetPackage/sphericalunet/utils/neigh_indices/sphere_'+ \
                        str(config['n_vertexs'][-1]) +'_rotated_0.vtk')
    for file in files:
        t1 = time.time()
        print('----------------------------------------------------------------------------')
        print("Start registration for: ", file)
        if file.split('.')[-1] == 'vtk':
            surf = read_vtk(file)
            tmp = np.concatenate((surf['sulc'][:, np.newaxis], surf['curv'][:, np.newaxis]), axis=1)
        else:
            tmp = np.load(file)
        sulc = S3_normalize(tmp[:, 0])[0:config['n_vertexs'][-1]]
        curv = S3_normalize(tmp[:, 1])[0:config['n_vertexs'][-1]]
        data = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
        moving_data = torch.from_numpy(data.astype(np.float32)).to(device)
            
        total_deform = 0.
        with torch.no_grad():
            for i_level in range(config['n_levels']):
                total_deform = regOnSingleLevel(i_level, moving_data, config, total_deform)
    
        if len(template_surf['vertices']) == len(surf['vertices']):
            total_deform = total_deform.cpu().numpy() * 100.0
        else:
            total_deform = resampleSphereSurf(template_surf['vertices'], surf['vertices'],
                                              total_deform.cpu().numpy() * 100.0)
        
        if file.split('.')[-1] == 'vtk':
            surf['vertices'] = total_deform
        else:
            surf = read_vtk(file.replace('.npy', '.vtk'))
            surf['vertices'] = total_deform
        
        print("Writing moved surface to", moved_files[files.index(file)])
        write_vtk(surf, moved_files[files.index(file)])
        
        t2 = time.time()
        print("Registration done! Cost: ", (t2-t1), "s")   
    
        
 


