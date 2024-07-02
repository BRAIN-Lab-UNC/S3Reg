#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

import time
import numpy as np
import argparse
import glob

from s3pipe.utils.vtk import read_vtk, write_vtk
from s3pipe.utils.utils import S3_normalize
from s3pipe.utils.interp_numpy import get_latlon_img, get_bi_inter, resampleSphereSurf
from s3pipe.surface.s3reg import initialRigidAlign, get_rot_mat_zyx
from s3pipe.models.models import RigidAlignNet
from s3pipe.utils.interp_torch import  get_rot_mat_zyx_torch


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='rigidly align a surface to an atlas based on geometric feature average conveity (i.e., sulc), \n' + 
                                     'the surface .vtk file should contain the sulc field. Note the deep learning model was only trained on 417 BCP surfaces, \n'+
                                     'so the resutls may be not as good as iterative_search method sometimes. In future, need to fine-tune the \n'+
                                     'model using more data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--moving_file', '-m', default=None,
                        help="filename of the moving surface. Either moving_file or moving_file_pattern is required.")
    parser.add_argument('--moving_file_pattern', default=None,
                        help="filename pattern of the the moving surfaces, e.g., '/path/to/my/file/sub*/sub*.vtk'. "+\
                        " This is for processing multiple files since loading the pretrained model once is faster.")
    parser.add_argument('--hemi', choices=['lh', 'rh'], required=True)
    parser.add_argument('--atlas', '-a', 
                        default='/proj/ganglilab/users/Fenqiang/Atlas/72/72_?h.SphereSurf.vtk',
                        help="filename of atlas. The atlas should be resampled to ico sphere with 163842 vertices, such as the atlases here https://www.nitrc.org/projects/infantsurfatlas.")
    parser.add_argument('--out_name', '-o', 
                        default=None, 
                        help="filename of output rigid aligned surface, if not given, will be moving_file.replace('.vtk', '.RigidAlignToAtlas.vtk')")
    parser.add_argument('--model', default='deep_learning', choices=['deep_learning', 'iterative_search'],
                        help="use deep learning network or conventional iterative search method to obtain the rotation angles. "+
                             "Deep learning (0.1s) is 100+ times faster than iterative search (30s). "+\
                             "If use deep_learning, ignore the following parameters for iterative_search. "+
                             "If deep_learning fails to rigidly align the surfaces, use iterative_search "+
                             "and finetune its parameters, you will finally get good results.")
    parser.add_argument('--numIntervals', '-numIntervals', 
                        default=8,
                        help="control the search interval. If results are not good, try to increase it. However, note that the complexity and time is n^3.")
    parser.add_argument('--SearchWidth', '-SearchWidth', 
                        default=64,
                        help="control the search width. If results are not good, try to increase it. Generally, finetuning seach width and serach interval will finally yield the optimal results.")
    parser.add_argument('--metric', '-metric', 
                        default='corr', choices=['mse', 'corr'],
                        help="sometimes, mse (mean square error) leads to suboptimal rigid alignment, try corr (pearson correlation coefficent) instead."+
                        " Found corr leads to better results at most time, so set corr as default.")
    parser.add_argument('--save_rot_mat', default=False, choices=['False', 'True'],
                        help="whether save the rotation matrix or not; if True,"+  \
                            " the default filename of the saved rotation matrix is" + \
                                " moving_file.replace(\'.vtk\', \'.RigidAlignToAtlas.rotmat.txt\')")
    
    
    args = parser.parse_args()
    moving_file = args.moving_file
    moving_file_pattern = args.moving_file_pattern
    atlas = args.atlas
    hemi = args.hemi
    out_name = args.out_name
    metric = args.metric
    model = args.model
    numIntervals = int(args.numIntervals)
    SearchWidth = int(args.SearchWidth)
    save_rot_mat = args.save_rot_mat
    
    if moving_file is None and moving_file_pattern is None:
        raise NotImplementedError('\nEither moving_file or moving_file_pattern is required.')
    if moving_file is not None:
        files = [moving_file]
        if out_name is None:
            out_name = [ file.replace('.vtk', '.RigidAlignToAtlas.vtk') for file in files ]
        else:
            out_name = [out_name]
    else:
        files = sorted(glob.glob(moving_file_pattern))
        if out_name is not None:
            print('\nGiven out_name is not taken because moving_file_pattern is given to process multiple files. The out_name will be automatically set.')
        out_name = [ file.replace('.vtk', '.RigidAlignToAtlas.vtk') for file in files ]
    
    if atlas == '/proj/ganglilab/users/Fenqiang/Atlas/72/72_?h.SphereSurf.vtk':
        if hemi == 'lh':
            atlas = '/proj/ganglilab/users/Fenqiang/Atlas/72/72_lh.SphereSurf.vtk'
        else:
            atlas = '/proj/ganglilab/users/Fenqiang/Atlas/72/72_rh.SphereSurf.vtk'
    
    print('\n-----------------------------------------------------------------')
    print('moving_file:', files)
    print('atlas:', atlas)
    print('hemi:', hemi)
    print('out_name:', out_name)
    print('model:', model)
    if model != 'deep_learning':
        print('numIntervals:', numIntervals)
        print('SearchWidth:', SearchWidth)
        print('metric:', metric)
    print('save_rot_mat:', save_rot_mat)   

    atlas = read_vtk(atlas)   
    if 'Convexity' in atlas:
        fixed_sulc = S3_normalize(atlas['Convexity'])[0: 163842]
    else:
        fixed_sulc = S3_normalize(atlas['sulc'])[0: 163842]
    
    
    if model == 'deep_learning':
        import torch
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        fixed_sulc = torch.from_numpy(fixed_sulc.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
        model = RigidAlignNet(in_ch=2, out_ch=3, level=8, n_res=6, rotated=0, complex_chs=16)
        print("\nLoading pretrained deep learning model's weights...")
        model.to(device)
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load('./pretrained_models/rigid_align_'+hemi+'_sulc_163842.mdl', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load('./pretrained_models/rigid_align_'+hemi+'_sulc_163842.mdl'))
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        model.eval()        
        
        for file in files:
            t1 = time.time()
            print()
            # print(files.index(file))
            print('Rigidly aligning', file)
            surf = read_vtk(file)
            if len(surf['vertices']) in [42,162,642,2562,10242,40962,163842] \
                and len(surf['faces']) == len(surf['vertices'])*2-4 \
                    and  surf['faces'][0][-1] == (len(surf['vertices'])+6)/4:  # faces seen in all my data empirically meet this requirement if it is an ico sphere
                        print('This is an ICOSAHEDRON_REPARAMETERIZED_SURFACE, no need to do resampling again!')
                        data = surf['sulc']            
            else:
                print('This is a NON_PARAMETERIZED_SURFACE, reparameterize to icosphere ...')
                data = resampleSphereSurf(surf['vertices'],  atlas['vertices'], surf['sulc'], faces=surf['faces'][:, 1:]).squeeze()
            assert data.shape == (163842, ), 'data.shape != (163842, )'
            data = S3_normalize(data)
            with torch.no_grad():
                moving = torch.from_numpy(data.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
                data_1 = torch.cat((moving, fixed_sulc), 1)
                rot_xyz = model(data_1).squeeze()
            print('Inference done, rotate by [{:4.3f} {:4.3f} {:4.3f}]'.format(rot_xyz[0].item(), rot_xyz[1].item(), rot_xyz[2].item()))
            
            rot_mat = get_rot_mat_zyx_torch(rot_xyz[0], rot_xyz[1], rot_xyz[2], device)
            rot_vertices = (rot_mat.cpu().numpy()).dot(np.transpose(surf['vertices']))
            rot_vertices = np.transpose(rot_vertices)
            surf['vertices'] = rot_vertices
            write_vtk(surf, out_name[files.index(file)])
            if save_rot_mat:
                print('save rotation matrix to', file.replace('.vtk', '.RigidAlignToAtlas.rotmat.txt'))
                np.savetxt(file.replace('.vtk', '.RigidAlignToAtlas.rotmat.txt'), rot_mat.cpu().numpy())
            t2 = time.time()
            print('save rigidly aligned surface to', out_name[files.index(file)])
            print("Rigid align done, took {:.2f} s".format(t2-t1))
            
    
    else:
        fixed_sulc = fixed_sulc[0:40962]
        bi_inter= get_bi_inter(len(fixed_sulc))[0]
        fixed_img = get_latlon_img(bi_inter, fixed_sulc)
        for file in files:
            t1 = time.time()
            print()
            print('Rigidly aligning', file)
            surf = read_vtk(file)
            sulc = S3_normalize(surf['sulc']) 
            print('Start searching for rigid alignment...')
            rot_angles, prev_energy, curr_energy = initialRigidAlign(sulc, fixed_sulc,
                                                   SearchWidth=SearchWidth/180*(np.pi), 
                                                   numIntervals=numIntervals, minSearchWidth=16/180*(np.pi),
                                                   moving_xyz=surf['vertices'], 
                                                   bi=True, fixed_img=fixed_img, 
                                                   metric=metric)
            print("\nFinal rotation after searching:", rot_angles)
            print("prev_energy: {:4.3f}, curr_energy: {:4.3f}".format(prev_energy, curr_energy))
        
            rot_mat = get_rot_mat_zyx(*rot_angles)
            rot_vertices = rot_mat.dot(np.transpose(surf['vertices']))
            rot_vertices = np.transpose(rot_vertices)
            surf['vertices'] = rot_vertices
            write_vtk(surf, out_name[files.index(file)])
            if save_rot_mat:
                print('save rotation matrix to', file.replace('.vtk', '.RigidAlignToAtlas.rotmat.txt'))
                np.savetxt(file.replace('.vtk', '.RigidAlignToAtlas.rotmat.txt'), rot_mat)
            t2 = time.time()
            print('save rigidly aligned surface to', out_name[files.index(file)])
            print("Rigid align done, took {:.2f} s".format(t2-t1))
            
            
            