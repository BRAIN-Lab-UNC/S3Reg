# [S3Reg: Superfast Spherical Surface Registration](https://ieeexplore.ieee.org/document/9389746)

## How to use it

### Install

1. Clone or download this repository.

2. This project depends on a lot of utilities in [Spherical U-Net package](https://github.com/zhaofenqiang/SphericalUNetPackage). So you need to install [Spherical U-Net package](https://pypi.org/project/spherical/) from PyPI:
```
pip install sphericalunet
```

### Example code
It currently is not that straightforward to use. You need to modify the `inference.py` to match your own spherical surface and atlas file. Then simply run:
```
python inference.py
```
The registration parameters in `regConfig_3level.txt` can be modified to balance the smoothness and similarity of the registration results:
```
levels 5,6,7   # levels of the feature resolution to run the registration on
features sulc,sulc,curv  # feature to be registered on each level
learning_rate 0.001
weight_smooth 0.8,3.2,3.9  # smooth weight of the deformation field. Note this parameter only has effect during training and has no influence for inference.
weight_l2 1.0,1.0,1.0  
weight_phi_consis 1.0,1.5,3.0
weight_corr 0.5,0.5,0.2
diffe True                   # differemorphic registration or not
bi True                      # use bilinear interpolation to speed up the resampling and thus the registration process
weight_level 1.0,1.0,1.0
num_composition 6            # number of the "scaling and squaring" layers
centra False
weight_centra 0,0,0
truncated True,True,True     # if the deformation are too large to induce topological errors, i.e., the folded triangles, truncated it or not
```
