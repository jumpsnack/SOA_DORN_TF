# DORN for Tensorflow

**Introduction**  
This is a Tensorflow implementation of [Deep Ordinal Regression Network for Monocular Depth Estimation](https://arxiv.org/abs/1806.02446).

*See `utils/confutil.py` for details.*

**KITTI**  
**Dataset path*
>input:  `~/_DepthPredic/DATASET/KITTI_eigen_dense/[%A]/[%B]/[%C]/[%D]/data/[filename].jpg`  
>gt: `~/_DepthPredic/DATASET/KITTI_eigen_dense/[%A]/[%B]/[%C]/[%D]/groundtruth/[filename].png`  
>%A = { train-set, test-set, valid-set }  
>%B = { 2011_09_26, ... }  
>%C = { 2011_09_26_drive_0002_sync, ... }  
>%D = { image_02, image_03 }  

**CSV path*
>`~/_DepthPredic/DATASET/KITTI_eigen_dense/[%A].csv`  
>%A = { train, test, valid }  

**CSV Format*  
[1] `../../DATASET/KITTI_eigen_dense/train-set/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000049.jpg, ../../DATASET/KITTI_eigen_dense/train-set/2011_09_26/2011_09_26_drive_0001_sync/image_02/groundtruth/0000000049.png`  
[2] `../../DATASET/KITTI_eigen_dense/train-set/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000057.jpg, ../../DATASET/KITTI_eigen_dense/train-set/2011_09_26/2011_09_26_drive_0001_sync/image_02/groundtruth/0000000057.png`  
[3]...  


**NYU**  
**Dataset path*
>input:  `~/_DepthPredic/DATASET/NYU/[%A]/[%B]/[filename].jpg`  
>gt: `~/_DepthPredic/DATASET/NYU/[%A]/[%B]/[filename].png`  
>%A = { train-set, test-set, valid-set }  
>%B = { basement_0001a_out, ... }  

**CSV path*
>`~/_DepthPredic/DATASET/NYU/[%A].csv`  
>%A = { train, test, valid }