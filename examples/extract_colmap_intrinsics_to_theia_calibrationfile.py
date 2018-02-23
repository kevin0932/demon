import colmap_utils as colmap
from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
import math
import h5py
import os
import cv2

# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# recondir = '/home/kevin/ThesisDATA/ToyDataset_Desk/dense/'
# recondir = '/home/kevin/ThesisDATA/gerrard-hall/dense/'
# recondir = '/home/kevin/ThesisDATA/person-hall/dense/'
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/dense/"
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/dense/"
# recondir = '/media/kevin/SamsungT5_F/ThesisDATA/southbuilding/demon_prediction/images_demon/dense'
recondir = '/media/kevin/SamsungT5_F/ThesisDATA/gerrard_hall/demon_prediction/images_demon/dense'

path = os.path.join(recondir,'sparse','theia_calibration_file.txt')

cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))

images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))

calibrFile = open(path,'w')

# for imgId,val in cameras.items():
for imgId,val in images.items():
    # print("images[imgId] = ", images[imgId])
    camId = val.cam_id
    cur_name = val.name
    # calibrFile.write("%s %f %f %f %f %f %f %f\n" % (cur_name, cameras[camId].params[0], cameras[camId].params[2], cameras[camId].params[3], 1, 0, 0, 0))
    calibrFile.write("%s %f %f %f %f %f %f %f\n" % (cur_name, 2737.64, 1536, 1152, 1, 0, 0, 0))

calibrFile.close()
