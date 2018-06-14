import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import argparse
import subprocess
import collections
import sqlite3
import h5py
import six
import cv2

print(sys.path)
from depthmotionnet.vis import *
from depthmotionnet.networks_original import *
from scipy import interpolate
import cv2
###### think about how to add lmbspecialops and depthmotionnet into anaconda env path!
# sys.path.append(r'/home/kevin/anaconda_tensorflow_demon_ws/demon/lmbspecialops/python/')
# sys.path.append(r'/home/kevin/anaconda_tensorflow_demon_ws/demon/python/depthmotionnet/')

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

def warp_flow(img, flow):
    flow = np.transpose(flow, [1,2,0])
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def convert_flow_to_plot_img(flow):
    flow = np.transpose(flow, [1,2,0])
    # h, w = flow.shape[:2]
    # # flow = flow*256
    # # flow = flow*128+128
    # # flow = flow*128+128
    # flow[:,:,0] = flow[:,:,0]*w
    # flow[:,:,1] = flow[:,:,1]*h
    # zeros = np.zeros([h,w])
    # # res = np.concatenate((flow,zeros), axis=2)
    # res = np.dstack((flow,zeros))
    # # res = flow
    # # res[:,:,2] = 0
    # print(res.shape)
    # return res
    h, w = flow.shape[:2]
    # fx, fy = flow[:,:,0]*w, flow[:,:,1]*h
    fx, fy = flow[:,:,0]*255, flow[:,:,1]*255
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def convert_flow_to_save_img(flow):
    flow = np.transpose(flow, [1,2,0])
    h, w = flow.shape[:2]
    # flow = flow*256
    flow[:,:,0] = flow[:,:,0]*w
    flow[:,:,1] = flow[:,:,1]*h
    print(np.max(flow), np.min(flow))
    # flow = flow*128+128
    zeros = np.zeros([h,w])
    # res = np.concatenate((flow,zeros), axis=2)
    res = np.dstack((flow,zeros))
    # res = flow
    # res[:,:,2] = 0
    print(res.shape)
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_demon_file_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--of_upsampling_scale", type=int, default=2)
    args = parser.parse_args()

    # input_dir = os.path.join(os.getcwd(), 'img')
    # output_dir = os.path.join(os.getcwd(), 'out')
    return args


def angleaxis_to_rotation_matrix(aa):
    """Converts the 3 element angle axis representation to a 3x3 rotation matrix

    aa: numpy.ndarray with 1 dimension and 3 elements
    Returns a 3x3 numpy.ndarray
    """
    if len(aa.shape) == 2:
        # aa = np.reshape(aa, (aa.shape[1]))
        aa = np.squeeze(aa)
    angle = np.sqrt(aa.dot(aa))

    if angle > 1e-6:
        c = np.cos(angle);
        s = np.sin(angle);
        u = np.array([aa[0]/angle, aa[1]/angle, aa[2]/angle]);

        R = np.empty((3,3), dtype=np.float32)
        R[0,0] = c+u[0]*u[0]*(1-c);      R[0,1] = u[0]*u[1]*(1-c)-u[2]*s; R[0,2] = u[0]*u[2]*(1-c)+u[1]*s;
        R[1,0] = u[1]*u[0]*(1-c)+u[2]*s; R[1,1] = c+u[1]*u[1]*(1-c);      R[1,2] = u[1]*u[2]*(1-c)-u[0]*s;
        R[2,0] = u[2]*u[0]*(1-c)-u[1]*s; R[2,1] = u[2]*u[1]*(1-c)+u[0]*s; R[2,2] = c+u[2]*u[2]*(1-c);
    else:
        R = np.eye(3)
    return R

def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256,192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256,192))
    img2_2 = img2.resize((64,48))

    # transform range from [0,255] to [-0.5,0.5]
    img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    img2_arr = np.array(img2).astype(np.float32)/255 -0.5
    img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5

    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2,0,1])
        img2_arr = img2_arr.transpose([2,0,1])
        img2_2_arr = img2_2_arr.transpose([2,0,1])
        image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)

    result = {
        'image_pair': image_pair[np.newaxis,:],
        'image1': img1_arr[np.newaxis,:], # first image
        'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }
    return result

import tensorflow as tf
from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.helpers import angleaxis_to_rotation_matrix
import colmap_utils as colmap
from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
import math
import h5py
import os
import cv2

def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256,192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256,192))
    img2_2 = img2.resize((64,48))

    # transform range from [0,255] to [-0.5,0.5]
    img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    img2_arr = np.array(img2).astype(np.float32)/255 -0.5
    img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5

    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2,0,1])
        img2_arr = img2_arr.transpose([2,0,1])
        img2_2_arr = img2_2_arr.transpose([2,0,1])
        image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)

    result = {
        'image_pair': image_pair[np.newaxis,:],
        'image1': img1_arr[np.newaxis,:], # first image
        'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }
    return result


def compute_visible_points( view1, view2 ):
    """Computes how many 3d points of view1 are visible in view2

    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    Returns the number of visible 3d points in view2
    """
    return np.count_nonzero( compute_visible_points_mask( view1, view2 ) )


def compute_valid_points( view ):
    """Count the valid depth values for this view

    view: View namedtuple

    Returns the number of valid depth values
    """
    return np.count_nonzero(view.depth[np.isfinite(view.depth)] > 0)


def compute_view_overlap( view1, view2 ):
    """Computes the overlap between the two views

    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    Returns the overlap ratio
    """
    valid_points1 = compute_valid_points(view1)
    visible_points12 = compute_visible_points(view1, view2)

    ratio12 = visible_points12/valid_points1
    return ratio12


# weights_dir = '/home/ummenhof/projects/demon/weights'
weights_dir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/weights'
# outdir = '/home/kevin/DeMoN_Prediction/south_building'
# outfile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions.h5'
## outfile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions_v1_05012018.h5'

# outdir = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction'
# outdir = '/home/kevin/ThesisDATA/person-hall/demon_prediction'
# outdir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction"
# outdir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction"
# outfile = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction/gerrard_hall_predictions.h5'
# outfile = '/home/kevin/ThesisDATA/person-hall/demon_prediction/person_hall_predictions.h5'
# outfile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction/CVG_barcelona_predictions.h5"
# outfile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction/CVG_redmond_predictions.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/southbuilding/demon_prediction"
# outfile = os.path.join(outdir, "more_pairs_southbuilding_predictions_05022018.h5")
# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'

# recondir = '/home/kevin/ThesisDATA/gerrard-hall/dense/'
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/gerrard_hall/demon_prediction"
# outfile = os.path.join(outdir, "more_pairs_gerrard_hall_predictions_22022018.h5")

# recondir = '/media/kevin/MYDATA/textureless_labwall_10032018/dense/'
# outdir = "/media/kevin/MYDATA/textureless_labwall_10032018/demon_prediction"
# recondir = '/media/kevin/MYDATA/textureless_desk_10032018/dense/'
# outdir = "/media/kevin/MYDATA/textureless_desk_10032018/demon_prediction"
# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# outdir = "/media/kevin/MYDATA/southbuilding_10032018/demon_prediction"
recondir = '/media/kevin/MYDATA/cab_front/dense_384_512/'
outdir = "/media/kevin/MYDATA/cab_front/dense_384_512/demon_prediction"
outfile = os.path.join(outdir, "more_pairs_textureless_predictions_10032018.h5")

outimagedir_small = os.path.join(outdir,'images_demon_demonsize')
outimagedir_large = os.path.join(outdir,'images_demon_undistorted')
os.makedirs(outdir, exist_ok=True)
os.makedirs(outimagedir_small, exist_ok=True)
os.makedirs(outimagedir_large, exist_ok=True)
# os.makedirs(os.path.join(outdir,'graydepthmap'), exist_ok=True)
# os.makedirs(os.path.join(outdir,'vizdepthmap'), exist_ok=True)




# recondir = '/misc/lmbraid12/depthmotionnet/datasets/mvs_colmap/south-building/mvs/'
# recondir = '/home/kevin/ThesisDATA/ToyDataset_Desk/dense/'
# recondir = '/home/kevin/ThesisDATA/gerrard-hall/dense/'
# recondir = '/home/kevin/ThesisDATA/person-hall/dense/'
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/dense/"
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/dense/"


cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))

images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))

views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))

knn = 15 # 25 # 5
max_angle = 60*math.pi/180  # 60*math.pi/180
min_overlap_ratio = 0.5     # 0.5
w = 256
h = 192
normalized_intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5],np.float32)
target_K = np.eye(3)
target_K[0,0] = w*normalized_intrinsics[0]
target_K[1,1] = h*normalized_intrinsics[1]
target_K[0,2] = w*normalized_intrinsics[2]
target_K[1,2] = h*normalized_intrinsics[3]

# w_large = 8*w
# h_large = 8*h
# w_large = 12.1*w
# h_large = 12.05*h
# w_large = 12*w
# h_large = 12*h

w_large = 4*w
h_large = 4*h

# w_large = 7.8125*w
# h_large = 7.8125*h
# #

# # # gerrard-hall, person-hall
# w_large = 7.8125*w
# h_large = 6.84896*h

# # # barcelona cvg
# w_large = 7.8125*w
# h_large = 7.8125*h  # barcelona cvg

# # # redmon cvg
# w_large = 7.8125*w
# h_large = 5.7917*h  # cvg redmond

target_K_large = np.eye(3)
target_K_large[0,0] = w_large*normalized_intrinsics[0]
target_K_large[1,1] = h_large*normalized_intrinsics[1]
target_K_large[0,2] = w_large*normalized_intrinsics[2]
target_K_large[1,2] = h_large*normalized_intrinsics[3]



#
# DeMoN has been trained for specific internal camera parameters.
#
# If you use your own images try to adapt the intrinsics by cropping
# to match the following normalized intrinsics:
#
#  K = (0.89115971  0           0.5)
#      (0           1.18821287  0.5)
#      (0           0           1  ),
#  where K(1,1), K(2,2) are the focal lengths for x and y direction.
#  and (K(1,3), K(2,3)) is the principal point.
#  The parameters are normalized such that the image height and width is 1.
#
def upsample_optical_flow(flow12, OF_scale_factor=1):
    x = np.array(range(flow12.shape[2]))
    y = np.array(range(flow12.shape[1]))
    xx, yy = np.meshgrid(x, y)
    # print(xx.shape)
    a = np.array(flow12[0,:,:])
    f = interpolate.interp2d(x, y, a, kind='linear')
    xnew = np.array(range(flow12.shape[2]*OF_scale_factor))/OF_scale_factor
    ynew = np.array(range(flow12.shape[1]*OF_scale_factor))/OF_scale_factor
    # print(xnew)
    # print(xnew.shape)
    znew = f(xnew, ynew)
    # print(znew.shape)
    flow12upsampled = np.zeros((2,flow12.shape[1]*OF_scale_factor, flow12.shape[2]*OF_scale_factor))
    flow12upsampled[0,:,:] = f(xnew, ynew)
    a = np.array(flow12[1,:,:])
    f = interpolate.interp2d(x, y, a, kind='linear')
    flow12upsampled[1,:,:] = f(xnew, ynew)
    # print(flow12upsampled.shape)
    return flow12upsampled


def warp_flow(img, flow):
    flow = np.transpose(flow, [1,2,0])
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main():
    args = parse_args()

    image_exts = [ '.jpg', '.JPG', '.jpeg', '.png' ]
    input_h5_file = args.input_demon_file_path
    output_dir = args.output_dir_path
    # output_dir = os.pa


    output_optical_flow_dir = os.path.join(output_dir, "optical_flow_scale_{0}".format(args.of_upsampling_scale))
    try:
        os.stat(output_optical_flow_dir)
    except:
        os.mkdir(output_optical_flow_dir)


    h5file = h5py.File(input_h5_file, "r")

    for of_name in h5file.keys():
        tmp = of_name.split('/')
        # tmp = tmp[0].split('---')
        # tmp0 = tmp[0].split('.')
        # tmp1 = tmp[1].split('.')
        # output_of_name = tmp0[0]+'---'+tmp0[0]+'.JPG'
        output_of_name = tmp[0]+'.JPG'
        flow = h5file[of_name]['flow'].value

        ### add code to upsample the predicted optical-flow
        if args.of_upsampling_scale > 1:
            flow_upsampled = upsample_optical_flow(flow, OF_scale_factor=args.of_upsampling_scale)
            # flow = flow_upsampled

        # # a colormap and a normalization instance
        cmap = plt.cm.jet
        # ofplot = convert_flow_to_save_img(flow2)
        ofplot = convert_flow_to_plot_img(flow_upsampled)
        # plt.imsave(os.path.join(output_optical_flow_dir, output_of_name), ofplot, cmap=cmap)
        cv2.imwrite(os.path.join(output_optical_flow_dir, output_of_name), ofplot)
        # ofplot = convert_flow_to_plot_img(flow2)
        # cv2.imshow('ofplot', ofplot)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # return

    h5file.close()   # be CERTAIN to close the file
    print("optical flow in HDF5 file is upsampled successfully:", output_optical_flow_dir)

if __name__ == "__main__":
    main()
