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
    parser.add_argument("--input_images_dir_path", required=True)
    # parser.add_argument("--output_h5_dir_path", required=True)
    # parser.add_argument("--image_scale", type=float, default=12)
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

# # # recondir = '/media/kevin/MYDATA/textureless_labwall_10032018/dense/'
# # # outdir = "/media/kevin/MYDATA/textureless_labwall_10032018/demon_prediction"
# # # recondir = '/media/kevin/MYDATA/textureless_desk_10032018/dense/'
# # # outdir = "/media/kevin/MYDATA/textureless_desk_10032018/demon_prediction"
# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# # outdir = "/media/kevin/MYDATA/southbuilding_10032018/demon_prediction"
# outdir = "/media/kevin/MYDATA/southbuilding_28032018/demon_prediction_20_30_060"
# # recondir = '/media/kevin/MYDATA/cab_front/dense_384_512/'
# # outdir = "/media/kevin/MYDATA/cab_front/dense_384_512/demon_prediction"
# # outfile = os.path.join(outdir, "more_pairs_textureless_predictions_10032018.h5")
# outfile = os.path.join(outdir, "less_pairs_textureless_predictions_10032018.h5")

# recondir = '/home/kevin/ThesisDATA/30032018/CNB_wc/dense_192_256/'
# outdir = "/home/kevin/ThesisDATA/30032018/CNB_wc/demon_prediction_20_30_060"
# outfile = os.path.join(outdir, "less_pairs_textureless_predictions_10032018.h5")


# recondir = '/home/kevin/ThesisDATA/labwall/DenseSIFT/dense_192_256/'
# outdir = "/home/kevin/ThesisDATA/labwall/demon_prediction_20_30_060"
# # outdir = "/home/kevin/ThesisDATA/labwall/demon_prediction_30_60_040"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

# recondir = '/home/kevin/ThesisDATA/NewLabWall/dense_192_256/'
# # outdir = "/home/kevin/ThesisDATA/NewLabWall/demon_prediction_20_30_060"
# outdir = "/home/kevin/ThesisDATA/NewLabWall/demon_prediction_30_60_040"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# outdir = "/home/kevin/ThesisDATA/southbuilding_01042018/demon_prediction_80_20_080"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

# recondir = '/home/kevin/ThesisDATA/CVG_Capitole/dense_192_256/'
# outdir = "/home/kevin/ThesisDATA/CVG_Capitole/demon_prediction_15_50_050"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

# recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/DenseSIFT/dense_192_256/'
# outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_15_50_050"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")
#
# recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/DenseSIFT/dense_192_256/'
# outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/demon_prediction_15_50_050"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

# recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/terrace/DenseSIFT/dense_192_256/'
# outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/terrace/demon_prediction_15_50_050"
# outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/DenseSIFT/dense_192_256/'
outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/demon_prediction_15_50_050"
outfile = os.path.join(outdir, "kevin_southbuilding_demon.h5")

# recondir = '/media/kevin/MYDATA/Datasets_14032018/CNB_labwall/dense_384_512'
# # outdir = "/media/kevin/MYDATA/Datasets_14032018/CNB_labwall/demon_prediction"
# # outfile = os.path.join(outdir, "less_pairs_labwall_predictions_10032018.h5")
# outdir = "/media/kevin/MYDATA/Datasets_14032018/CNB_labwall/demon_prediction_more_pairs"
# outfile = os.path.join(outdir, "more_pairs_labwall_predictions_10032018.h5")

# recondir = '/media/kevin/MYDATA/Datasets_14032018/CalibBoard/dense_384_512'
# outdir = "/media/kevin/MYDATA/Datasets_14032018/CalibBoard/demon_prediction"
# outfile = os.path.join(outdir, "labwall_predictions_10032018.h5")

# recondir = '/media/kevin/MYDATA/cab_front/dense_384_512'
# outdir = "/media/kevin/MYDATA/cab_front/demon_prediction_betterpairs"
# outfile = os.path.join(outdir, "cab_front_predictions_10032018.h5")
# # outfile = os.path.join(outdir, "less_pairs_labwall_predictions_10032018.h5")
# #outdir = "/media/kevin/MYDATA/Datasets_14032018/CNB_labwall/demon_prediction_more_pairs"
# #outfile = os.path.join(outdir, "more_pairs_labwall_predictions_10032018.h5")

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

knn = 15 # 5
max_angle = 50*math.pi/180  # 60*math.pi/180
min_overlap_ratio = 0.5    # 0.5
# knn = 20 # 25 # 5
# max_angle = 45*math.pi/180  # 60*math.pi/180
# min_overlap_ratio = 0.75     # 0.5
# knn = 10 # 25 # 5
# max_angle = 30*math.pi/180  # 60*math.pi/180
# min_overlap_ratio = 0.8     # 0.5
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

def warp_flow(img, flow):
    flow = np.transpose(flow, [1,2,0])
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main():
    if True:
        views = []
        id_image_list = []
        cnt = 0
        for image_id, image in images.items():
            tmp_dict = {image_id: image}
            tmp_views = colmap.create_views(cameras, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
            # print("type(tmp_views) = ", type(tmp_views))
            # print("(tmp_views) = ", (tmp_views))
            new_v = adjust_intrinsics(tmp_views[0], target_K, w, h,)
            # print("type(new_v) = ", type(new_v))
            new_v_large = adjust_intrinsics(tmp_views[0], target_K_large, w_large, h_large,)
            new_v_large.image.save(os.path.join(outimagedir_large,image.name))
            new_v.image.save(os.path.join(outimagedir_small,image.name))
            if not new_v is None:
                id_image_list.append((image_id,image))
                views.append(new_v)
                # Kevin: visualization
                # visualize_views(tmp_views)
                cnt += 1
            # if cnt >= 2:
            #     break
        distances = compute_view_distances(views)

        pairs_to_compute = set()

        for idx, view in enumerate(views):
            print(idx, len(views), end=' ')
            view_dists = distances[idx]

            # find k nearest neighbours
            neighbours = sorted([(d,i) for i,d in enumerate(view_dists)])[1:knn+1]
            good_neighbours = []
            for _, neighbour_idx in neighbours:
                if not (idx, neighbour_idx) in pairs_to_compute:
                    neighbour_view = views[neighbour_idx]
                    if compute_view_angle(view, neighbour_view) < max_angle:
                        overlap = compute_view_overlap(view, neighbour_view)
                        if overlap > min_overlap_ratio:
                            good_neighbours.append(neighbour_idx)

            print(len(good_neighbours))
            for neighbour_idx in good_neighbours:
                pairs_to_compute.add((idx, neighbour_idx))
                pass

        len(pairs_to_compute)

    if tf.test.is_gpu_available(True):
        data_format='channels_first'
    else: # running on cpu requires channels_last data format
        data_format='channels_last'

    gpu_options = tf.GPUOptions()
    gpu_options.per_process_gpu_memory_fraction=0.8
    session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    # init networks
    bootstrap_net = BootstrapNet(session, data_format)
    iterative_net = IterativeNet(session, data_format)
    refine_net = RefinementNet(session, data_format)

    session.run(tf.global_variables_initializer())

    # load weights
    saver = tf.train.Saver()
    saver.restore(session,os.path.join(weights_dir,'demon_original'))

    args = parse_args()

    image_exts = [ '.jpg', '.JPG', '.jpeg', '.png' ]
    input_dir = args.input_images_dir_path
    # output_dir = args.output_h5_dir_path
    output_dir = outdir

    # Create output directory, if not present
    try:
        os.stat(output_dir)
    except:
        os.mkdir(output_dir)
    try:
        os.stat(os.path.join(output_dir, "vizdepthmap"))
    except:
        os.mkdir(os.path.join(output_dir, "vizdepthmap"))
    try:
        os.stat(os.path.join(output_dir, "optical_flow_48_64"))
    except:
        os.mkdir(os.path.join(output_dir, "optical_flow_48_64"))
    try:
        os.stat(os.path.join(output_dir, "graydepthmap"))
    except:
        os.mkdir(os.path.join(output_dir, "graydepthmap"))
    try:
        os.stat(os.path.join(output_dir, "flowconf_48_64"))
    except:
        os.mkdir(os.path.join(output_dir, "flowconf_48_64"))
    try:
        os.stat(os.path.join(output_dir, "flowconf_x_48_64"))
    except:
        os.mkdir(os.path.join(output_dir, "flowconf_x_48_64"))
    try:
        os.stat(os.path.join(output_dir, "flowconf_y_48_64"))
    except:
        os.mkdir(os.path.join(output_dir, "flowconf_y_48_64"))

    print("Write a NeXus HDF5 file")
    output_h5_filename = u"kevin_southbuilding_demon.h5"
    timestamp = u"20107-11-22T15:17:04-0500"
    output_h5_filepath = os.path.join(output_dir, output_h5_filename)
    h5file = h5py.File(output_h5_filepath, "w")
    # point to the default data to be plotted
    h5file.attrs[u'default']          = u'entry'
    # give the HDF5 root some more attributes
    h5file.attrs[u'file_name']        = output_h5_filename
    h5file.attrs[u'file_time']        = timestamp
    h5file.attrs[u'instrument']       = u'DeMoN'
    h5file.attrs[u'creator']          = u'DeMoN_prediction_to_h5.py'
    h5file.attrs[u'NeXus_version']    = u'4.3.0'
    h5file.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
    h5file.attrs[u'h5py_version']     = six.u(h5py.version.version)

    iteration = 0
    # # Iterate over working directory
    # for file1 in os.listdir(input_dir):
    #     for file2 in os.listdir(input_dir):
    # # for file1 in os.listdir(input_dir)[:3]:
    # #     for file2 in os.listdir(input_dir)[:3]:

    for i, pair in enumerate(pairs_to_compute):
        print(i, len(pairs_to_compute))
        # view1 = views[pair[0]]
        # view2 = views[pair[1]]
        # input_data = prepare_input_data(view1.image, view2.image, data_format)
        if True:
            # file1_path = os.path.join(input_dir, file1)
            # file1_name, file1_ext = os.path.splitext(file1_path)
            # file2_path = os.path.join(input_dir, file2)
            # file2_name, file2_ext = os.path.splitext(file2_path)
            file1_name = id_image_list[pair[0]][1].name
            file1 = file1_name
            file1_ext = ".JPG"
            file2_ext = ".JPG"
            file2_name = id_image_list[pair[1]][1].name
            file2 = file2_name
            file1_path = os.path.join(input_dir, file1_name)
            file2_path = os.path.join(input_dir, file2_name)
            # # Check if file is an image file
            # if file1_ext not in image_exts:
            #     print("Skipping " + file1 + " (not an image file)")
            #     continue
            # if file2_ext not in image_exts:
            #     print("Skipping " + file2 + " (not an image file)")
            #     continue
            # if file1_name == file2_name:
            #     print("Skipping identical files: " + file1 + "---" + file2 + "...")
            #     continue

            print("Processing image pair = " + file1 + "---" + file2 + "...")

            img1 = Image.open(file1_path)
            img2 = Image.open(file2_path)

            input_data = prepare_input_data(img1,img2,data_format)

            # run the network
            result = bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])
            for i in range(3):
                result = iterative_net.eval(
                    input_data['image_pair'],
                    input_data['image2_2'],
                    result['predict_depth2'],
                    result['predict_normal2'],
                    result['predict_rotation'],
                    result['predict_translation']
                )
            # rotation = result['predict_rotation']
            rotation = result['predict_rotation'].squeeze()
            rotation_matrix = angleaxis_to_rotation_matrix(rotation)
            # print('rotation = ', rotation)
            # print(len(rotation.shape))
            # print(len(rotation[:].shape))
            # print('rotation matrix = ', angleaxis_to_rotation_matrix(rotation))
            # translation = result['predict_translation']
            translation = result['predict_translation'].squeeze()
            # print('translation = ', translation)
            # print(type(result))
            # print(result.keys())
            # depth_48by64 = result['predict_depth2']
            # print(depth_48by64.shape)
            depth_48by64 = result['predict_depth2'].squeeze()
            # print(depth_48by64.shape)
            # print(result.keys())
            flow2 = result['predict_flow2'].squeeze()
            # if tf.test.is_gpu_available(True) and data_format == 'channels_first':
            flow2 = flow2.transpose([2, 0, 1])
            print(flow2.shape)
            ### also save the confidence of optical flow 2
            flowconf2 = result['predict_flowconf2'].squeeze()
            # if tf.test.is_gpu_available(True) and data_format == 'channels_first':
            flowconf2 = flowconf2.transpose([2, 0, 1])
            print(flowconf2.shape)
            # flow5 = result['predict_flow5'].squeeze()
            # print(flow5.shape)
            scale = result['predict_scale'].squeeze().astype(np.float32)

            result = refine_net.eval(input_data['image1'],result['predict_depth2'])
            # depth_upsampled = result['predict_depth0']
            # print(depth_upsampled.shape)
            depth_upsampled = result['predict_depth0'].squeeze()
            # print(type(depth_upsampled))
            # print(depth_upsampled.shape)
            # print(type(result))
            # # for k, v in result.items():
            # #     print(k, v)
            # print(result.keys())
            # plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
            # plt.show()

            # # a colormap and a normalization instance
            cmap = plt.cm.jet
            plt.imsave(os.path.join(output_dir, "graydepthmap", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), result['predict_depth0'].squeeze(), cmap='Greys')
            plt.imsave(os.path.join(output_dir, "vizdepthmap", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), result['predict_depth0'].squeeze(), cmap=cmap)

            # ofplot = warp_flow(input_data['image2_2'], flow2)
            # cv2.imshow('ofplot', ofplot)

            # # try to visualize the point cloud
            # try:
            #     # from depthmotionnet.vis import *
            #     visualize_prediction(
            #         inverse_depth=result['predict_depth0'],
            #         image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
            #         rotation=rotation,
            #         translation=translation)
            # except ImportError as err:
            #     print("Cannot visualize as pointcloud.", err)

            # h5file.create_dataset(("/" + file1 + "---" + file2 + "/rotation_angleaxis"), data=rotation)
            # h5file.create_dataset(("/" + file1 + "---" + file2 + "/rotation_matrix"), data=rotation_matrix)
            # h5file.create_dataset(("/" + file1 + "---" + file2 + "/translation"), data=translation)
            # h5file.create_dataset(("/" + file1 + "---" + file2 + "/depth"), data=depth_48by64)
            # h5file.create_dataset(("/" + file1 + "---" + file2 + "/depth_upsampled"), data=depth_upsampled)
            # h5file.create_dataset(("/" + file1 + "---" + file2 + "/flow"), data=flow2)

            h5file.create_dataset((file1 + "---" + file2 + "/rotation_angleaxis"), data=rotation)
            #h5file.create_dataset((file1 + "---" + file2 + "/rotation_matrix"), data=rotation_matrix)
            h5file.create_dataset((file1 + "---" + file2 + "/rotation"), data=rotation_matrix)
            h5file.create_dataset((file1 + "---" + file2 + "/translation"), data=translation)
            h5file.create_dataset((file1 + "---" + file2 + "/depth"), data=depth_48by64)
            h5file.create_dataset((file1 + "---" + file2 + "/depth_upsampled"), data=depth_upsampled)
            h5file.create_dataset((file1 + "---" + file2 + "/flow"), data=flow2)
            h5file.create_dataset((file1 + "---" + file2 + "/flowconf"), data=flowconf2)
            h5file.create_dataset((file1 + "---" + file2 + "/scale"), data=scale)

            # ofplot = warp_flow(input_data['image2_2'], flow2)
            # ofplot = convert_flow_to_save_img(flow2)
            ofplot = convert_flow_to_plot_img(flow2)
            plt.imsave(os.path.join(output_dir, "optical_flow_48_64", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), ofplot, cmap=cmap)
            # ofplot = convert_flow_to_plot_img(flow2)
            # cv2.imshow('ofplot', ofplot)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # return
            plt.imsave(os.path.join(output_dir, "flowconf_x_48_64", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), flowconf2[0,:,:], cmap='Greys')
            plt.imsave(os.path.join(output_dir, "flowconf_y_48_64", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), flowconf2[1,:,:], cmap='Greys')
            # ofplot = convert_flow_to_plot_img(flow2)
            combinedFlowConf2 = np.sqrt(np.square(flowconf2[0,:,:])+np.square(flowconf2[1,:,:]))
            tmpMin = np.min(combinedFlowConf2)
            tmpMax = np.max(combinedFlowConf2)
            combinedFlowConf2 = ((combinedFlowConf2-tmpMin)/(tmpMax-tmpMin))*255
            plt.imsave(os.path.join(output_dir, "flowconf_48_64", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), combinedFlowConf2, cmap='Greys')


    h5file.close()   # be CERTAIN to close the file
    print("HDF5 file is written successfully:", output_h5_filepath)

if __name__ == "__main__":
    main()
