import os
import argparse
import subprocess
import collections
import sqlite3
import h5py
import numpy as np
import math
import functools

from pyquaternion import Quaternion
import nibabel.quaternions as nq

import PIL.Image
from matplotlib import pyplot as plt
#import os
import sys
import colmap_utils as colmap
from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.helpers import angleaxis_to_rotation_matrix
import re
import six
# import scipy
from scipy.spatial import distance
from scipy import interpolate
import cv2
import shutil

from collections import namedtuple
from minieigen import Quaternion, Matrix3
# examples_dir = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(examples_dir, '..', 'lmbspecialops', 'python'))

ColmapImage = namedtuple('ColmapImage',['cam_id','name','R','t'])

def quaternion_to_rotation_matrix(q):
    """Converts quaternion to rotation matrix

    q: tuple with 4 elements

    Returns a 3x3 numpy array
    """
    q = Quaternion(*q)
    R = q.toRotationMatrix()
    return np.array([list(R.row(0)), list(R.row(1)), list(R.row(2))],dtype=np.float32)

def read_images_txt(filename):
    """Simple reader for the images.txt file

    filename: str
        path to the images.txt

    Returns a dictionary will all cameras
    """
    result = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()

        line1 = line
        line2 = f.readline()

        while line1:
            items = line1.split(' ')
            q = tuple([float(x) for x in items[1:5]])
            t = tuple([float(x) for x in items[5:8]])
            image = ColmapImage(
                cam_id = int(items[8]),
                name = items[9].strip(),
                # q = tuple([float(x) for x in items[1:5]]),
                # t = tuple([float(x) for x in items[5:8]])
                R = quaternion_to_rotation_matrix(q).astype(np.float64),
                t = np.array(t, dtype=np.float32).astype(np.float64)
            )

            # result[int(items[0])] = image
            result[items[9].strip()] = image

            line1 = f.readline()
            line2 = f.readline()

    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--colmap_images_txt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--demon_path", required=True)
    parser.add_argument("--input_good_pairs_path", required=True)
    parser.add_argument("--scale_factor", type=int, default=24)
    parser.add_argument("--min_num_features", type=int, default=1)
    parser.add_argument("--ratio_threshold", type=float, default=0.75)
    parser.add_argument("--max_descriptor_distance", type=float, default=1.00)
    parser.add_argument("--max_pixel_error", type=float, default=1.00)
    parser.add_argument("--OF_scale_factor", type=int, default=1)
    parser.add_argument("--survivor_ratio", type=float, default=0.50)
    parser.add_argument("--rotation_consistency_error_deg", type=float, default=20.0)
    parser.add_argument("--translation_consistency_error_deg", type=float, default=20.0)
    args = parser.parse_args()
    return args

def flow_to_matches_float32Pixels(flow):
    fx = (flow[0] * flow.shape[2]).astype(np.float32)
    fy = (flow[1] * flow.shape[1]).astype(np.float32)
    fx_int = np.round(flow[0] * flow.shape[2]).astype(np.int)
    fy_int = np.round(flow[1] * flow.shape[1]).astype(np.int)
    y1, x1 = np.mgrid[0:flow.shape[1], 0:flow.shape[2]]
    x2 = x1.ravel() + fx.ravel()
    y2 = y1.ravel() + fy.ravel()
    x2_int = x1.ravel() + fx_int.ravel()
    y2_int = y1.ravel() + fy_int.ravel()
    # mask = (x2 >= 0) & (x2 < flow.shape[2]) & \
    #        (y2 >= 0) & (y2 < flow.shape[1]) & depthMask1D
    mask = (x2_int >= 0) & (x2_int < flow.shape[2]) & \
           (y2_int >= 0) & (y2_int < flow.shape[1])
    matches = np.zeros((mask.size, 2), dtype=np.uint32)
    matches[:, 0] = np.arange(mask.size)
    matches[:, 1] = y2_int * flow.shape[2] + x2_int
    matches = matches[mask].copy()
    # print(np.max(matches[:, 0]), " ", np.max(matches[:, 1]))
    # print(mask.size, " ", depthMask1D.size)
    coords1 = np.column_stack((x1.ravel(), y1.ravel()))[mask]
    coords2 = np.column_stack((x2, y2))[mask]
    return matches, coords1, coords2

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

def get_tf_data_format():
    if tf.test.is_gpu_available(True):
        data_format='channels_first'
    else: # running on cpu requires channels_last data format
        data_format='channels_last'

    return data_format

def cross_check_matches_float32Pixel(matches12, coords121, coords122,
                        matches21, coords211, coords212,
                        max_reproj_error):
    if matches12.size == 0 or matches21.size == 0:
        return np.zeros((0, 2), dtype=np.uint32)

    matches121 = collections.defaultdict(list)
    # coord_12_1_by_idx2 = collections.defaultdict(list)
    # for match, coord in zip(matches12, coords121):
    #     matches121[match[1]].append((match[0], coord))
    for match, coord, coord2 in zip(matches12, coords121, coords122):
        matches121[match[1]].append((match[0], coord, coord2))
        # coord_12_1_by_idx2[match[1]].append(coord)

    max_reproj_error = max_reproj_error**2

    matches = []
    float32_coords_1 = []
    float32_coords_2 = []
    all_matches_pixel_consistencies = []
    for match, coord, coord1 in zip(matches21, coords212, coords211):
        if match[0] not in matches121:
            continue
        match121 = matches121[match[0]]
        coord_12_2 = match121[0][2]
        coord_12_1 = match121[0][1]
        coord_21_2 = coord1
        coord_21_1 = coord
        if len(match121) > 1:
            continue
        # if match121[0][0] == match[1]:
        #     matches.append((match[1], match[0]))
        diff = match121[0][1] - coord
        squared_error = diff[0] * diff[0] + diff[1] * diff[1]
        all_matches_pixel_consistencies.append((match[1], match[0], np.sqrt(squared_error)))
        if squared_error <= max_reproj_error:
            matches.append((match[1], match[0]))
            float32_coords_1.append( coord_12_1 )
            float32_coords_2.append( coord_12_2 )
            # matches.append((match[1], match[0]))
            # float32_coords_1.append( coord_21_1 )
            # float32_coords_2.append( coord_21_2 )

    return np.array(matches, dtype=np.uint32), np.array(float32_coords_1, dtype=np.float32), np.array(float32_coords_2, dtype=np.float32), np.array(all_matches_pixel_consistencies, dtype=np.float32)

def upsample_optical_flow(flow12, OF_scale_factor=1):
    x = np.array(range(flow12.shape[2]))
    y = np.array(range(flow12.shape[1]))
    xx, yy = np.meshgrid(x, y)
    # print(xx.shape)
    a = np.array(flow12[0,:,:])
    f = interpolate.interp2d(x, y, a, kind='linear')
    # xnew = np.array(range(flow12.shape[2]*OF_scale_factor))
    # ynew = np.array(range(flow12.shape[1]*OF_scale_factor))
    xnew = np.array(range(flow12.shape[2]*OF_scale_factor))/OF_scale_factor
    ynew = np.array(range(flow12.shape[1]*OF_scale_factor))/OF_scale_factor
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
    flow = np.transpose(flow, [1, 2, 0])
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def TheiaClamp( f, a, b):
    return max(a, min(f, b))

def main():
    # data_format = get_tf_data_format()
    w = 64
    h = 48
    normalized_intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5],np.float32)
    target_K = np.eye(3)
    target_K[0,0] = w*normalized_intrinsics[0]
    target_K[1,1] = h*normalized_intrinsics[1]
    target_K[0,2] = w*normalized_intrinsics[2]
    target_K[1,2] = h*normalized_intrinsics[3]

    args = parse_args()

    ColmapImages = read_images_txt(args.colmap_images_txt)
    print("(ColmapImages) = ", (ColmapImages))
    print("len(ColmapImages) = ", len(ColmapImages))
    # return

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    try:
        os.stat(args.output_path)
    except:
        os.mkdir(args.output_path)

    images_id_to_name = {}
    images_name_to_id = {}
    features_list = {}
    descriptors_list = {}
    quantization_list = {}

    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        image_name = row[2]
        images_id_to_name[image_id] = image_name
        images_name_to_id[image_name] = image_id

    # with open(os.path.join(args.output_path, 'test_keypoints.txt'), "w") as fid:
    #     cursor.execute("SELECT image_id, data FROM keypoints WHERE rows>=?;",
    #                    (args.min_num_features,))
    #     for row in cursor:
    #         image_id = row[0]
    #         features = np.fromstring(row[1], dtype=np.float32).reshape(-1, 6)
    #         featuresMat = features[:,0:2]
    #         image_name = images_id_to_name[image_id]
    #         features_list[image_name] = featuresMat
    #         # # featuresMat = np.concatenate((featuresMat, np.array([], dtype=np.float32))), axis=0)
    #         # fid.write("%s %d\n" % (image_name, features.shape[0]))
    #         # for i in range(features.shape[0]):
    #         #     print(features[i])
    #         #     fid.write("%d %d %d %d %d %d\n" % tuple(features[i]))
    #
    #
    # cursor.close()
    # connection.close()

    # print("features_list.keys() = ", features_list.keys())
    # print("len(features_list.keys()) = ", len(features_list.keys()))

    # good_pairs = []
    # with open((args.input_good_pairs_path), "r") as fid:
    #     while True:
    #         line = fid.readline()
    #         if not line:
    #             break
    #         line = line.strip()
    #         if len(line) > 0 and line[0] != "#":
    #             elems = line.split()
    #             good_pair = (elems[0])
    #             good_pairs.append(good_pair)
    # print("good_pairs num = ", good_pairs)

    data = h5py.File(args.demon_path)
    OF_scale_factor = args.OF_scale_factor

    upsampled_width = OF_scale_factor*w
    upsampled_height = OF_scale_factor*h

    image_pairs = set()
    valid_pair_num = 0
    survivor_ratio_list = []
    survivor_ratio_list_bidir = []
    translation_pose_campos_consistency_list = []
    translation_pose_camvec_consistency_list = []
    translation_pose_camvec_accuracy_list = []
    rotation_pose_consistency_list = []
    rotation_pose_accuracy_list = []
    avg_flow_consistency_error_121_list = []
    avg_flow_consistency_error_212_list = []
    pixel_consistency_conf_mat = []
    twoview_baseline_length_list = []
    twoview_view_angle_list = []


    for image_pair12 in data.keys():
        image_name1, image_name2 = image_pair12.split('---')
        if image_pair12 not in data.keys():
            continue
        print("Processing ", image_pair12)

        if image_pair12 in image_pairs:
            continue

        image_pair21 = "{}---{}".format(image_name2, image_name1)
        image_pairs.add(image_pair12)
        # image_pairs.add(image_pair21)
        if image_pair21 not in data.keys():
            continue

        flow12 = data[image_pair12]["flow"]
        # flow12 = np.transpose(flow12, [2, 0, 1])
        print(flow12.shape)

        flow21 = data[image_pair21]["flow"]
        # flow21 = np.transpose(flow21, [2, 0, 1])
        # print(flow21.shape)
        flowconf12 = data[image_pair12]["flowconf"]
        flowconf21 = data[image_pair21]["flowconf"]
        print("flowconf12.shape = ", flowconf12.shape, "; flowconf21.shape = ", flowconf21.shape)

        rotation12 = data[image_pair12]["rotation"].value
        rotation21 = data[image_pair21]["rotation"].value
        print("rotation12 = ", rotation12, "; rotation21 = ", rotation21)
        print("rotation12.shape = ", rotation12.shape, "; rotation21.shape = ", rotation21.shape)


        translation12 = data[image_pair12]["translation"].value
        translation21 = data[image_pair21]["translation"].value
        print("translation12 = ", translation12, "; translation21 = ", translation21)


        # ### add code to upsample the predicted optical-flow
        # if OF_scale_factor > 1:
        #     flow12_upsampled = upsample_optical_flow(flow12, OF_scale_factor=OF_scale_factor)
        #     flow21_upsampled = upsample_optical_flow(flow21, OF_scale_factor=OF_scale_factor)
        #     flow12 = flow12_upsampled
        #     flow21 = flow21_upsampled
        #     print("updampled flow12.shape = ", flow12.shape)

        ### cross check from 2-1-2
        matches12, coords121, coords122 = flow_to_matches_float32Pixels(flow21)
        matches21, coords211, coords212 = flow_to_matches_float32Pixels(flow12)

        print("  => Found", matches12.size/2, "<->", matches21.size/2, "matches")
        if  matches12.size/2 <= 0 or matches21.size/2 <= 0:
            continue

        matches, coords_12_1, coords_12_2, all_matches_pixel_consistencies = cross_check_matches_float32Pixel(matches12, coords121, coords122,
                                      matches21, coords211, coords212,
                                      # args.max_pixel_error*OF_scale_factor)
                                      args.max_pixel_error)
                                      # max_reproj_error)
        print("matches.shape = ", matches.shape, "; ", "coords_12_1.shape = ", coords_12_1.shape, "coords_12_2.shape = ", coords_12_2.shape)
        if matches.size == 0:
            continue
        print("  => Cross-checked", matches.shape[0], "matches")

        survivor_ratio_21 = matches.shape[0] / matches12.shape[0]
        avg_flow_consistency_error_212_list.append(np.mean(all_matches_pixel_consistencies[:,2]))


        ### cross check from 1-2-1
        matches12, coords121, coords122 = flow_to_matches_float32Pixels(flow12)
        matches21, coords211, coords212 = flow_to_matches_float32Pixels(flow21)

        print("  => Found", matches12.size/2, "<->", matches21.size/2, "matches")
        if  matches12.size/2 <= 0 or matches21.size/2 <= 0:
            continue

        matches, coords_12_1, coords_12_2, all_matches_pixel_consistencies = cross_check_matches_float32Pixel(matches12, coords121, coords122,
                                      matches21, coords211, coords212,
                                      # args.max_pixel_error*OF_scale_factor)
                                      args.max_pixel_error)
                                      # max_reproj_error)
        print("matches.shape = ", matches.shape, "; ", "coords_12_1.shape = ", coords_12_1.shape, "coords_12_2.shape = ", coords_12_2.shape)
        if matches.size == 0:
            continue
        print("  => Cross-checked", matches.shape[0], "matches")

        survivor_ratio = matches.shape[0] / matches12.shape[0]
        survivor_ratio_list.append(survivor_ratio)
        avg_flow_consistency_error_121_list.append(np.mean(all_matches_pixel_consistencies[:,2]))


        survivor_ratio_list_bidir.append([survivor_ratio, survivor_ratio_21])

        #### rotation errors
        # rotationError = np.dot(rotation12, rotation21)
        r, _ = cv2.Rodrigues(rotation12.dot(rotation21))
        rotation_error_from_identity = np.linalg.norm(r)
        rotation_pose_consistency_list.append(rotation_error_from_identity)

        ColmapR12 = np.dot(ColmapImages[image_name2].R, ColmapImages[image_name1].R.T)
        r, _ = cv2.Rodrigues(rotation12.dot(ColmapR12.T))
        rotation_error_from_ColmapR = np.linalg.norm(r)
        rotation_pose_accuracy_list.append(rotation_error_from_ColmapR*180/math.pi)


        ## translation errors
        cam_pos_12 = -np.dot(rotation12.T, translation12)
        cam_pos_21 = -np.dot(rotation21.T, translation21)
        extrinsic12_4by4 = np.eye(4)
        extrinsic12_4by4[0:3,0:3] = rotation12
        extrinsic12_4by4[0:3,3] = translation12
        cam_pos_2in1_4 = np.ones(4)
        cam_pos_2in1_4[0:3] = cam_pos_12
        cam_pos_2in2_4 = np.dot(extrinsic12_4by4, cam_pos_2in1_4)
        print("cam_pos_2in2_4 = ", cam_pos_2in2_4, ", which should be at the origin (0,0,0,1)")
        cam_pos_1in1_4 = np.zeros(4)
        cam_pos_1in1_4[3] = 1
        cam_pos_1in2_4 = np.dot(extrinsic12_4by4, cam_pos_1in1_4)
        print("cam_pos_1in2_4 = ", cam_pos_1in2_4, ", which should be at ", cam_pos_21)
        cam_pos_error_from_origin = np.linalg.norm(cam_pos_2in2_4[0:3])
        translation_pose_campos_consistency_list.append(cam_pos_error_from_origin)
        camPosVecBy12 = cam_pos_1in2_4[0:3]
        camPosVecBy21 = cam_pos_21
        TransMagBy12 = np.linalg.norm(camPosVecBy12)
        TransMagBy21 = np.linalg.norm(camPosVecBy21)
        # TransDistErr = TransMag12 - TransMag21   # can be different if normalized or not?
        tmp = TheiaClamp(np.dot(camPosVecBy12, camPosVecBy21)/(TransMagBy12*TransMagBy21), -1, 1)   # can be different if normalized or not?
        TransAngularErr = math.acos( tmp )
        translation_pose_camvec_consistency_list.append(TransAngularErr)

        ColmapCamPos2 = -np.dot(ColmapImages[image_name2].R.T, ColmapImages[image_name2].t)
        ColmapCamPos1 = -np.dot(ColmapImages[image_name1].R.T, ColmapImages[image_name1].t)
        cam_pos_12_Colmap = np.dot(ColmapImages[image_name1].R, (ColmapCamPos2-ColmapCamPos1))
        TransMagBy12 = np.linalg.norm(cam_pos_12)
        TransMagBy21 = np.linalg.norm(cam_pos_12_Colmap)
        tmp = TheiaClamp(np.dot(cam_pos_12, cam_pos_12_Colmap)/(TransMagBy12*TransMagBy21), -1, 1)   # can be different if normalized or not?
        TransAngularErrFromColmap = math.acos( tmp )
        translation_pose_camvec_accuracy_list.append(TransAngularErrFromColmap*180/math.pi)

        ### record the length of baseline, view angles of two-views to find out its relationship with the accuracy of predictions
        baselineLength = np.linalg.norm(ColmapCamPos2-ColmapCamPos1)
        twoview_baseline_length_list.append(baselineLength)

        Extrinsic1_groundtruth = np.eye(4)
        Extrinsic1_groundtruth[0:3,0:3] = ColmapImages[image_name1].R
        Extrinsic1_groundtruth[0:3,3] = ColmapImages[image_name1].t
        Extrinsic2_groundtruth = np.eye(4)
        Extrinsic2_groundtruth[0:3,0:3] = ColmapImages[image_name2].R
        Extrinsic2_groundtruth[0:3,3] = ColmapImages[image_name2].t
        camDir1_global = np.dot(np.linalg.inv(Extrinsic1_groundtruth), np.array([0,0,1,0]))
        camDir2_global = np.dot(np.linalg.inv(Extrinsic2_groundtruth), np.array([0,0,1,0]))
        view_angle_imagepair = math.acos(TheiaClamp(np.dot(camDir1_global,camDir2_global) / np.linalg.norm(camDir2_global) / np.linalg.norm(camDir1_global), -1, 1))
        twoview_view_angle_list.append(view_angle_imagepair*180/math.pi)


        # survivor_ratio = matches.shape[0] / matches12.shape[0]
        # survivor_ratio_list.append(survivor_ratio)

        for i in range(all_matches_pixel_consistencies.shape[0]):
            curItem = all_matches_pixel_consistencies[i,:]

            pixel_in_1_y = math.floor(int(curItem[0])/w)
            pixel_in_1_x = (int(curItem[0])%w)
            # print(curItem, "; (x,y) = (", pixel_in_1_x, ",", pixel_in_1_y,")")
            comb_xyconf = np.linalg.norm(flowconf12[:,pixel_in_1_y,pixel_in_1_x])
            pixel_consistency_conf_mat.append([curItem[2], flowconf12[0,pixel_in_1_y,pixel_in_1_x], flowconf12[1,pixel_in_1_y,pixel_in_1_x], comb_xyconf, survivor_ratio, rotation_error_from_identity * 180.0 / math.pi, TransAngularErr * 180.0 / math.pi, survivor_ratio_21])

        # if survivor_ratio >= args.survivor_ratio:
        #     valid_pair_num += 1
        #     print("cross-check-survivor-ratio = ",survivor_ratio)
        #     print("bi-dir survivor-ratio = ", survivor_ratio, " ", survivor_ratio_21)
        if survivor_ratio >= args.survivor_ratio and survivor_ratio_21 >= args.survivor_ratio and rotation_error_from_identity*180/math.pi <= args.rotation_consistency_error_deg and TransAngularErr*180/math.pi <= args.translation_consistency_error_deg:
            valid_pair_num += 1
            print("cross-check-survivor-ratio = ",survivor_ratio)
            print("bi-dir survivor-ratio = ", survivor_ratio, " ", survivor_ratio_21, "; pose consistency errors ", rotation_error_from_identity*180/math.pi, ", ", TransAngularErr*180/math.pi)
                # return
    ### copy the saved quantization map to another file with valid_pair_num and delete the original one
    total_pair_num = len(list(data.keys()))
    print("valid_pair_num = ", valid_pair_num, "/ ", total_pair_num)
    survivor_ratio_list_bidir_npArr = np.array(survivor_ratio_list_bidir)
    print("survivor_ratio_list_bidir_npArr.shape = ", survivor_ratio_list_bidir_npArr.shape)

    plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,3], s=pixel_consistency_conf_numpyArr[:,0], alpha=0.5)#, c=colors, alpha=0.5)
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], survivor_ratio_list_bidir_npArr[:,1], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('survivor_ratio 2-1-2');
    plt.xlabel('survivor_ratio 1-2-1');
    plt.title("survivor_ratio 2-1-2 vs survivor_ratio 1-2-1")
    plt.savefig(os.path.join(args.output_path,"survivor_ratio_2_1_2_vs_survivor_ratio_1_2_1.png"))
    #plt.show()



    # maskInliers = np.array(survivor_ratio_list_bidir_npArr[:,0]>=args.survivor_ratio) & np.array(survivor_ratio_list_bidir_npArr[:,1]>=args.survivor_ratio) & np.array(np.array(rotation_pose_consistency_list) <= args.rotation_consistency_error_deg) & np.array(np.array(translation_pose_camvec_consistency_list) <= args.translation_consistency_error_deg)
    maskInliers = np.array(survivor_ratio_list_bidir_npArr[:,0]>=args.survivor_ratio) & np.array(survivor_ratio_list_bidir_npArr[:,1]>=args.survivor_ratio) & np.array(np.array(rotation_pose_consistency_list) <= args.rotation_consistency_error_deg) & np.array(np.array(translation_pose_camvec_consistency_list) <= args.translation_consistency_error_deg)

    plt.figure()
    plt.scatter(np.array(twoview_baseline_length_list)[maskInliers], np.array(rotation_pose_accuracy_list)[maskInliers], s=1)
    plt.ylabel('rotation_prediction_error_from_GT');
    plt.xlabel('twoview_baseline_length');
    plt.title("rotation_prediction_error_from_GT vs twoview_baseline_length")
    plt.savefig(os.path.join(args.output_path,"rotation_prediction_error_from_GT_vs_twoview_baseline_length.png"))
    #plt.show()

    plt.figure()
    plt.scatter(np.array(twoview_view_angle_list)[maskInliers], np.array(rotation_pose_accuracy_list)[maskInliers], s=1)
    plt.ylabel('rotation_prediction_error_from_GT');
    plt.xlabel('twoview_view_angle');
    plt.title("rotation_prediction_error_from_GT vs twoview_view_angle")
    plt.savefig(os.path.join(args.output_path,"rotation_prediction_error_from_GT_vs_twoview_view_angle.png"))
    #plt.show()

    plt.figure()
    plt.scatter(np.array(twoview_baseline_length_list)[maskInliers], np.array(translation_pose_camvec_accuracy_list)[maskInliers], s=1)
    plt.ylabel('translation_prediction_error_from_GT');
    plt.xlabel('twoview_baseline_length');
    plt.title("translation_prediction_error_from_GT vs twoview_baseline_length")
    plt.savefig(os.path.join(args.output_path,"translation_prediction_error_from_GT_vs_twoview_baseline_length.png"))
    #plt.show()

    plt.figure()
    plt.scatter(np.array(twoview_view_angle_list)[maskInliers], np.array(translation_pose_camvec_accuracy_list)[maskInliers], s=1)
    plt.ylabel('translation_prediction_error_from_GT');
    plt.xlabel('twoview_view_angle');
    plt.title("translation_prediction_error_from_GT vs twoview_view_angle")
    plt.savefig(os.path.join(args.output_path,"translation_prediction_error_from_GT_vs_twoview_view_angle.png"))
    #plt.show()


    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0][maskInliers], np.array(avg_flow_consistency_error_121_list)[maskInliers], s=1)
    plt.ylabel('avg_flow_consistency_error_121');
    plt.xlabel('survivor_ratio_1_2_1');
    plt.title("avg_flow_consistency_error_121 vs survivor_ratio_1_2_1")
    plt.savefig(os.path.join(args.output_path,"avg_flow_consistency_error_121_vs_survivor_ratio_1_2_1.png"))
    #plt.show()

    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], np.array(avg_flow_consistency_error_121_list), s=1)
    plt.ylabel('avg_flow_consistency_error_121');
    plt.xlabel('survivor_ratio_1_2_1');
    plt.title("exhaustivePairs_avg_flow_consistency_error_121 vs survivor_ratio_1_2_1")
    plt.savefig(os.path.join(args.output_path,"exhaustivePairs_avg_flow_consistency_error_121_vs_survivor_ratio_1_2_1.png"))
    #plt.show()


#####################################
    survivorRatioThreshold = 0.4
    print("(survivor_ratio_list_bidir_npArr[:,0]>=survivorRatioThreshold).shape = ", (survivor_ratio_list_bidir_npArr[:,0]>=survivorRatioThreshold).shape)
    print("(survivor_ratio_list_bidir_npArr[:,1]>=survivorRatioThreshold).shape = ", (survivor_ratio_list_bidir_npArr[:,1]>=survivorRatioThreshold).shape)
    maskBothSideSurvivor = np.array(survivor_ratio_list_bidir_npArr[:,0]>=survivorRatioThreshold) & np.array(survivor_ratio_list_bidir_npArr[:,1]>=survivorRatioThreshold)
    # maskEitherSideSurvivor = np.array(survivor_ratio_list_bidir_npArr[:,0]>=survivorRatioThreshold) | np.array(survivor_ratio_list_bidir_npArr[:,1]>=survivorRatioThreshold)
    plt.figure()
    plt.scatter(np.array(translation_pose_camvec_consistency_list)[maskBothSideSurvivor]*180/math.pi, np.array(translation_pose_camvec_accuracy_list)[maskBothSideSurvivor], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('translation_error_from_colmap');
    plt.xlabel('translation_pose_camvec_consistency_list');
    plt.title("biDirSurvivor: translation_error_from_colmap vs translation_pose_camvec_consistency_list")
    plt.savefig(os.path.join(args.output_path,"biDirSurvivor_translation_error_from_colmap_vs_translation_pose_camvec_consistency_list.png"))
    #plt.show()

    plt.figure()
    plt.scatter(np.array(rotation_pose_consistency_list)[maskBothSideSurvivor]*180/math.pi, np.array(rotation_pose_accuracy_list)[maskBothSideSurvivor], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('rotation_error_from_colmap');
    plt.xlabel('rotation_pose_consistency_list');
    plt.title("biDirSurvivor: rotation_error_from_colmap vs rotation_pose_consistency_list")
    plt.savefig(os.path.join(args.output_path,"biDirSurvivor_rotation_error_from_colmap_vs_rotation_pose_consistency_list.png"))
    #plt.show()

    # pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.5,:]
    # plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3])
    # plt.ylabel('flow prediction confidence values');
    # plt.xlabel('pixel consistency errors');
    # plt.title("survivor_ratio >= 0.5: scatter plot of y (flowconf) vs x (pixel consistency error)")
    # plt.savefig(os.path.join(args.output_path,"survivor_ratio_050_flowconf_vs_pixel_consistencey_error.png"))
    # #plt.show()
    # pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.7,:]
    # print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    # plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3])
    # plt.ylabel('flow prediction confidence values');
    # plt.xlabel('pixel consistency errors');
    # plt.title("survivor_ratio >= 0.7: scatter plot of y (flowconf) vs x (pixel consistency error)")
    # plt.savefig(os.path.join(args.output_path,"survivor_ratio_070_flowconf_vs_pixel_consistencey_error.png"))
    # #plt.show()
    # pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.9,:]
    # print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    # plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3])
    # plt.ylabel('flow prediction confidence values');
    # plt.xlabel('pixel consistency errors');
    # plt.title("survivor_ratio >= 0.9: scatter plot of y (flowconf) vs x (pixel consistency error)")
    # plt.savefig(os.path.join(args.output_path,"survivor_ratio_090_flowconf_vs_pixel_consistencey_error.png"))
    # #plt.show()

    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[maskBothSideSurvivor,0], np.array(translation_pose_camvec_accuracy_list)[maskBothSideSurvivor], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('translation_error_from_colmap');
    plt.xlabel('survivor_ratio 1-2-1 (survivor_ratio 2-1-2)');
    plt.title("translation_error_from_colmap vs survivor_ratio 1-2-1 (survivor_ratio 2-1-2)")
    plt.savefig(os.path.join(args.output_path,"translation_error_from_colmap_vs_survivor_ratio_1_2_1_survivor_ratio_2_1_2.png"))
    #plt.show()

    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[maskBothSideSurvivor,0], np.array(rotation_pose_accuracy_list)[maskBothSideSurvivor], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('rotation_error_from_colmap');
    plt.xlabel('survivor_ratio 1-2-1 (survivor_ratio 2-1-2)');
    plt.title("rotation_error_from_colmap vs survivor_ratio 1-2-1 (survivor_ratio 2-1-2)")
    plt.savefig(os.path.join(args.output_path,"rotation_error_from_colmap_vs_survivor_ratio_1_2_1_survivor_ratio_2_1_2.png"))
    #plt.show()

#####################################

    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], translation_pose_camvec_accuracy_list, s=1)#, c=colors, alpha=0.5)
    plt.ylabel('translation_error_from_colmap');
    plt.xlabel('survivor_ratio 1-2-1');
    plt.title("translation_error_from_colmap vs survivor_ratio 1-2-1")
    plt.savefig(os.path.join(args.output_path,"translation_error_from_colmap_vs_survivor_ratio_1_2_1.png"))
    #plt.show()

    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], rotation_pose_accuracy_list, s=1)#, c=colors, alpha=0.5)
    plt.ylabel('rotation_error_from_colmap');
    plt.xlabel('survivor_ratio 1-2-1');
    plt.title("rotation_error_from_colmap vs survivor_ratio 1-2-1")
    plt.savefig(os.path.join(args.output_path,"rotation_error_from_colmap_vs_survivor_ratio_1_2_1.png"))
    #plt.show()


    plt.figure()
    plt.scatter(np.array(translation_pose_camvec_consistency_list)*180/math.pi, translation_pose_camvec_accuracy_list, s=1)#, c=colors, alpha=0.5)
    plt.ylabel('translation_error_from_colmap');
    plt.xlabel('translation_pose_camvec_consistency_list');
    plt.title("translation_error_from_colmap vs translation_pose_camvec_consistency_list")
    plt.savefig(os.path.join(args.output_path,"translation_error_from_colmap_vs_translation_pose_camvec_consistency_list.png"))
    #plt.show()

    plt.figure()
    plt.scatter(np.array(rotation_pose_consistency_list)*180/math.pi, rotation_pose_accuracy_list, s=1)#, c=colors, alpha=0.5)
    plt.ylabel('rotation_error_from_colmap');
    plt.xlabel('rotation_pose_consistency_list');
    plt.title("rotation_error_from_colmap vs rotation_pose_consistency_list")
    plt.savefig(os.path.join(args.output_path,"rotation_error_from_colmap_vs_rotation_pose_consistency_list.png"))
    #plt.show()

    plt.figure()
    plt.hist(survivor_ratio_list, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("survivor_ratio_list histogram")
    plt.savefig(os.path.join(args.output_path,"survivor_ratio_list_histogram.png"))
    #plt.show()

    plt.figure()
    plt.hist(rotation_pose_consistency_list, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("rotation_pose_consistency_list histogram")
    plt.savefig(os.path.join(args.output_path,"rotation_pose_consistency_list_histogram.png"))
    #plt.show()
    plt.figure()
    plt.hist(np.array(rotation_pose_consistency_list) * 180.0 / math.pi, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("rotation_pose_consistency_list histogram in degrees")
    plt.savefig(os.path.join(args.output_path,"rotation_pose_consistency_list_histogram_inDegrees.png"))
    #plt.show()

    plt.figure()
    plt.hist(translation_pose_campos_consistency_list, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("translation_pose_campos_consistency_list histogram")
    plt.savefig(os.path.join(args.output_path,"translation_pose_campos_consistency_list_histogram.png"))
    #plt.show()
    plt.figure()
    plt.hist(translation_pose_camvec_consistency_list, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("translation_pose_camvec_consistency_list histogram in radians")
    plt.savefig(os.path.join(args.output_path,"translation_pose_camvec_consistency_list_histogram_inRadians.png"))
    #plt.show()
    plt.figure()
    plt.hist(np.array(translation_pose_camvec_consistency_list) * 180.0 / math.pi, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("translation_pose_camvec_consistency_list histogram in degrees")
    plt.savefig(os.path.join(args.output_path,"translation_pose_camvec_consistency_list_histogram_inDegrees.png"))
    #plt.show()


    plt.figure()
    plt.hist(rotation_pose_accuracy_list, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("rotation_pose_accuracy_list histogram in degrees")
    plt.savefig(os.path.join(args.output_path,"rotation_pose_accuracy_list_histogram_inDegrees.png"))
    #plt.show()
    plt.figure()
    plt.hist(translation_pose_camvec_accuracy_list, normed=False, bins=int(total_pair_num/20))
    plt.ylabel('Counts');
    plt.title("translation_pose_camvec_accuracy_list histogram in degrees")
    plt.savefig(os.path.join(args.output_path,"translation_pose_camvec_accuracy_list_histogram_inDegrees.png"))
    #plt.show()



    pixel_consistency_conf_numpyArr = np.array(pixel_consistency_conf_mat)
    print("### pixel_consistency_conf_numpyArr.shape = ", pixel_consistency_conf_numpyArr.shape)
    # np.save(os.path.join(args.output_path,"pixel_consistency_conf_numpyArr.npy"), pixel_consistency_conf_numpyArr)
    plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,1], s=pixel_consistency_conf_numpyArr[:,0], alpha=0.5)#, c=colors, alpha=0.5)
    plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,1], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('flow x prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("scatter plot of y (flowconf x) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"flowconf_x_vs_pixel_consistencey_error.png"))
    #plt.show()
    plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,2], s=pixel_consistency_conf_numpyArr[:,0], alpha=0.5)#, c=colors, alpha=0.5)
    plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,2], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('flow y prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("scatter plot of y (flowconf y) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"flowconf_y_vs_pixel_consistencey_error.png"))
    #plt.show()
    plt.figure()
    # plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,3], s=pixel_consistency_conf_numpyArr[:,0], alpha=0.5)#, c=colors, alpha=0.5)
    plt.scatter(pixel_consistency_conf_numpyArr[:,0], pixel_consistency_conf_numpyArr[:,3], s=1)#, c=colors, alpha=0.5)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()

    ### filter by survivor_ratio and check inlier pairs correlations
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.3,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("survivor_ratio >= 0.3: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"survivor_ratio_030_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.5,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("survivor_ratio >= 0.5: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"survivor_ratio_050_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.7,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("survivor_ratio >= 0.7: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"survivor_ratio_070_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,4]>=0.9,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("survivor_ratio >= 0.9: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"survivor_ratio_090_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()

    ### filter by pixel_consistency_err and check inlier pairs correlations
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=1,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("pixel_consistency_err <= 1 pixels: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"pixel_consistency_err_1_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=2,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("pixel_consistency_err <= 2 pixels: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"pixel_consistency_err_2_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=3,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("pixel_consistency_err <= 3 pixels: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"pixel_consistency_err_3_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=5,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("pixel_consistency_err <= 5 pixels: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"pixel_consistency_err_5_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()
    pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=10,:]
    print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    plt.figure()
    plt.scatter(pixel_consistency_conf_numpyArr_filtered[:,0], pixel_consistency_conf_numpyArr_filtered[:,3], s=1)
    plt.ylabel('flow prediction confidence values');
    plt.xlabel('pixel consistency errors');
    plt.title("pixel_consistency_err <= 10 pixels: scatter plot of y (flowconf) vs x (pixel consistency error)")
    plt.savefig(os.path.join(args.output_path,"pixel_consistency_err_10_flowconf_vs_pixel_consistencey_error.png"))
    #plt.show()


    ### max_pixel_error_1_inliers_flowconf_histogram
    # pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=1,:]
    # print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    # plt.figure()
    # bins = np.linspace(0, 1.6, 320)
    # plt.hist([pixel_consistency_conf_numpyArr_filtered[:,1], pixel_consistency_conf_numpyArr_filtered[:,2], pixel_consistency_conf_numpyArr_filtered[:,3]], bins, label=['x', 'y', '|xy|'])
    # plt.ylabel('Counts');
    # plt.xlabel('flow confidence');
    # plt.title("pixel_error <= 1, flowconf histogram")
    # plt.legend(loc='upper right')
    # plt.savefig(os.path.join(args.output_path,"max_pixel_error_1_inliers_flowconf_histogram.png"))
    # #plt.show()
    #
    # pixel_consistency_conf_numpyArr_filtered = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>1,:]
    # print("pixel_consistency_conf_numpyArr_filtered.shape = ", pixel_consistency_conf_numpyArr_filtered.shape)
    # plt.figure()
    # bins = np.linspace(0, 1.6, 320)
    # plt.hist([pixel_consistency_conf_numpyArr_filtered[:,1], pixel_consistency_conf_numpyArr_filtered[:,2], pixel_consistency_conf_numpyArr_filtered[:,3]], bins, label=['x', 'y', '|xy|'])
    # plt.ylabel('Counts');
    # plt.xlabel('flow confidence');
    # plt.title("pixel_error > 1, flowconf histogram")
    # plt.legend(loc='upper right')
    # plt.savefig(os.path.join(args.output_path,"max_pixel_error_1_outliers_flowconf_histogram.png"))
    # #plt.show()
    #
    # # pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=1,:]
    # # print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    # # pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>1,:]
    # # print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    # # plt.figure()
    # # bins = np.linspace(0, 1.6, 320)
    # # plt.hist([pixel_consistency_conf_numpyArr_inliers[:,1], pixel_consistency_conf_numpyArr_inliers[:,2], pixel_consistency_conf_numpyArr_inliers[:,3], pixel_consistency_conf_numpyArr_outliers[:,1], pixel_consistency_conf_numpyArr_outliers[:,2], pixel_consistency_conf_numpyArr_outliers[:,3]], bins, label=['x-inliers', 'y-inliers', '|xy|-inliers', 'x-outliers', 'y-outliers', '|xy|-outliers'])
    # # plt.ylabel('Counts');
    # # plt.xlabel('flow confidence');
    # # plt.title("pixel_error <= 1, flowconf histogram")
    # # plt.legend(loc='upper right')
    # # plt.savefig(os.path.join(args.output_path,"max_pixel_error_1_flowconf_histogram.png"))
    # # #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=1,:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>1,:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(0.9, 1.0, 100)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,1], pixel_consistency_conf_numpyArr_inliers[:,2], pixel_consistency_conf_numpyArr_outliers[:,1], pixel_consistency_conf_numpyArr_outliers[:,2]], bins, label=['x-inliers', 'y-inliers', 'x-outliers', 'y-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("pixel_error <= 1, flowconf histogram")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"max_pixel_error_1_flowconf_histogram.png"))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=1,:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>1,:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(1.2, 1.5, 200)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,3], pixel_consistency_conf_numpyArr_outliers[:,3]], bins, label=['|xy|-inliers', '|xy|-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("pixel_error <= 1, flowconf_xy histogram")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"max_pixel_error_1_flowconf_xy_histogram.png"))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=3,:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>3,:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(0.9, 1.0, 100)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,1], pixel_consistency_conf_numpyArr_inliers[:,2], pixel_consistency_conf_numpyArr_outliers[:,1], pixel_consistency_conf_numpyArr_outliers[:,2]], bins, label=['x-inliers', 'y-inliers', 'x-outliers', 'y-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("pixel_error <= 3, flowconf histogram")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"max_pixel_error_3_flowconf_histogram.png"))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<3,:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>3,:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(1.2, 1.5, 200)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,3], pixel_consistency_conf_numpyArr_outliers[:,3]], bins, label=['|xy|-inliers', '|xy|-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("pixel_error <= 3, flowconf_xy histogram")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"max_pixel_error_3_flowconf_xy_histogram.png"))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=0.2,:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>0.2,:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(0.9, 1.0, 100)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,1], pixel_consistency_conf_numpyArr_inliers[:,2], pixel_consistency_conf_numpyArr_outliers[:,1], pixel_consistency_conf_numpyArr_outliers[:,2]], bins, label=['x-inliers', 'y-inliers', 'x-outliers', 'y-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("pixel_error <= 0.2, flowconf histogram")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"max_pixel_error_02_flowconf_histogram.png"))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]<=0.2,:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[pixel_consistency_conf_numpyArr[:,0]>0.2,:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(1.2, 1.5, 200)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,3], pixel_consistency_conf_numpyArr_outliers[:,3]], bins, label=['|xy|-inliers', '|xy|-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("pixel_error <= 0.2, flowconf_xy histogram")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"max_pixel_error_02_flowconf_xy_histogram.png"))
    #plt.show()


    ### check the correlation between survivor_ratio and prediction consistency
    print("len(survivor_ratio_list_bidir_npArr) = ", len(survivor_ratio_list_bidir_npArr))
    print("len(translation_pose_camvec_consistency_list) = ", len(translation_pose_camvec_consistency_list))
    print("len(translation_pose_campos_consistency_list) = ", len(translation_pose_campos_consistency_list))
    print("len(rotation_pose_consistency_list) = ", len(rotation_pose_consistency_list))
    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], translation_pose_camvec_consistency_list, s=1)
    plt.ylabel('translation_pose_camvec_consistency');
    plt.xlabel('survivor_ratio');
    plt.title("scatter plot of y (translation_pose_camvec_consistency) vs x (survivor_ratio)")
    plt.savefig(os.path.join(args.output_path,"translation_pose_camvec_consistency_vs_survivor_ratio.png"))
    #plt.show()
    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], translation_pose_campos_consistency_list, s=1)
    plt.ylabel('translation_pose_campos_consistency');
    plt.xlabel('survivor_ratio');
    plt.title("scatter plot of y (translation_pose_campos_consistency) vs x (survivor_ratio)")
    plt.savefig(os.path.join(args.output_path,"translation_pose_campos_consistency_vs_survivor_ratio.png"))
    #plt.show()
    plt.figure()
    plt.scatter(survivor_ratio_list_bidir_npArr[:,0], rotation_pose_consistency_list, s=1)
    plt.ylabel('rotation_pose_consistency_list');
    plt.xlabel('survivor_ratio');
    plt.title("scatter plot of y (rotation_pose_consistency_list) vs x (survivor_ratio)")
    plt.savefig(os.path.join(args.output_path,"rotation_pose_consistency_list_vs_survivor_ratio.png"))
    #plt.show()


    ### check the filtering by survivor_ration + relative rotation consistency
    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]<=1) & (pixel_consistency_conf_numpyArr[:,5]<=float(args.rotation_consistency_error_deg)),:] #  & (pixel_consistency_conf_numpyArr[:,6]<=float(args.translation_consistency_error_deg))
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]>1) | (pixel_consistency_conf_numpyArr[:,5]>float(args.rotation_consistency_error_deg)),:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(0.9, 1.0, 100)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,1], pixel_consistency_conf_numpyArr_inliers[:,2], pixel_consistency_conf_numpyArr_outliers[:,1], pixel_consistency_conf_numpyArr_outliers[:,2]], bins, label=['x-inliers', 'y-inliers', 'x-outliers', 'y-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("rotation_consistency_error <= %s & pixel_error <= 1, flowconf histogram" % (args.rotation_consistency_error_deg))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"rotation_consistency_error_%s_max_pixel_error_1_flowconf_histogram.png" % (args.rotation_consistency_error_deg)))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]<=1) & (pixel_consistency_conf_numpyArr[:,5]<=float(args.rotation_consistency_error_deg)),:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]>1) | (pixel_consistency_conf_numpyArr[:,5]>float(args.rotation_consistency_error_deg)),:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(1.2, 1.5, 200)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,3], pixel_consistency_conf_numpyArr_outliers[:,3]], bins, label=['|xy|-inliers', '|xy|-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("rotation_consistency_error <= %s & pixel_error <= 1, flowconf |x,y| histogram" % (args.rotation_consistency_error_deg))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"rotation_consistency_error_%s_max_pixel_error_1_flowconf_xy_histogram.png" % (args.rotation_consistency_error_deg)))
    # plt.show()

    ### check the filtering by survivor_ration + relative rotation consistency + relative translation consistency
    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]<=1) & (pixel_consistency_conf_numpyArr[:,5]<=float(args.rotation_consistency_error_deg)) & (pixel_consistency_conf_numpyArr[:,6]<=float(args.translation_consistency_error_deg)),:] #  & (pixel_consistency_conf_numpyArr[:,6]<=float(args.translation_consistency_error_deg))
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]>1) | (pixel_consistency_conf_numpyArr[:,5]>float(args.rotation_consistency_error_deg)) | (pixel_consistency_conf_numpyArr[:,6]>float(args.translation_consistency_error_deg)),:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(0.9, 1.0, 100)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,1], pixel_consistency_conf_numpyArr_inliers[:,2], pixel_consistency_conf_numpyArr_outliers[:,1], pixel_consistency_conf_numpyArr_outliers[:,2]], bins, label=['x-inliers', 'y-inliers', 'x-outliers', 'y-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("rotation_consistency_error <= %s & translation_consistency_error <= %s & pixel_error <= 1, flowconf histogram" % (args.rotation_consistency_error_deg, args.translation_consistency_error_deg))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"rotation_consistency_error_%s_translation_consistency_error_%s_max_pixel_error_1_flowconf_histogram.png" % (args.rotation_consistency_error_deg, args.translation_consistency_error_deg)))
    #plt.show()

    pixel_consistency_conf_numpyArr_inliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]<=1) & (pixel_consistency_conf_numpyArr[:,5]<=float(args.rotation_consistency_error_deg)) & (pixel_consistency_conf_numpyArr[:,6]<=float(args.translation_consistency_error_deg)),:]
    print("pixel_consistency_conf_numpyArr_inliers.shape = ", pixel_consistency_conf_numpyArr_inliers.shape)
    pixel_consistency_conf_numpyArr_outliers = pixel_consistency_conf_numpyArr[(pixel_consistency_conf_numpyArr[:,0]>1) | (pixel_consistency_conf_numpyArr[:,5]>float(args.rotation_consistency_error_deg)) | (pixel_consistency_conf_numpyArr[:,6]>float(args.translation_consistency_error_deg)),:]
    print("pixel_consistency_conf_numpyArr_outliers.shape = ", pixel_consistency_conf_numpyArr_outliers.shape)
    plt.figure()
    bins = np.linspace(1.2, 1.5, 200)
    plt.hist([pixel_consistency_conf_numpyArr_inliers[:,3], pixel_consistency_conf_numpyArr_outliers[:,3]], bins, label=['|xy|-inliers', '|xy|-outliers'])
    plt.ylabel('Counts');
    plt.xlabel('flow confidence');
    plt.title("rotation_consistency_error <= %s & translation_consistency_error <= %s & pixel_error <= 1, flowconf |x,y| histogram" % (args.rotation_consistency_error_deg, args.translation_consistency_error_deg))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.output_path,"rotation_consistency_error_%s_translation_consistency_error_%s_max_pixel_error_1_flowconf_xy_histogram.png" % (args.rotation_consistency_error_deg, args.translation_consistency_error_deg)))
    # plt.show()


if __name__ == "__main__":
    main()
