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
    parser.add_argument("--GT_depth_dir_path", required=False)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--demon_path", required=True)
    parser.add_argument("--input_good_pairs_path", required=True)
    # parser.add_argument("--scale_factor", type=int, default=24)
    parser.add_argument("--min_num_features", type=int, default=1)
    parser.add_argument("--ratio_threshold", type=float, default=0.75)
    parser.add_argument("--max_descriptor_distance", type=float, default=1.00)
    parser.add_argument("--max_pixel_error", type=float, default=1.00)
    parser.add_argument("--OF_scale_factor", type=int, default=1)
    parser.add_argument("--survivor_ratio", type=float, default=0.50)
    parser.add_argument("--rotation_consistency_error_deg", type=float, default=360.0)
    parser.add_argument("--translation_consistency_error_deg", type=float, default=360.0)
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
        os.stat(os.path.join(args.output_path, 'thesis_report'))
    except:
        os.mkdir(args.output_path)
        os.mkdir(os.path.join(args.output_path, 'thesis_report'))

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

    max_sq_pixel_errs = [1, 4, 9, 16, 25, 36, 49, 64]
    for max_sq_pixel_err in max_sq_pixel_errs:
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
        twoview_medianDepthOverBaseline = []
        twoview_meanDepthOverBaseline = []


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
                                          max_sq_pixel_err)
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
                                          max_sq_pixel_err)
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

            tmpGTDepth1 = np.fromfile(os.path.join(args.GT_depth_dir_path, image_name1),dtype=np.float32)
            # tmpGTDepth1[np.isinf(tmpGTDepth1)]=0.0
            # tmpGTDepth1 = np.reshape(tmpGTDepth1, [4032,6048])
            median_GT_Depth_1 = np.nanmedian(tmpGTDepth1[~np.isinf(tmpGTDepth1)])
            mean_GT_Depth_1 = np.nanmean(tmpGTDepth1[~np.isinf(tmpGTDepth1)])
            print(median_GT_Depth_1, ", ", mean_GT_Depth_1, ", ", baselineLength)
            twoview_medianDepthOverBaseline.append(median_GT_Depth_1/baselineLength)
            twoview_meanDepthOverBaseline.append(mean_GT_Depth_1/baselineLength)

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


            # # survivor_ratio = matches.shape[0] / matches12.shape[0]
            # # survivor_ratio_list.append(survivor_ratio)
            #
            # for i in range(all_matches_pixel_consistencies.shape[0]):
            #     curItem = all_matches_pixel_consistencies[i,:]
            #
            #     pixel_in_1_y = math.floor(int(curItem[0])/w)
            #     pixel_in_1_x = (int(curItem[0])%w)
            #     # print(curItem, "; (x,y) = (", pixel_in_1_x, ",", pixel_in_1_y,")")
            #     comb_xyconf = np.linalg.norm(flowconf12[:,pixel_in_1_y,pixel_in_1_x])
            #     pixel_consistency_conf_mat.append([curItem[2], flowconf12[0,pixel_in_1_y,pixel_in_1_x], flowconf12[1,pixel_in_1_y,pixel_in_1_x], comb_xyconf, survivor_ratio, rotation_error_from_identity * 180.0 / math.pi, TransAngularErr * 180.0 / math.pi, survivor_ratio_21])
            #
            # # if survivor_ratio >= args.survivor_ratio:
            # #     valid_pair_num += 1
            # #     print("cross-check-survivor-ratio = ",survivor_ratio)
            # #     print("bi-dir survivor-ratio = ", survivor_ratio, " ", survivor_ratio_21)
            # if survivor_ratio >= args.survivor_ratio and survivor_ratio_21 >= args.survivor_ratio and rotation_error_from_identity*180/math.pi <= args.rotation_consistency_error_deg and TransAngularErr*180/math.pi <= args.translation_consistency_error_deg:
            #     valid_pair_num += 1
            #     print("cross-check-survivor-ratio = ",survivor_ratio)
            #     print("bi-dir survivor-ratio = ", survivor_ratio, " ", survivor_ratio_21, "; pose consistency errors ", rotation_error_from_identity*180/math.pi, ", ", TransAngularErr*180/math.pi)
            #         # return
        ### copy the saved quantization map to another file with valid_pair_num and delete the original one
        # total_pair_num = len(list(data.keys()))
        # print("valid_pair_num = ", valid_pair_num, "/ ", total_pair_num)
        survivor_ratio_list_bidir_npArr = np.array(survivor_ratio_list_bidir)
        print("survivor_ratio_list_bidir_npArr.shape = ", survivor_ratio_list_bidir_npArr.shape)

        survivor_ratio_threshold_list = np.linspace(0.0, 1.0, num=51, endpoint=True)
        survivor_pair_num_list = []
        update1_survivor_pair_num_list = []
        update2_survivor_pair_num_list = []
        update3_survivor_pair_num_list = []
        update4_survivor_pair_num_list = []
        update5_survivor_pair_num_list = []

        survivor_mask_5_5_pair_num_from_inliers_5_5 = []
        survivor_mask_15_15_pair_num_from_inliers_15_15 = []
        survivor_mask_25_25_pair_num_from_inliers_25_25 = []
        survivor_mask_35_35_pair_num_from_inliers_35_35 = []

        survivor_mask_5_5_pair_num_from_inliers_25_25 = []
        survivor_mask_10_10_pair_num_from_inliers_25_25 = []
        survivor_mask_15_15_pair_num_from_inliers_25_25 = []
        survivor_mask_25_25_pair_num_from_inliers_25_25 = []
        survivor_mask_35_35_pair_num_from_inliers_25_25 = []
        survivor_mask_50_50_pair_num_from_inliers_25_25 = []
        survivor_mask_75_75_pair_num_from_inliers_25_25 = []
        survivor_mask_100_100_pair_num_from_inliers_25_25 = []
        survivor_mask_150_150_pair_num_from_inliers_25_25 = []

        survivor_mask_25_25_percentage = []
        survivor_mask_25_25_pair_num_from_inliers_5_5 = []
        survivor_mask_25_25_pair_num_from_inliers_15_15 = []
        survivor_mask_25_25_pair_num_from_inliers_25_25 = []
        survivor_mask_25_25_pair_num_from_inliers_35_35 = []
        survivor_mask_25_25_pair_num_from_inliers_50_50 = []
        survivor_mask_25_25_pair_num_from_inliers_75_75 = []
        survivor_mask_25_25_pair_num_from_inliers_100_100 = []
        survivor_mask_25_25_pair_num_from_inliers_150_150 = []


        survivor_mask_360_360_pair_num_from_inliers_5_5 = []
        survivor_mask_360_360_pair_num_from_inliers_15_15 = []
        survivor_mask_360_360_pair_num_from_inliers_25_25 = []
        survivor_mask_360_360_pair_num_from_inliers_35_35 = []
        survivor_mask_360_360_pair_num_from_inliers_50_50 = []
        survivor_mask_360_360_pair_num_from_inliers_75_75 = []
        survivor_mask_360_360_pair_num_from_inliers_100_100 = []
        survivor_mask_360_360_pair_num_from_inliers_150_150 = []

        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_5_5 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_15_15 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_25_25 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_35_35 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_50_50 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_75_75 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_100_100 = []
        inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_150_150 = []

        survivor_mask_25_50_percentage = []
        survivor_mask_25_50_pair_num_from_inliers_5_5 = []
        survivor_mask_25_50_pair_num_from_inliers_15_15 = []
        survivor_mask_25_50_pair_num_from_inliers_25_25 = []
        survivor_mask_25_50_pair_num_from_inliers_35_35 = []

        good_pair_mask_25_50_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=50)&np.array(np.array(rotation_pose_accuracy_list)<=25))
        good_pair_mask_25_25_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=25)&np.array(np.array(rotation_pose_accuracy_list)<=25))
        good_pair_mask_15_30_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=30)&np.array(np.array(rotation_pose_accuracy_list)<=15))
        good_pair_mask_15_15_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=15)&np.array(np.array(rotation_pose_accuracy_list)<=15))

        good_pair_mask_5_10_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=10)&np.array(np.array(rotation_pose_accuracy_list)<=5))
        good_pair_mask_5_5_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=5)&np.array(np.array(rotation_pose_accuracy_list)<=5))
        good_pair_mask_10_10_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=10)&np.array(np.array(rotation_pose_accuracy_list)<=10))
        good_pair_mask_30_30_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=30)&np.array(np.array(rotation_pose_accuracy_list)<=30))
        good_pair_mask_35_35_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=35)&np.array(np.array(rotation_pose_accuracy_list)<=35))

        good_pair_mask_50_50_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=50)&np.array(np.array(rotation_pose_accuracy_list)<=50))
        good_pair_mask_75_75_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=75)&np.array(np.array(rotation_pose_accuracy_list)<=75))
        good_pair_mask_100_100_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=100)&np.array(np.array(rotation_pose_accuracy_list)<=100))
        good_pair_mask_150_150_from_allpairs = (np.array(np.array(translation_pose_camvec_accuracy_list)<=150)&np.array(np.array(rotation_pose_accuracy_list)<=150))

        rot_accuracy_mean = []
        rot_accuracy_median = []
        rot_accuracy_std = []
        trans_accuracy_mean = []
        trans_accuracy_median = []
        trans_accuracy_std = []


        for tmpSurvivorRatio in survivor_ratio_threshold_list:
            survivor_pair_num = np.sum(np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_pair_num_list.append(survivor_pair_num)
            update1_survivor_pair_num = np.sum(np.array(np.array(rotation_pose_accuracy_list)<=25))
            update1_survivor_pair_num_list.append(update1_survivor_pair_num)
            update2_survivor_pair_num = np.sum(np.array(np.array(rotation_pose_accuracy_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            update2_survivor_pair_num_list.append(update2_survivor_pair_num)
            update3_survivor_pair_num = np.sum(np.array(np.array(translation_pose_camvec_consistency_list)<=50)&np.array(np.array(rotation_pose_consistency_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            update3_survivor_pair_num_list.append(update3_survivor_pair_num)
            update4_survivor_pair_num = np.sum(np.array(np.array(translation_pose_camvec_accuracy_list)<=50)&np.array(np.array(rotation_pose_accuracy_list)<=25)&np.array(np.array(translation_pose_camvec_consistency_list)<=50)&np.array(np.array(rotation_pose_consistency_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            update4_survivor_pair_num_list.append(update4_survivor_pair_num)
            update5_survivor_pair_num = np.sum(np.array(np.array(translation_pose_camvec_accuracy_list)<=50)&np.array(np.array(rotation_pose_accuracy_list)<=25))
            update5_survivor_pair_num_list.append(update5_survivor_pair_num)

            print(survivor_pair_num, " pairs survive if ratio threshold is set to ", tmpSurvivorRatio)

            survivor_mask_INF_INF_from_allpairs = (np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_INF_INF_from_allpairs = np.array(survivor_mask_INF_INF_from_allpairs)
            rotation_pose_accuracy_npArr = np.array(rotation_pose_accuracy_list)
            translation_pose_camvec_accuracy_npArr = np.array(translation_pose_camvec_accuracy_list)
            rot_accuracy_mean.append(np.mean(rotation_pose_accuracy_npArr[survivor_mask_INF_INF_from_allpairs]))
            rot_accuracy_median.append(np.median(rotation_pose_accuracy_npArr[survivor_mask_INF_INF_from_allpairs]))
            rot_accuracy_std.append(np.std(rotation_pose_accuracy_npArr[survivor_mask_INF_INF_from_allpairs]))
            trans_accuracy_mean.append(np.mean(translation_pose_camvec_accuracy_npArr[survivor_mask_INF_INF_from_allpairs]))
            trans_accuracy_median.append(np.median(translation_pose_camvec_accuracy_npArr[survivor_mask_INF_INF_from_allpairs]))
            trans_accuracy_std.append(np.std(translation_pose_camvec_accuracy_npArr[survivor_mask_INF_INF_from_allpairs]))

            survivor_mask_5_5_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=5)&np.array(np.array(rotation_pose_consistency_list)<=5)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_5_5_pair_num_from_inliers_5_5.append(np.sum(np.array(survivor_mask_5_5_from_allpairs&good_pair_mask_5_5_from_allpairs))/np.sum(good_pair_mask_5_5_from_allpairs))

            survivor_mask_15_15_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=15)&np.array(np.array(rotation_pose_consistency_list)<=15)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_15_15_pair_num_from_inliers_15_15.append(np.sum(np.array(survivor_mask_15_15_from_allpairs&good_pair_mask_15_15_from_allpairs))/np.sum(good_pair_mask_15_15_from_allpairs))

            survivor_mask_25_25_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=25)&np.array(np.array(rotation_pose_consistency_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            #survivor_mask_25_25_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))

            survivor_mask_35_35_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=35)&np.array(np.array(rotation_pose_consistency_list)<=35)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_35_35_pair_num_from_inliers_35_35.append(np.sum(np.array(survivor_mask_35_35_from_allpairs&good_pair_mask_35_35_from_allpairs))/np.sum(good_pair_mask_35_35_from_allpairs))

            survivor_mask_50_50_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=50)&np.array(np.array(rotation_pose_consistency_list)<=50)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_75_75_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=75)&np.array(np.array(rotation_pose_consistency_list)<=75)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_100_100_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=100)&np.array(np.array(rotation_pose_consistency_list)<=100)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_150_150_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=150)&np.array(np.array(rotation_pose_consistency_list)<=150)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))


            survivor_mask_25_25_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=25)&np.array(np.array(rotation_pose_consistency_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_25_25_pair_num_from_inliers_5_5.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_5_5_from_allpairs))/np.sum(good_pair_mask_5_5_from_allpairs))
            survivor_mask_25_25_pair_num_from_inliers_15_15.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_15_15_from_allpairs))/np.sum(good_pair_mask_15_15_from_allpairs))
            survivor_mask_25_25_pair_num_from_inliers_35_35.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_35_35_from_allpairs))/np.sum(good_pair_mask_35_35_from_allpairs))

            survivor_mask_25_25_pair_num_from_inliers_50_50.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_50_50_from_allpairs))/np.sum(good_pair_mask_50_50_from_allpairs))
            survivor_mask_25_25_pair_num_from_inliers_75_75.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_75_75_from_allpairs))/np.sum(good_pair_mask_75_75_from_allpairs))
            survivor_mask_25_25_pair_num_from_inliers_100_100.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_100_100_from_allpairs))/np.sum(good_pair_mask_100_100_from_allpairs))
            survivor_mask_25_25_pair_num_from_inliers_150_150.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_150_150_from_allpairs))/np.sum(good_pair_mask_150_150_from_allpairs))


            survivor_mask_25_50_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=50)&np.array(np.array(rotation_pose_consistency_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_25_50_pair_num_from_inliers_5_5.append(np.sum(np.array(survivor_mask_25_50_from_allpairs&good_pair_mask_5_5_from_allpairs))/np.sum(good_pair_mask_5_5_from_allpairs))
            survivor_mask_25_50_pair_num_from_inliers_15_15.append(np.sum(np.array(survivor_mask_25_50_from_allpairs&good_pair_mask_15_15_from_allpairs))/np.sum(good_pair_mask_15_15_from_allpairs))
            survivor_mask_25_50_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_25_50_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_25_50_pair_num_from_inliers_35_35.append(np.sum(np.array(survivor_mask_25_50_from_allpairs&good_pair_mask_35_35_from_allpairs))/np.sum(good_pair_mask_35_35_from_allpairs))

            survivor_mask_5_5_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=5)&np.array(np.array(rotation_pose_consistency_list)<=5)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_15_15_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=15)&np.array(np.array(rotation_pose_consistency_list)<=15)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_25_25_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=25)&np.array(np.array(rotation_pose_consistency_list)<=25)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_35_35_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=35)&np.array(np.array(rotation_pose_consistency_list)<=35)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_5_5_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_5_5_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_15_15_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_15_15_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_25_25_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_25_25_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_35_35_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_35_35_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))

            survivor_mask_50_50_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_50_50_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_75_75_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_75_75_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_100_100_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_100_100_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_150_150_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_150_150_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))


            survivor_mask_360_360_from_allpairs = (np.array(np.array(translation_pose_camvec_consistency_list)<=360)&np.array(np.array(rotation_pose_consistency_list)<=360)&np.array(survivor_ratio_list_bidir_npArr[:,0]>=tmpSurvivorRatio)&np.array(survivor_ratio_list_bidir_npArr[:,1]>=tmpSurvivorRatio))
            survivor_mask_360_360_pair_num_from_inliers_5_5.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_5_5_from_allpairs))/np.sum(good_pair_mask_5_5_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_15_15.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_15_15_from_allpairs))/np.sum(good_pair_mask_15_15_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(good_pair_mask_25_25_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_35_35.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_35_35_from_allpairs))/np.sum(good_pair_mask_35_35_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_50_50.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_50_50_from_allpairs))/np.sum(good_pair_mask_50_50_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_75_75.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_75_75_from_allpairs))/np.sum(good_pair_mask_75_75_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_100_100.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_100_100_from_allpairs))/np.sum(good_pair_mask_100_100_from_allpairs))
            survivor_mask_360_360_pair_num_from_inliers_150_150.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_150_150_from_allpairs))/np.sum(good_pair_mask_150_150_from_allpairs))

            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_5_5.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_5_5_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_15_15.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_15_15_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_25_25.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_25_25_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_35_35.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_35_35_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_50_50.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_50_50_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_75_75.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_75_75_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_100_100.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_100_100_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))
            inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_150_150.append(np.sum(np.array(survivor_mask_360_360_from_allpairs&good_pair_mask_150_150_from_allpairs))/np.sum(survivor_mask_360_360_from_allpairs))



        plt.figure()
        plt.errorbar(survivor_ratio_threshold_list, np.array(rot_accuracy_mean), np.array(rot_accuracy_std), c='b', markersize=4, linewidth=1.5, linestyle='-', marker='^', label='mean rotation error with standard deviation')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(rot_accuracy_median), markersize=4, linewidth=1.5, c='r', marker="X", label='median rotation error')#, c=colors, alpha=0.5)
        plt.ylabel('avg./median rotation error in degrees with std', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"FinalRotErrorPlot_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.errorbar(survivor_ratio_threshold_list, np.array(trans_accuracy_mean), np.array(trans_accuracy_std), c='b', markersize=4, linewidth=1.5, linestyle='-', marker='^', label='mean translation error with standard deviation')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(trans_accuracy_median), markersize=4, linewidth=1.5, c='r', marker="X", label='median translation error')#, c=colors, alpha=0.5)
        plt.ylabel('avg./median translation error in degrees with std', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"FinalTransErrorPlot_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.errorbar(survivor_ratio_threshold_list, np.array(rot_accuracy_mean), np.array(rot_accuracy_std), markersize=4, linewidth=1.5, linestyle='-', marker='^')#, c=colors, alpha=0.5)
        plt.ylabel('avg. rotation error in degrees with std', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"meanRotErrorPlot_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.errorbar(survivor_ratio_threshold_list, np.array(trans_accuracy_mean), np.array(trans_accuracy_std), markersize=4, linewidth=1.5, linestyle='-', marker='^')#, c=colors, alpha=0.5)
        plt.ylabel('avg. translation error in degrees with std', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"meanTransErrorPlot_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.errorbar(survivor_ratio_threshold_list, np.array(rot_accuracy_median), np.array(rot_accuracy_std), markersize=4, linewidth=1.5, linestyle='-', marker='^')#, c=colors, alpha=0.5)
        plt.ylabel('median rotation error in degrees with std', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"medianRotErrorPlot_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.errorbar(survivor_ratio_threshold_list, np.array(trans_accuracy_median), np.array(trans_accuracy_std), markersize=4, linewidth=1.5, linestyle='-', marker='^')#, c=colors, alpha=0.5)
        plt.ylabel('median translation error in degrees with std', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"medianTransErrorPlot_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_5_5), markersize=4, linewidth=1.5, c='b', marker="X", label='inliers: R_error<=5, t_error<=5')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_15_15), markersize=4, linewidth=1.5, c='g', marker="*", label='inliers: R_error<=15, t_error<=15')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='r', marker="o", label='inliers: R_error<=25, t_error<=25')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_35_35), markersize=4, linewidth=1.5, c='c', marker="D", label='inliers: R_error<=35, t_error<=35')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_50_50), markersize=4, linewidth=1.5, c='m', marker="X", label='inliers: R_error<=50, t_error<=50')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_75_75), markersize=4, linewidth=1.5, c='y', marker="*", label='inliers: R_error<=75, t_error<=75')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_100_100), markersize=4, linewidth=1.5, c='k', marker="o", label='inliers: R_error<=100, t_error<=100')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_150_150), markersize=4, linewidth=1.5, c='#0F5684', marker="D", label='inliers: R_error<=150, t_error<=150')#, c=colors, alpha=0.5)
        plt.ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        #plt.title("varying accuracy: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"varing_accuracy_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        plt.figure()
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_5_5), markersize=4, linewidth=1.5, c='b', marker="X", label='inliers: R_error<=5, t_error<=5')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_15_15), markersize=4, linewidth=1.5, c='g', marker="*", label='inliers: R_error<=15, t_error<=15')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='r', marker="o", label='inliers: R_error<=25, t_error<=25')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_35_35), markersize=4, linewidth=1.5, c='c', marker="D", label='inliers: R_error<=35, t_error<=35')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_50_50), markersize=4, linewidth=1.5, c='m', marker="X", label='inliers: R_error<=50, t_error<=50')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_75_75), markersize=4, linewidth=1.5, c='y', marker="*", label='inliers: R_error<=75, t_error<=75')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_100_100), markersize=4, linewidth=1.5, c='k', marker="o", label='inliers: R_error<=100, t_error<=100')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_150_150), markersize=4, linewidth=1.5, c='#0F5684', marker="D", label='inliers: R_error<=150, t_error<=150')#, c=colors, alpha=0.5)
        plt.ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        #plt.title("varying accuracy: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"varing_accuracy_360_360_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        #########################################
        # plt.figure()
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_5_5), markersize=4, linewidth=1.5, c='b', marker="X", label='inliers: R_error<=5, t_error<=5')#, c=colors, alpha=0.5)
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_15_15), markersize=4, linewidth=1.5, c='g', marker="*", label='inliers: R_error<=15, t_error<=15')#, c=colors, alpha=0.5)
        # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='r', marker="o", label='inliers: R_error<=25, t_error<=25')#, c=colors, alpha=0.5)
        # plt.plot(survivor_ratio_threshold_list, np.array(inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='b', marker="*", label='inliers: R_error<=25, t_error<=25')#, c=colors, alpha=0.5)
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_35_35), markersize=4, linewidth=1.5, c='c', marker="D", label='inliers: R_error<=35, t_error<=35')#, c=colors, alpha=0.5)
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_50_50), markersize=4, linewidth=1.5, c='m', marker="X", label='inliers: R_error<=50, t_error<=50')#, c=colors, alpha=0.5)
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_75_75), markersize=4, linewidth=1.5, c='y', marker="*", label='inliers: R_error<=75, t_error<=75')#, c=colors, alpha=0.5)
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_100_100), markersize=4, linewidth=1.5, c='k', marker="o", label='inliers: R_error<=100, t_error<=100')#, c=colors, alpha=0.5)
        # # plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_150_150), markersize=4, linewidth=1.5, c='#0F5684', marker="D", label='inliers: R_error<=150, t_error<=150')#, c=colors, alpha=0.5)
        # plt.ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs');
        # plt.xlabel('survivor_ratio_threshold_list');
        # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='r', marker="o", label='inlier_survivor_pair_ratio_of_all_inlier_pairs')#, c=colors, alpha=0.5)
        ax2.plot(survivor_ratio_threshold_list, np.array(inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='b', marker="*", label='inlier_survivor_pair_ratio_of_all_survivor_pairs')#, c=colors, alpha=0.5)

        ax1.set_xlabel('survivor_ratio_threshold_list')
        ax1.set_ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', color='r', fontsize=12)
        ax2.set_ylabel('inlier_survivor_pair_ratio_of_all_survivor_pairs', color='b', fontsize=12)
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

        plt.title('inlier criteria: R_error<=25 and t_error<=25')
        #plt.title("varying accuracy: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"doubleplot_25_varing_accuracy_360_360_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_5_5), markersize=4, linewidth=1.5, c='r', marker="o", label='inlier_survivor_pair_ratio_of_all_inlier_pairs')#, c=colors, alpha=0.5)
        ax2.plot(survivor_ratio_threshold_list, np.array(inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_5_5), markersize=4, linewidth=1.5, c='b', marker="*", label='inlier_survivor_pair_ratio_of_all_survivor_pairs')#, c=colors, alpha=0.5)

        ax1.set_xlabel('survivor_ratio_threshold_list')
        ax1.set_ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', color='r', fontsize=12)
        ax2.set_ylabel('inlier_survivor_pair_ratio_of_all_survivor_pairs', color='b', fontsize=12)
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

        plt.title('inlier criteria: R_error<=5 and t_error<=5')
        #plt.title("varying accuracy: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"doubleplot_5_varing_accuracy_360_360_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_50_50), markersize=4, linewidth=1.5, c='r', marker="o", label='inlier_survivor_pair_ratio_of_all_inlier_pairs')#, c=colors, alpha=0.5)
        ax2.plot(survivor_ratio_threshold_list, np.array(inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_50_50), markersize=4, linewidth=1.5, c='b', marker="*", label='inlier_survivor_pair_ratio_of_all_survivor_pairs')#, c=colors, alpha=0.5)

        ax1.set_xlabel('survivor_ratio_threshold_list')
        ax1.set_ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', color='r', fontsize=12)
        ax2.set_ylabel('inlier_survivor_pair_ratio_of_all_survivor_pairs', color='b', fontsize=12)
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

        plt.title('inlier criteria: R_error<=50 and t_error<=50')
        #plt.title("varying accuracy: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"doubleplot_50_varing_accuracy_360_360_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(survivor_ratio_threshold_list, np.array(survivor_mask_360_360_pair_num_from_inliers_75_75), markersize=4, linewidth=1.5, c='r', marker="o", label='inlier_survivor_pair_ratio_of_all_inlier_pairs')#, c=colors, alpha=0.5)
        ax2.plot(survivor_ratio_threshold_list, np.array(inlierRatioFromAllSurvivors_survivor_mask_360_360_pair_num_from_inliers_75_75), markersize=4, linewidth=1.5, c='b', marker="*", label='inlier_survivor_pair_ratio_of_all_survivor_pairs')#, c=colors, alpha=0.5)

        ax1.set_xlabel('survivor_ratio_threshold_list')
        ax1.set_ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', color='r', fontsize=12)
        ax2.set_ylabel('inlier_survivor_pair_ratio_of_all_survivor_pairs', color='b', fontsize=12)
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

        plt.title('inlier criteria: R_error<=75 and t_error<=75')
        #plt.title("varying accuracy: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"doubleplot_75_varing_accuracy_360_360_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        #########################################

        plt.figure()
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_5_5_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='b', marker="X", label='inliers: R_consistency_error<=5, t_consistency_error<=5')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_15_15_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='g', marker="*", label='inliers: R_consistency_error<=15, t_consistency_error<=15')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_25_25_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='r', marker="o", label='inliers: R_consistency_error<=25, t_consistency_error<=25')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_35_35_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='c', marker="D", label='inliers: R_consistency_error<=35, t_consistency_error<=35')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_50_50_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='m', marker="X", label='inliers: R_consistency_error<=50, t_consistency_error<=50')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_75_75_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='y', marker="*", label='inliers: R_consistency_error<=75, t_consistency_error<=75')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_100_100_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='k', marker="o", label='inliers: R_consistency_error<=100, t_consistency_error<=100')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, np.array(survivor_mask_150_150_pair_num_from_inliers_25_25), markersize=4, linewidth=1.5, c='#0F5684', marker="D", label='inliers: R_consistency_error<=150, t_consistency_error<=150')#, c=colors, alpha=0.5)
        plt.ylabel('inlier_survivor_pair_ratio_of_all_inlier_pairs', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        #plt.title("varying consistency: inlier_survivor_pair_ratio_of_all_inlier_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"varing_consistency_max_sq_pixel_err_%s_goodpair_survivor_percentage_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        survivor_pair_num_list_npArr = np.array(survivor_pair_num_list)
        total_pair_num = survivor_pair_num_list_npArr[0]
        survivor_pair_num_list_npArr = survivor_pair_num_list_npArr/total_pair_num
        plt.figure()
        plt.plot(survivor_ratio_threshold_list, survivor_pair_num_list_npArr, markersize=4)#, c=colors, alpha=0.5)
        plt.ylabel('survivor_pair_num_list', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        #plt.title("inlier_survivor_pair_ratio_list vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"max_sq_pixel_err_%s_inlier_survivor_pair_ratio_list_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        update1_survivor_pair_num_list_npArr = np.array(update1_survivor_pair_num_list)
        update1_survivor_pair_num_list_npArr = update1_survivor_pair_num_list_npArr/total_pair_num
        plt.figure()
        plt.plot(survivor_ratio_threshold_list, update1_survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='b', marker="s", label='inliers_from_rotation')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='r', marker="o", label='inliers_from_leftright')#, c=colors, alpha=0.5)
        # ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
        # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
        plt.ylabel('survivor_pair_percentage_of_all_pairs', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        #plt.title("survivor_pair_percentage_of_all_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"max_sq_pixel_err_%s_survivor_pair_percentage_of_all_pairs_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        update2_survivor_pair_num_list_npArr = np.array(update2_survivor_pair_num_list)
        update2_survivor_pair_num_list_npArr = update2_survivor_pair_num_list_npArr/total_pair_num
        plt.figure()
        plt.plot(survivor_ratio_threshold_list, update2_survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='b', marker="s", label='inliers_from_rotation_and_leftright')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, update1_survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='g', marker="*", label='inliers_from_rotation')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='r', marker="o", label='inliers_from_leftright')#, c=colors, alpha=0.5)
        # ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
        # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
        plt.ylabel('survivor_pair_percentage_of_all_pairs', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        #plt.title("survivor_pair_percentage_of_all_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"max_sq_pixel_err_%s_bothInliers_survivor_pair_percentage_of_all_pairs_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()

        update5_survivor_pair_num_list_npArr = np.array(update5_survivor_pair_num_list)
        update5_survivor_pair_num_list_npArr = update5_survivor_pair_num_list_npArr/total_pair_num
        update3_survivor_pair_num_list_npArr = np.array(update3_survivor_pair_num_list)
        update3_survivor_pair_num_list_npArr = update3_survivor_pair_num_list_npArr/total_pair_num
        update4_survivor_pair_num_list_npArr = np.array(update4_survivor_pair_num_list)
        update4_survivor_pair_num_list_npArr = update4_survivor_pair_num_list_npArr/total_pair_num
        plt.figure()
        plt.plot(survivor_ratio_threshold_list, update4_survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='b', marker="s", label='inliers_from_pose_and_leftright')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, update5_survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='g', marker="*", label='inliers_from_pose')#, c=colors, alpha=0.5)
        plt.plot(survivor_ratio_threshold_list, update3_survivor_pair_num_list_npArr, markersize=4, linewidth=1.5, c='r', marker="o", label='inliers_from_leftright')#, c=colors, alpha=0.5)
        plt.ylabel('survivor_pair_percentage_of_all_pairs', fontsize=12);
        plt.xlabel('survivor_ratio_threshold_list', fontsize=12);
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        #plt.title("survivor_pair_percentage_of_all_pairs vs survivor_ratio_threshold_list")
        plt.savefig(os.path.join(args.output_path, 'thesis_report',"max_sq_pixel_err_%s_bothOFandPoseInliers_survivor_pair_percentage_of_all_pairs_vs_survivor_ratio_threshold_list.png"%(max_sq_pixel_err)))
        # plt.show()



if __name__ == "__main__":
    main()
