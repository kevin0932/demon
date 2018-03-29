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
# examples_dir = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(examples_dir, '..', 'lmbspecialops', 'python'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
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
        if diff[0] * diff[0] + diff[1] * diff[1] <= max_reproj_error:
            matches.append((match[1], match[0]))
            float32_coords_1.append( coord_12_1 )
            float32_coords_2.append( coord_12_2 )
            # matches.append((match[1], match[0]))
            # float32_coords_1.append( coord_21_1 )
            # float32_coords_2.append( coord_21_2 )

    return np.array(matches, dtype=np.uint32), np.array(float32_coords_1, dtype=np.float32), np.array(float32_coords_2, dtype=np.float32)

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

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

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

    with open(os.path.join(args.output_path, 'test_keypoints.txt'), "w") as fid:
        cursor.execute("SELECT image_id, data FROM keypoints WHERE rows>=?;",
                       (args.min_num_features,))
        for row in cursor:
            image_id = row[0]
            features = np.fromstring(row[1], dtype=np.float32).reshape(-1, 6)
            featuresMat = features[:,0:2]
            image_name = images_id_to_name[image_id]
            features_list[image_name] = featuresMat
            # # featuresMat = np.concatenate((featuresMat, np.array([], dtype=np.float32))), axis=0)
            # fid.write("%s %d\n" % (image_name, features.shape[0]))
            # for i in range(features.shape[0]):
            #     print(features[i])
            #     fid.write("%d %d %d %d %d %d\n" % tuple(features[i]))


    cursor.close()
    connection.close()

    print("features_list.keys() = ", features_list.keys())
    print("len(features_list.keys()) = ", len(features_list.keys()))

    good_pairs = []
    with open((args.input_good_pairs_path), "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                good_pair = (elems[0])
                good_pairs.append(good_pair)
    print("good_pairs num = ", good_pairs)

    data = h5py.File(args.demon_path)
    OF_scale_factor = args.OF_scale_factor

    image_pairs = set()
    valid_pair_num = 0
    output_file_path = os.path.join(args.output_path, 'CrossCheckSurvivor_full_quantization_map_OFscale_'+str(OF_scale_factor)+'_err_'+str(int(args.max_pixel_error*1000))+'_survivorRatio_'+str(int(args.survivor_ratio*1000))+'.txt')
    # output_file_path_clean = os.path.join(args.output_path, 'Clean_CrossCheckSurvivor_full_quantization_map_OFscale_'+str(OF_scale_factor)+'_err_'+str(int(args.max_pixel_error*1000))+'_survivorRatio_'+str(int(args.survivor_ratio*1000))+'.txt')
    output_OFfile_path = os.path.join(args.output_path, 'CrossCheckSurvivor_OpticalFlow_OFscale_'+str(OF_scale_factor)+'_err_'+str(int(args.max_pixel_error*1000))+'_survivorRatio_'+str(int(args.survivor_ratio*1000))+'.txt')
    with open(output_OFfile_path, "w") as fOFid:
        # with open(output_file_path_clean, "w") as f2id:
        if True:
            with open(output_file_path, "w") as fid:
                for image_name1 in features_list.keys():
                    for image_name2 in features_list.keys():
                        if image_name1 == image_name2:
                            continue

                        image_pair12 = image_name1+'---'+image_name2
                        if image_pair12 not in data.keys():
                            continue
                        print("Processing", image_pair12, "; img1 has ", features_list[image_name1].shape[0], " features", "; img2 has ", features_list[image_name2].shape[0], " features")
                        if image_pair12 in image_pairs:
                            continue

                        #fid.write("%s %s\n" % (image_name1, image_name2))

                        # image_pair21 = "{}---{}".format(image_name2, image_name1)
                        # image_pairs.add(image_pair12)
                        # image_pairs.add(image_pair21)
                        # if image_pair21 not in data.keys():
                        #     continue
                        #
                        # # if image_pair12 not in good_pairs:
                        # if image_pair12 not in good_pairs or image_pair21 not in good_pairs :
                        #     print("skip the image pair because it is not a good pair by visual inspection!")
                        #     continue


                        flow12 = data[image_pair12]["flow"]
                        # flow12 = np.transpose(flow12, [2, 0, 1])
                        print(flow12.shape)

                        # flow21 = data[image_pair21]["flow"]
                        # # flow21 = np.transpose(flow21, [2, 0, 1])
                        # # print(flow21.shape)

                        # ### add code to upsample the predicted optical-flow
                        # if OF_scale_factor > 1:
                        #     flow12_upsampled = upsample_optical_flow(flow12, OF_scale_factor=OF_scale_factor)
                        #     flow21_upsampled = upsample_optical_flow(flow21, OF_scale_factor=OF_scale_factor)
                        #     flow12 = flow12_upsampled
                        #     flow21 = flow21_upsampled
                        #     print("updampled flow12.shape = ", flow12.shape)

                        matches12, coords121, coords122 = flow_to_matches_float32Pixels(flow12)
                        if  matches12.size/2 <= 0:
                            continue

                        # matches21, coords211, coords212 = flow_to_matches_float32Pixels(flow21)
                        #
                        # print("  => Found", matches12.size/2, "<->", matches21.size/2, "matches")
                        # if  matches12.size/2 <= 0 or matches21.size/2 <= 0:
                        #     continue
                        #
                        # matches, coords_12_1, coords_12_2 = cross_check_matches_float32Pixel(matches12, coords121, coords122,
                        #                               matches21, coords211, coords212,
                        #                               # args.max_pixel_error*OF_scale_factor)
                        #                               args.max_pixel_error)
                        #                               # max_reproj_error)
                        # print("matches.shape = ", matches.shape, "; ", "coords_12_1.shape = ", coords_12_1.shape, "coords_12_2.shape = ", coords_12_2.shape)
                        #
                        # if matches.size == 0:
                        #     continue
                        # print("  => Cross-checked", matches.shape[0], "matches")

                        # if matches.shape[0] / matches12.shape[0] >= args.survivor_ratio:
                        if True:
                            valid_pair_num += 1
                            # print("cross-check-survivor-ratio = ", matches.shape[0] / matches12.shape[0])

                            # ### add code to upsample the predicted optical-flow
                            # if OF_scale_factor > 1:
                            #     flow12_upsampled = upsample_optical_flow(flow12, OF_scale_factor=OF_scale_factor)
                            #     # flow21_upsampled = upsample_optical_flow(flow21, OF_scale_factor=OF_scale_factor)
                            #     flow12 = flow12_upsampled
                            #     # flow21 = flow21_upsampled
                            #     print("updampled flow12.shape = ", flow12.shape)
                            #
                            # matches12, coords121, coords122 = flow_to_matches_float32Pixels(flow12)

                            fid.write("%s %s\n" % (image_name1, image_name2))
                            # guide_mapping_dict = {}
                            for i in range(matches12.shape[0]):
                                # guide_mapping_dict[matches12[i,0]] = matches12[i,1]
                                # fid.write("%s %s\n" % (matches12[i,0], matches12[i,1]))
                                ### record match id1, id2, flowx, flowy, pt1_x, pt1_y, pt2_x, pt2_y
                                fid.write("%s %s %s %s %s %s %s %s\n" % (matches12[i,0], matches12[i,1], (coords122[i,0]-coords121[i,0]), (coords122[i,1]-coords121[i,1]), coords121[i,0], coords121[i,1], coords122[i,0], coords122[i,1]))

                            fid.write("\n") # empty line is added for colmap custom_match format

                            # f2id.write("%s %s\n" % (image_name1, image_name2))
                            # # guide_mapping_dict2 = {}
                            # for i in range(matches.shape[0]):
                            #     # guide_mapping_dict2[matches12[i,0]] = matches12[i,1]
                            #     ### record match id1, id2, flowx, flowy, pt1_x, pt1_y, pt2_x, pt2_y
                            #     f2id.write("%s %s %s %s %s %s %s %s\n" % (matches[i,0], matches[i,1], (coords_12_2[i,0]-coords_12_1[i,0]), (coords_12_2[i,1]-coords_12_1[i,1]), coords_12_1[i,0], coords_12_1[i,1], coords_12_2[i,0], coords_12_2[i,1]))
                            #
                            # f2id.write("\n") # empty line is added for colmap custom_match format

                            fOFid.write("%s %s\n" % (image_name1, image_name2))
                            for y in range(flow12.shape[1]):
                                for x in range(flow12.shape[2]):
                                    # fOFid.write("%s %s %s %s\n" % (flow12[0, y, x], flow12[1, y, x], flow21[0, y, x], flow21[1, y, x]))
                                    fOFid.write("%s %s\n" % (flow12[0, y, x], flow12[1, y, x]))
                            fOFid.write("\n") # empty line is added for colmap custom_match format

                # return
    ### copy the saved quantization map to another file with valid_pair_num and delete the original one
    print("valid_pair_num = ", valid_pair_num)
    final_output_file_path = os.path.join(args.output_path, 'CrossCheckSurvivor_full_quantization_map_OFscale_'+str(OF_scale_factor)+'_err_'+str(int(args.max_pixel_error*1000))+'_survivorRatio_'+str(int(args.survivor_ratio*1000))+'_validPairNum_'+str(int(valid_pair_num))+'.txt')
    shutil.copy(output_file_path, final_output_file_path)
    os.remove(output_file_path)
    # final_output_file_path_clean = os.path.join(args.output_path, 'Clean_CrossCheckSurvivor_full_quantization_map_OFscale_'+str(OF_scale_factor)+'_err_'+str(int(args.max_pixel_error*1000))+'_survivorRatio_'+str(int(args.survivor_ratio*1000))+'_validPairNum_'+str(int(valid_pair_num))+'.txt')
    # shutil.copy(output_file_path_clean, final_output_file_path_clean)
    # os.remove(output_file_path_clean)
    final_output_OFfile_path = os.path.join(args.output_path, 'CrossCheckSurvivor_OpticalFlow_OFscale_'+str(OF_scale_factor)+'_err_'+str(int(args.max_pixel_error*1000))+'_survivorRatio_'+str(int(args.survivor_ratio*1000))+'_validPairNum_'+str(int(valid_pair_num))+'.txt')
    shutil.copy(output_OFfile_path, final_output_OFfile_path)
    os.remove(output_OFfile_path)

if __name__ == "__main__":
    main()
