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

# examples_dir = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(examples_dir, '..', 'lmbspecialops', 'python'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--demon_path", required=True)
    parser.add_argument("--scale_factor", type=int, default=24)
    parser.add_argument("--min_num_features", type=int, default=1)
    parser.add_argument("--ratio_threshold", type=float, default=0.75)
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

    # with open(os.path.join(args.output_path), "w") as fid:
    #     # featuresMat = np.zeros([1,2])
    #     cursor.execute("SELECT image_id, data FROM keypoints WHERE rows>=?;",
    #                    (args.min_num_features,))
    #     for row in cursor:
    #         image_id = row[0]
    #         features = np.fromstring(row[1], dtype=np.float32).reshape(-1, 6)
    #         featuresMat = features[:,0:2]
    #         image_name = images_id_to_name[image_id]
    #         features_list[image_name] = featuresMat
    #         # featuresMat = np.concatenate((featuresMat, np.array([], dtype=np.float32))), axis=0)
    #         fid.write("%s %d\n" % (image_name, features.shape[0]))
    #         for i in range(features.shape[0]):
    #             fid.write("%d %d %d %d %d %d\n" % tuple(features[i]))

    with open(os.path.join(args.output_path, 'test_keypoints.txt'), "w") as fid:
        cursor.execute("SELECT image_id, data FROM keypoints WHERE rows>=?;",
                       (args.min_num_features,))
        for row in cursor:
            image_id = row[0]
            features = np.fromstring(row[1], dtype=np.float32).reshape(-1, 6)
            featuresMat = features[:,0:2]
            image_name = images_id_to_name[image_id]
            features_list[image_name] = featuresMat
            # featuresMat = np.concatenate((featuresMat, np.array([], dtype=np.float32))), axis=0)
            fid.write("%s %d\n" % (image_name, features.shape[0]))
            for i in range(features.shape[0]):
                fid.write("%d %d %d %d %d %d\n" % tuple(features[i]))

    with open(os.path.join(args.output_path, 'test_descriptors.txt'), "w") as fid:
        cursor.execute("SELECT image_id, data FROM descriptors WHERE rows>=?;",
                           (args.min_num_features,))
        for row in cursor:
            image_id = row[0]
            # print("row[1] = ", np.fromstring(row[1],dtype=np.uint8))
            print("image_id = ", image_id)
            descriptors = np.fromstring(row[1],dtype=np.uint8).reshape(-1, 128)
            # print("descriptors.shape = ", descriptors.shape)
            image_name = images_id_to_name[image_id]
            descriptors_list[image_name] = descriptors
            fid.write("%s %d\n" % (image_name, descriptors.shape[0]))
            for i in range(descriptors.shape[0]):
                for j in range(128):
                    fid.write("%d" % (descriptors[i,j]))
                if j < 127:
                    fid.write(" ")

    cursor.close()
    connection.close()

    print("features_list.keys() = ", features_list.keys())
    print("len(features_list.keys()) = ", len(features_list.keys()))
    print("features_list['P1180180.JPG'].shape = ", features_list['P1180180.JPG'].shape)

    for image_name in features_list.keys():
        features = features_list[image_name]
        # print("features_list[image_name].shape = ", features_list[image_name].shape)
        tmp_x = np.array(features[:,0] / args.scale_factor).astype(np.int)
        # print("tmp_x.shape = ", tmp_x.shape)
        tmp_y = np.array(features[:,1] / args.scale_factor).astype(np.int)
        # print("tmp_y.shape = ", tmp_y.shape)
        quantization_ids = (tmp_x) + 64 * (tmp_y)
        quantization_ids = quantization_ids.astype(np.int)
        # print("quantization_ids = ", quantization_ids)
        # print("quantization_ids.shape = ", quantization_ids.shape)
        quantization_list[image_name] = quantization_ids

    data = h5py.File(args.demon_path)

    image_pairs = set()
    with open(os.path.join(args.output_path, 'match_guide.txt'), "w") as fid:
        for image_name1 in features_list.keys():
            for image_name2 in features_list.keys():
                if image_name1 == image_name2:
                    continue

                fid.write("%s %s\n" % (image_name1, image_name2))

                image_pair12 = image_name1+'---'+image_name2
                if image_pair12 not in data.keys():
                    continue
                print("Processing", image_pair12, "; img1 has ", features_list[image_name1].shape[0], " features", "; img2 has ", features_list[image_name2].shape[0], " features")
                if image_pair12 in image_pairs:
                    continue

                # image_pair21 = "{}---{}".format(image_name2, image_name1)
                image_pairs.add(image_pair12)
                # image_pairs.add(image_pair21)

                # if image_pair21 not in data:
                #     continue

                # img1PIL = view1Colmap.image
                # #img1PIL.save(os.path.join(small_undistorted_images_dir, image_name1))
                # img2PIL = view2Colmap.image
                # #img2PIL.save(os.path.join(small_undistorted_images_dir, image_name2))

                # image_pair12_rotmat = data[image_pair12]["rotation"].value
                # image_pair21_rotmat = data[image_pair21]["rotation"].value
                #
                # image_pair12_transVec = data[image_pair12]["translation"].value
                # image_pair21_transVec = data[image_pair21]["translation"].value
                #
                flow12 = data[image_pair12]["flow"]
                # flow12 = np.transpose(flow12, [2, 0, 1])
                print(flow12.shape)
                # flow21 = data[image_pair21]["flow"]
                # flow21 = np.transpose(flow21, [2, 0, 1])
                # print(flow21.shape)

                matches12, coords121, coords122 = flow_to_matches_float32Pixels(flow12)
                guide_mapping_dict = {}
                for i in range(matches12.shape[0]):
                    guide_mapping_dict[matches12[i,0]] = matches12[i,1]

                features1 = features_list[image_name1]
                features2 = features_list[image_name2]
                descriptors1 = descriptors_list[image_name1]
                descriptors2 = descriptors_list[image_name2]
                quantization_ids1 = quantization_list[image_name1]
                quantization_ids2 = quantization_list[image_name2]
                for id1 in range(features1.shape[0]):
                    print("image 1's feature ", id1)
                    search_space_quantization_id = guide_mapping_dict[quantization_ids1[id1]]
                    search_mask1d = (quantization_ids2==search_space_quantization_id)
                    if sum(search_mask1d)<=0:
                        continue
                    search_ids = quantization_ids2[search_mask1d]
                    queryFeat = descriptors1[id1, :]
                    queryFeat = np.reshape(queryFeat, [1, queryFeat.shape[0]])
                    candidateFeats = descriptors1[search_mask1d, :]
                    if sum(search_mask1d)==1:
                        candidateFeats = np.reshape(candidateFeats, [1, candidateFeats.shape[0]])

                    tmp_dists = distance.cdist(candidateFeats, queryFeat)
                    print("tmp_dists.shape = ", tmp_dists)
                    dist_results = tmp_dists.squeeze()
                    print("dist_results.shape = ", dist_results)
                    sorted_indices = np.argsort(dist_results)
                    if dist_results[sorted_indices[0]]/dist_results[sorted_indices[1]] <= args.ratio_threshold:
                        fid.write("%s %s\n" % (id1, search_ids[sorted_indices[0]]))

            return

if __name__ == "__main__":
    main()
