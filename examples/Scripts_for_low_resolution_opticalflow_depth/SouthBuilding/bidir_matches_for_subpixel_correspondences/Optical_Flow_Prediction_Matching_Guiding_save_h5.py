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
# import os
import sys
import colmap_utils as colmap
from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.helpers import angleaxis_to_rotation_matrix
import re
import six
import tensorflow as tf

# cur_file_dir = os.path.dirname(__file__)
# print("cur_file_dir = ", cur_file_dir)
# sys.path.insert(0, os.path.join(cur_file_dir, '../../../..', 'lmbspecialops', 'python'))
sys.path.insert(0, '/home/kevin/anaconda_tensorflow_demon_ws/demon/lmbspecialops/python')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--scale_factor", type=int, default=24)
    parser.add_argument("--min_num_features", type=int, default=1)
    args = parser.parse_args()
    return args

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
    data_format = get_tf_data_format()

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

    # with open(os.path.join(args.output_path, 'test_descriptors.txt'), "w") as fid:
    #     cursor.execute("SELECT image_id, data FROM descriptors WHERE rows>=?;",
    #                        (args.min_num_features,))
    #     for row in cursor:
    #         image_id = row[0]
    #         # print("row[1] = ", np.fromstring(row[1],dtype=np.uint8))
    #         print("image_id = ", image_id)
    #         descriptors = np.fromstring(row[1],dtype=np.uint8).reshape(-1, 128)
    #         print("descriptors.shape = ", descriptors.shape)
    #         image_name = images_id_to_name[image_id]
    #         descriptors_list[image_name] = descriptors
    #         fid.write("%s %d\n" % (image_name, descriptors.shape[0]))
    #         for i in range(descriptors.shape[0]):
    #             for j in range(128):
    #                 fid.write("%d" % (descriptors[i,j]))
    #             if j < 127:
    #                 fid.write(" ")

    cursor.close()
    connection.close()

    print("features_list.keys() = ", features_list.keys())
    print("len(features_list.keys()) = ", len(features_list.keys()))
    print("features_list['P1180180.JPG'].shape = ", features_list['P1180180.JPG'].shape)

    for image_name in features_list.keys():
        features = features_list[image_name]
        print("features_list[image_name].shape = ", features_list[image_name].shape)
        tmp_x = np.array(features[:,0] / args.scale_factor).astype(np.int)
        print("tmp_x.shape = ", tmp_x.shape)
        tmp_y = np.array(features[:,1] / args.scale_factor).astype(np.int)
        print("tmp_y.shape = ", tmp_y.shape)
        quantization_ids = (tmp_x) + 64 * (tmp_y)
        quantization_ids = quantization_ids.astype(np.int)
        print("quantization_ids = ", quantization_ids)
        print("quantization_ids.shape = ", quantization_ids.shape)
        quantization_list[image_name] = quantization_ids


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
    saver.restore(session,'/home/kevin/anaconda_tensorflow_demon_ws/demon/weights/demon_original')

    print("Write a NeXus HDF5 file")
    output_h5_filename = u"exhaustive_prediction_for_later_access.h5"
    timestamp = u"20107-11-22T15:17:04-0500"
    output_h5_filepath = os.path.join(args.output_path, output_h5_filename)
    h5file = h5py.File(output_h5_filepath, "w")
    # point to the default data to be plotted
    h5file.attrs[u'default']          = u'entry'
    # give the HDF5 root some more attributes
    h5file.attrs[u'file_name']        = output_h5_filename
    h5file.attrs[u'file_time']        = timestamp
    h5file.attrs[u'instrument']       = u'DeMoN'
    h5file.attrs[u'creator']          = u'Optical_Flow_Prediction_Matching_Guilding.py'
    h5file.attrs[u'NeXus_version']    = u'4.3.0'
    h5file.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
    h5file.attrs[u'h5py_version']     = six.u(h5py.version.version)

    image_pairs = set()
    cnt = 0
    with open(os.path.join(args.output_path, 'match_guide.txt'), "w") as fid:
        for image_name1 in features_list.keys():
            cnt += 1
            print("Processing", image_name1, "; ", cnt, "/", len(features_list.keys()))
            for image_name2 in features_list.keys():
                if image_name1 == image_name2:
                    continue

                fid.write("%s %s\n" % (image_name1, image_name2))

                image_pair12 = image_name1+'---'+image_name2
                # print("Processing", image_pair12)
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
                # flow12 = data[image_pair12]["flow"]
                # flow12 = np.transpose(flow12, [2, 0, 1])
                # print(flow12.shape)
                # flow21 = data[image_pair21]["flow"]
                # flow21 = np.transpose(flow21, [2, 0, 1])
                # print(flow21.shape)

                img1 = Image.open(os.path.join(args.image_path, image_name1))
                img2 = Image.open(os.path.join(args.image_path, image_name2))

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
                rotation = result['predict_rotation'].squeeze()
                rotation_matrix = angleaxis_to_rotation_matrix(rotation)
                translation = result['predict_translation'].squeeze()
                depth_48by64 = result['predict_depth2'].squeeze()
                flow2 = result['predict_flow2'].squeeze()
                flow2 = flow2.transpose([2, 0, 1])
                scale = result['predict_scale'].squeeze().astype(np.float32)

                result = refine_net.eval(input_data['image1'],result['predict_depth2'])
                depth_upsampled = result['predict_depth0'].squeeze()

                h5file.create_dataset((image_name1 + "---" + image_name2 + "/rotation_angleaxis"), data=rotation)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/rotation"), data=rotation_matrix)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/translation"), data=translation)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/depth"), data=depth_48by64)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/depth_upsampled"), data=depth_upsampled)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/flow"), data=flow2)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/scale"), data=scale)

                # matches12, coords121, coords122 = flow_to_matches_float32Pixels_withDepthFiltering(flow12, real_depth_map1)

    h5file.close()   # be CERTAIN to close the file
    print("HDF5 file is written successfully:", output_h5_filepath)


if __name__ == "__main__":
    main()
