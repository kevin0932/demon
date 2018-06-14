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

print(sys.path)
from depthmotionnet.vis import *
from depthmotionnet.networks_original import *

###### think about how to add lmbspecialops and depthmotionnet into anaconda env path!
# sys.path.append(r'/home/kevin/anaconda_tensorflow_demon_ws/demon/lmbspecialops/python/')
# sys.path.append(r'/home/kevin/anaconda_tensorflow_demon_ws/demon/python/depthmotionnet/')

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images_dir_path", required=True)
    parser.add_argument("--output_h5_dir_path", required=True)
    parser.add_argument("--freiburg_prediction_h5", required=True)
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

def main():
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
    output_dir = args.output_h5_dir_path

    Freiburg_Data = h5py.File(args.freiburg_prediction_h5)
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
        os.stat(os.path.join(output_dir, "rawdepthmap"))
    except:
        os.mkdir(os.path.join(output_dir, "rawdepthmap"))
    try:
        os.stat(os.path.join(output_dir, "graydepthmap"))
    except:
        os.mkdir(os.path.join(output_dir, "graydepthmap"))

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

    # # Iterate over working directory
    # for file1 in os.listdir(input_dir):
    #     for file2 in os.listdir(input_dir):
    # Iterate over the same pairs as in the data provided by Freiburg
    for image_pair12 in Freiburg_Data.keys():
        print("Processing", image_pair12)


        file1, file2 = image_pair12.split("---")
        image_pair21 = "{}---{}".format(file2, file1)

        if file1 not in os.listdir(input_dir):
            continue
        if file2 not in os.listdir(input_dir):
            continue

        file1_path = os.path.join(input_dir, file1)
        file1_name, file1_ext = os.path.splitext(file1_path)
        file2_path = os.path.join(input_dir, file2)
        file2_name, file2_ext = os.path.splitext(file2_path)
        # Check if file is an image file
        if file1_ext not in image_exts:
            print("Skipping " + file1 + " (not an image file)")
            continue
        if file2_ext not in image_exts:
            print("Skipping " + file2 + " (not an image file)")
            continue
        if file1_name == file2_name:
            print("Skipping identical files: " + file1 + "---" + file2 + "...")
            continue

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
        # print(flow2.shape)
        # flow5 = result['predict_flow5'].squeeze()
        # print(flow5.shape)

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
        h5file.create_dataset((file1 + "---" + file2 + "/rotation_matrix"), data=rotation_matrix)
        h5file.create_dataset((file1 + "---" + file2 + "/translation"), data=translation)
        h5file.create_dataset((file1 + "---" + file2 + "/depth"), data=depth_48by64)
        h5file.create_dataset((file1 + "---" + file2 + "/depth_upsampled"), data=depth_upsampled)
        h5file.create_dataset((file1 + "---" + file2 + "/flow"), data=flow2)
    h5file.close()   # be CERTAIN to close the file
    print("HDF5 file is written successfully:", output_h5_filepath)

if __name__ == "__main__":
    main()
