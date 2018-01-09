import os
import argparse
import subprocess
import collections
import sqlite3
import h5py
import numpy as np
import math
import functools
import six

from pyquaternion import Quaternion
import nibabel.quaternions as nq

import PIL.Image
from matplotlib import pyplot as plt
#import os
import sys

import tensorflow as tf
from depthmotionnet.vis import *
# from depthmotionnet.vis import visualize_prediction
import pkg_resources

#import numpy as np
import scipy.ndimage.interpolation as interp
import scipy.misc as misc

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import vtk

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

#SIMPLE_RADIAL_CAMERA_MODEL = 2

Image = collections.namedtuple(
    "Image", ["id", "camera_id", "name", "qvec", "tvec", "rotmat", "angleaxis"])
# ImagePairGT = collections.namedtuple(
#     "ImagePairGT", ["id1", "id2", "qvec12", "tvec12", "camera_id1", "name1", "camera_id2", "name2"])

ImagePairGT = collections.namedtuple("ImagePairGT", ["id1", "id2", "qvec12", "tvec12", "camera_id1", "name1", "camera_id2", "name2", "rotmat12"])
RotationAngularErrorRecord = collections.namedtuple("RotationAngularErrorRecord", ["id1", "name1", "id2_minErr", "name2_minErr", "RSymAngularError", "tSymAngularError"])

def read_relative_poses_text(path):
    image_pair_gt = {}
    dummy_image_pair_id = 1
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id1 = int(elems[0])
                image_id2 = int(elems[1])
                qvec12 = np.array(tuple(map(float, elems[2:6])))
                tvec12 = np.array(tuple(map(float, elems[6:9])))
                camera_id1 = int(elems[9])
                image_name1 = elems[10]
                camera_id2 = int(elems[11])
                image_name2 = elems[12]
                rotmat_r1 = np.array(tuple(map(float, elems[13:16])))
                rotmat_r2 = np.array(tuple(map(float, elems[16:19])))
                rotmat_r3 = np.array(tuple(map(float, elems[19:22])))
                RelativeRotationMat = np.array([rotmat_r1, rotmat_r2, rotmat_r3])
                # print("RelativeRotationMat.shape = ", RelativeRotationMat.shape)
                image_pair_gt[dummy_image_pair_id] = ImagePairGT(id1=image_id1, id2=image_id2, qvec12=qvec12, tvec12=tvec12, camera_id1=camera_id1, name1=image_name1, camera_id2=camera_id2, name2=image_name2, rotmat12 = RelativeRotationMat)
                dummy_image_pair_id += 1
    return image_pair_gt

def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                camera_id = int(elems[1])
                image_name = elems[2]
                qvec = np.array(tuple(map(float, elems[3:7])))
                tvec = np.array(tuple(map(float, elems[7:10])))
                rotmat_row0 = np.array(tuple(map(float, elems[10:13])))
                rotmat_row1 = np.array(tuple(map(float, elems[13:16])))
                rotmat_row2 = np.array(tuple(map(float, elems[16:19])))
                rotmat = np.vstack( (rotmat_row0, rotmat_row1) )
                rotmat = np.vstack( (rotmat, rotmat_row2) )
                angleaxis = np.array(tuple(map(float, elems[19:22])))
                # print("rotmat.shape = ", rotmat.shape)
                images[image_id] = Image(id=image_id, camera_id=camera_id, name=image_name, qvec=qvec, tvec=tvec, rotmat=rotmat, angleaxis=angleaxis)
    return images

def read_images_colmap_format_text(path):
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    images = {}
    featureLine = False
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if featureLine == True:
                featureLine = False
                continue
            if len(line) > 0 and line[0] != "#" and featureLine != True:
                elems = line.split()
                image_id = int(elems[0])
                camera_id = int(elems[8])
                image_name = elems[9]
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                rotmat_row0 = np.array(tuple(map(float, elems[10:13])))
                rotmat_row1 = np.array(tuple(map(float, elems[13:16])))
                rotmat_row2 = np.array(tuple(map(float, elems[16:19])))
                rotmat = np.vstack( (rotmat_row0, rotmat_row1) )
                rotmat = np.vstack( (rotmat, rotmat_row2) )
                # angleaxis = np.array(tuple(map(float, elems[19:22])))
                angleaxis = np.array([0,0,0])
                # print("rotmat.shape = ", rotmat.shape)
                images[image_id] = Image(id=image_id, camera_id=camera_id, name=image_name, qvec=qvec, tvec=tvec, rotmat=rotmat, angleaxis=angleaxis)
                featureLine = True
    return images

# def read_relative_poses_text(path):
#     image_pair_gt = {}
#     dummy_image_pair_id = 1
#     with open(path, "r") as fid:
#         while True:
#             line = fid.readline()
#             if not line:
#                 break
#             line = line.strip()
#             if len(line) > 0 and line[0] != "#":
#                 elems = line.split()
#                 image_id1 = int(elems[0])
#                 image_id2 = int(elems[1])
#                 qvec12 = np.array(tuple(map(float, elems[2:6])))
#                 tvec12 = np.array(tuple(map(float, elems[6:9])))
#                 camera_id1 = int(elems[9])
#                 image_name1 = elems[10]
#                 camera_id2 = int(elems[11])
#                 image_name2 = elems[12]
#                 image_pair_gt[dummy_image_pair_id] = ImagePairGT(id1=image_id1, id2=image_id2, qvec12=qvec12, tvec12=tvec12, camera_id1=camera_id1, name1=image_name1, camera_id2=camera_id2, name2=image_name2)
#                 dummy_image_pair_id += 1
#     return image_pair_gt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_demon_path", required=True)
    # parser.add_argument("--output_h5_dir_path", required=True)
    parser.add_argument("--images_path", required=True)
    parser.add_argument("--image_scale", type=float, default=12)
    parser.add_argument("--focal_length", type=float, default=228.13688) # shall we use this one ? 2457.60 / 12 = 204.8
    args = parser.parse_args()
    return args


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def euler2angle_axis(z=0, y=0, x=0):
    ''' Return angle, axis corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs

    Examples
    --------
    >>> theta, vec = euler2angle_axis(0, 1.5, 0)
    >>> print(theta)
    1.5
    >>> np.allclose(vec, [0, 1, 0])
    True
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))


def rotmat_To_angleaxis(image_pair12_rotmat):
    eulerAnlges = mat2euler(image_pair12_rotmat)
    recov_angle_axis_result = euler2angle_axis(eulerAnlges[0], eulerAnlges[1], eulerAnlges[2])
    R_angleaxis = recov_angle_axis_result[0]*(recov_angle_axis_result[1])
    R_angleaxis = np.array(R_angleaxis, dtype=np.float32)
    return R_angleaxis

def TheiaClamp( f, a, b):
    return max(a, min(f, b))

def prepare_input_data(img1, img2, data_format):
    # also convert original image to compatible format
    # transform range from [0,255] to [-0.5,0.5]
    orig_img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    orig_img2_arr = np.array(img2).astype(np.float32)/255 -0.5

    if data_format == 'channels_first':
        orig_img1_arr = orig_img1_arr.transpose([2,0,1])
        orig_img2_arr = orig_img2_arr.transpose([2,0,1])
        orig_image_pair = np.concatenate((orig_img1_arr,orig_img2_arr), axis=0)
    else:
        orig_image_pair = np.concatenate((orig_img1_arr,orig_img2_arr),axis=-1)

    orig_result = {
        'image_pair': orig_image_pair[np.newaxis,:],
        'image1': orig_img1_arr[np.newaxis,:], # first image
        'image2': orig_img2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }

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

    return result, orig_result


def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])




# http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.184.3942&rep=rep1&type=pdf
def quaternion2RotMat(qw, qx, qy, qz):
    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz

    # invs (inverse square length) is only required if quaternion is not already normalised
    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw)*invs     # since sqw + sqx + sqy + sqz =1/invs*invs
    m11 = (-sqx + sqy - sqz + sqw)*invs
    m22 = (-sqx - sqy + sqz + sqw)*invs

    tmp1 = qx*qy;
    tmp2 = qz*qw;
    m10 = 2.0 * (tmp1 + tmp2)*invs
    m01 = 2.0 * (tmp1 - tmp2)*invs

    tmp1 = qx*qz
    tmp2 = qy*qw
    m20 = 2.0 * (tmp1 - tmp2)*invs
    m02 = 2.0 * (tmp1 + tmp2)*invs
    tmp1 = qy*qz
    tmp2 = qx*qw
    m21 = 2.0 * (tmp1 + tmp2)*invs ;
    m12 = 2.0 * (tmp1 - tmp2)*invs

    rotation_matrix = np.array( [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] )
    return rotation_matrix


def rotmat_To_angleaxis(image_pair12_rotmat):
    eulerAnlges = mat2euler(image_pair12_rotmat)
    recov_angle_axis_result = euler2angle_axis(eulerAnlges[0], eulerAnlges[1], eulerAnlges[2])
    R_angleaxis = recov_angle_axis_result[0]*(recov_angle_axis_result[1])
    R_angleaxis = np.array(R_angleaxis, dtype=np.float32)
    return R_angleaxis

def TheiaClamp( f, a, b):
    return max(a, min(f, b))

# https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud
def point_cloud(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    """
    # cx = 0.5
    # cy = 0.5
    # fx = 2457.60/3072
    # fy = 2457.60/2304

    cx = 1536/12.0
    cy = 1152/12.0
    fx = 2457.60/12.0
    fy = 2457.60/12.0

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))


def angleaxis_to_rotation_matrix(aa):
    """Converts the 3 element angle axis representation to a 3x3 rotation matrix

    aa: numpy.ndarray with 1 dimension and 3 elements

    Returns a 3x3 numpy.ndarray
    """
    angle = np.sqrt(aa.dot(aa))

    if angle > 1e-6:
        c = np.cos(angle);
        s = np.sin(angle);
        u = np.array([aa[0]/angle, aa[1]/angle, aa[2]/angle]);

        R = np.empty((3,3))
        R[0,0] = c+u[0]*u[0]*(1-c);      R[0,1] = u[0]*u[1]*(1-c)-u[2]*s; R[0,2] = u[0]*u[2]*(1-c)+u[1]*s;
        R[1,0] = u[1]*u[0]*(1-c)+u[2]*s; R[1,1] = c+u[1]*u[1]*(1-c);      R[1,2] = u[1]*u[2]*(1-c)-u[0]*s;
        R[2,0] = u[2]*u[0]*(1-c)-u[1]*s; R[2,1] = u[2]*u[1]*(1-c)+u[0]*s; R[2,2] = c+u[2]*u[2]*(1-c);
    else:
        R = np.eye(3)
    return R

def read_global_poses_theia_output(path, path_img_id_map):
    image_id_name_pair = {}
    with open(path_img_id_map, "r") as fid1:
        while True:
            line = fid1.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                image_name = (elems[1])
                print("image_id = ", image_id, "; image_name = ", image_name)
                image_id_name_pair[image_id] = image_name

    images = {}
    dummy_image_id = 1
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                camera_id = image_id
                image_name = image_id_name_pair[image_id]
                R_angleaxis = np.array(tuple(map(float, elems[1:4])), dtype=np.float64)
                tvec = np.array(tuple(map(float, elems[4:7])))
                #rotmat_To_angleaxis(image_pair12_rotmat)
                rotmat = angleaxis_to_rotation_matrix(R_angleaxis)
                images[image_id] = Image(id=image_id, camera_id=camera_id, name=image_name, qvec=np.array([1,0,0,0]), tvec=tvec, rotmat=rotmat, angleaxis=R_angleaxis)

                dummy_image_id += 1

    print("total pairs = ", dummy_image_id-1)
    return images

ImagePairTheia = collections.namedtuple("ImagePair", ["id1", "name1", "id2", "name2", "R_rotmat", "R_angleaxis", "t_vec"])

def read_relative_poses_theia_output(path, path_img_id_map):
    #122 122 124 124 -0.00737405 0.26678 -0.0574713 -0.798498 -0.0794296 -0.596734
    image_id_name_pair = {}
    with open(path_img_id_map, "r") as fid1:
        while True:
            line = fid1.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                image_name = (elems[1])
                print("image_id = ", image_id, "; image_name = ", image_name)
                image_id_name_pair[image_id] = image_name

    image_pair_gt = {}
    dummy_image_pair_id = 1
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id1 = int(elems[1])
                image_id2 = int(elems[3])
                R_angleaxis = np.array(tuple(map(float, elems[8:11])), dtype=np.float64)
                t_vec = np.array(tuple(map(float, elems[13:16])), dtype=np.float64)
                R_rotmat = np.array(tuple(map(float, elems[17:26])), dtype=np.float64)
                R_rotmat = np.reshape(R_rotmat, [3,3])
                image_pair_gt[dummy_image_pair_id] = ImagePairTheia(id1=image_id1, name1=image_id_name_pair[image_id1], id2=image_id2, name2=image_id_name_pair[image_id2], R_rotmat=R_rotmat, R_angleaxis=R_angleaxis, t_vec=t_vec)
                dummy_image_pair_id += 1

    print("total num of input pairs = ", dummy_image_pair_id-1)
    return image_pair_gt


def transform_pc_to_global_coordinate(points_N_3, MatT_4_4):
    tmp = np.empty((points_N_3.shape[0],points_N_3.shape[1]+1),dtype=points_N_3.dtype)
    tmp[:,0:3] = points_N_3
    tmp[:,3] = 1
    transformedPoints_4_N = np.dot(MatT_4_4, tmp.transpose())
    transformedPoints_N_3 = transformedPoints_4_N[0:3,:].transpose()
    return transformedPoints_N_3

def main():
    args = parse_args()

    if tf.test.is_gpu_available(True):
        data_format='channels_first'
    else: # running on cpu requires channels_last data format
        data_format='channels_last'

    images = dict()
    image_pairs = set()

    # # reading theia intermediate output relative poses from textfile
    # TheiaRtfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results/RelativePoses_after_step7_global_position_estimation.txt'
    # TheiaIDNamefilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results/viewid_imagename_pairs_file.txt'
    # TheiaRelativePosesGT = read_relative_poses_theia_output(TheiaRtfilepath,TheiaIDNamefilepath)
    # # reading theia intermediate output global poses from textfile
    # TheiaGlobalPosesfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results/after_step7_global_position_estimation.txt'
    # TheiaGlobalPosesGT = read_global_poses_theia_output(TheiaGlobalPosesfilepath,TheiaIDNamefilepath)

    # reading colmap output as ground truth from textfile
    # ColmapGTfilepath = '/home/kevin/JohannesCode/ws1/sparse/0/textfiles_final/images.txt'
    # ColmapGTfilepath = '/home/kevin/ThesisDATA/person-hall/sparse/images.txt'
    # ColmapGTfilepath = '/home/kevin/ThesisDATA/gerrard-hall/sparse/images.txt'
    # ColmapGTfilepath = '/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/dense/sparse/images.txt'
    ColmapGTfilepath = '/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/dense/sparse/images.txt'
    imagesGT = read_images_colmap_format_text(ColmapGTfilepath)
    # print("imagesGT = ", imagesGT)
    # the .h5 file contains the filtered DeMoN prediction so that only one pair is kept for each input view
    data = h5py.File(args.filtered_demon_path)

    viewNum = 300000 # set to value larger than the number of input views
    translation_scales = {}

    # for image_pair12 in data.keys():
    #     image_name1, image_name2 = image_pair12.split("---")
    #     for ids,val in TheiaRelativePosesGT.items():
    #         if val.name1 == image_name1 and val.name2 == image_name2:
    #             transScale = np.linalg.norm(val.t_vec)
    #             translation_scales[image_name1] = transScale
    #             print("translation_scales[image_name1] = ", transScale)
    #             # translation_scales[image_name2] = transScale
    #             # print("translation_scales[image_name2] = ", transScale)

    it = 0

    renderer = vtk.vtkRenderer()
    # renderer.SetBackground(0, 0, 0)
    appendFilterPC = vtk.vtkAppendPolyData()
    appendFilterModel = vtk.vtkAppendPolyData()

    for image_pair12 in data.keys():
        print("it = ", it)
        print("Processing", image_pair12)

        image_name1, image_name2 = image_pair12.split("---")
        image_pair21 = "{}---{}".format(image_name2, image_name1)
        # # if image_name1 == 'P1180141.JPG' or image_name1 == 'P1180142.JPG' or image_name1 == 'P1180143.JPG' or image_name1 == 'P1180144.JPG' or image_name1 == 'P1180145.JPG':
        # if image_name1 != 'P1180216.JPG':
        #     continue

        # if image_name1 not in translation_scales.keys():
        #     continue
        # if image_name2 not in translation_scales.keys():
        #     continue

        image_pairs.add(image_pair12)

        if it==0:
            init_image_name1 = image_name1
            init_image_name2 = image_name2

        pred_rotmat12 = data[image_pair12]["rotation"].value
        pred_rotmat12_angleaxis = rotmat_To_angleaxis(pred_rotmat12)
        # pred_rotmat12 = data[image_pair12]["rotation_matrix"].value
        pred_trans12 = data[image_pair12]['translation'].value
        pred_invDepth121 = data[image_pair12]['depth_upsampled'].value
        pred_scale = data[image_pair12]['scale'].value

        imagepath1 = os.path.join(args.images_path, image_name1)
        imagepath2 = os.path.join(args.images_path, image_name2)


        img1PIL = PIL.Image.open(imagepath1)
        img2PIL = PIL.Image.open(imagepath2)
        # print("data_format = ", data_format)
        input_data, orig_data = prepare_input_data(img1PIL,img2PIL,data_format)
        # print("max depth = ", np.max(1/pred_invDepth121))
        print("pred_trans12 = ", pred_trans12)

        # for ids,val in TheiaRelativePosesGT.items():
        #     # if val.name1 == init_image_name1 and val.name2 == init_image_name2:
        #     #     init_scale = np.linalg.norm(val.t_vec)
        #     if val.name1 == image_name1 and val.name2 == image_name2:
        #         transScale = np.linalg.norm(val.t_vec)
        #         # transScale = np.linalg.norm(val.t_vec) / init_scale
        #         # transScale = init_scale / np.linalg.norm(val.t_vec)
        #         # translation_scales[image_name1] = transScale
        #         print("transScale = ", transScale)
        #
        #
        # TheiaExtrinsics_4by4 = np.eye(4)
        # for ids,val in TheiaGlobalPosesGT.items():
        #     if val.name == image_name1:
        #         TheiaExtrinsics_4by4[0:3,0:3] = val.rotmat
        #         #TheiaExtrinsics_4by4[0:3,0:3] = val.rotmat.T
        #         TheiaExtrinsics_4by4[0:3,3] = -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        ColmapExtrinsics_4by4 = np.eye(4)
        ColmapExtrinsics2_4by4 = np.eye(4)
        for ids,val in imagesGT.items():
            if val.name == image_name1:
                #ColmapExtrinsics_R = val.rotmat
                #ColmapExtrinsics_t = val.tvec
                ColmapExtrinsics_4by4[0:3,0:3] = val.rotmat
                ColmapExtrinsics_4by4[0:3,3] = val.tvec
                # print("ColmapExtrinsics_4by4 = ", ColmapExtrinsics_4by4)
            if val.name == image_name2:
                #ColmapExtrinsics_R = val.rotmat
                #ColmapExtrinsics_t = val.tvec
                ColmapExtrinsics2_4by4[0:3,0:3] = val.rotmat
                ColmapExtrinsics2_4by4[0:3,3] = val.tvec
        transScale = np.linalg.norm(-np.dot(ColmapExtrinsics_4by4[0:3,0:3].T, ColmapExtrinsics_4by4[0:3,3]) + np.dot(ColmapExtrinsics2_4by4[0:3,0:3].T, ColmapExtrinsics2_4by4[0:3,3]))
        print("pred_scale = ", pred_scale, "; calculated_scale_from_globalSfM transScale = ", transScale)
        if it==0:
            scaleRecordMat = np.array([pred_scale, transScale])
            # scaleRecordMat = np.reshape(scaleRecordMat,[1,2])
        else:
            # scaleRecordMat = np.concatenate((scaleRecordMat, np.array([pred_scale, transScale])), axis=0)
            scaleRecordMat = np.vstack((scaleRecordMat, np.array([pred_scale, transScale])))
        print("scaleRecordMat.shape = ", scaleRecordMat.shape)
        # tmp_PointCloud = make_pointcloud_prediction_in_global_coordinate(
        #             inverse_depth=pred_invDepth121,
        #             intrinsics = np.array([2457.60/3072, 2457.60/2304, 0.5, 0.5]),#################################
        #             image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
        #             R1=TheiaExtrinsics_4by4[0:3,0:3],
        #             t1=TheiaExtrinsics_4by4[0:3,3],
        #             scale=transScale)
        #             # rotation=None,
        #             # translation=None)
        #             # rotation=rotmat_To_angleaxis(np.eye(3)),
        #             # translation=np.zeros(3))

        tmp_PointCloud = visualize_prediction(
        # tmp_PointCloud = make_pointcloud_prediction_in_global_coordinate(
                    inverse_depth=pred_invDepth121,
                    # inverse_depth=(1/result['predict_depth0']),
                    # intrinsics = np.array([2457.60/3072, 2457.60/2304, 0.5, 0.5]),#################################
                    # intrinsics = np.array([2457.60/256, 2457.60/192, 0.5, 0.5]),#################################
                    # intrinsics = np.array([2457.60/3099, 2457.60/2314, 0.5, 0.5]),#################################
                    # intrinsics = np.array([0.89115971*256/3072, 1.18821287*192/2304, 0.5, 0.5]),#################################
                    intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                    image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                    #R1=TheiaExtrinsics_4by4[0:3,0:3],
                    #t1=TheiaExtrinsics_4by4[0:3,3],
                    R1=ColmapExtrinsics_4by4[0:3,0:3],
                    t1=ColmapExtrinsics_4by4[0:3,3],
                    rotation=pred_rotmat12_angleaxis,
                    translation=pred_trans12,
                    scale=transScale)
                    # scale=pred_scale/transScale)

        # # ColmapExtrinsics_T = np.eye(4)
        # # for ids,val in imagesGT.items():
        # #     if val.name == image_name1:
        # #         #ColmapExtrinsics_R = val.rotmat
        # #         #ColmapExtrinsics_t = val.tvec
        # #         ColmapExtrinsics_T[0:3,0:3] = val.rotmat
        # #         ColmapExtrinsics_T[0:3,3] = val.tvec
        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'], ColmapExtrinsics_T.T)
        #
        # # if it==0:
        # #     init_translation_norm = 1
        # # for ids,val in TheiaRelativePosesGT.items():
        # #     if val.name1 == image_name1 and val.name2 == image_name2:
        # #         if it==0:
        # #             init_translation_norm = np.linalg.norm(val.t_vec)
        # #
        # #         transScale = np.linalg.norm(val.t_vec) / init_translation_norm
        # #         print("transScale = ", transScale)

        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'], TheiaExtrinsics_4by4.T)
        # # tmp_PointCloud['points'] = transScale * transform_pointcloud_points(tmp_PointCloud['points'], TheiaExtrinsics_4by4.T)
        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'], TheiaExtrinsics_4by4.T) / transScale
        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'], TheiaExtrinsics_4by4.T) * transScale
        #
        # # tmp_PointCloud['points'] = (tmp_PointCloud['points'] / translation_scales[image_name1])
        # tmp_PointCloud['points'] = (tmp_PointCloud['points'] * translation_scales[image_name1])
        #
        #
        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'] / translation_scales[image_name1], TheiaExtrinsics_4by4.T)
        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'] * translation_scales[image_name1], TheiaExtrinsics_4by4.T)
        # # tmp_PointCloud['points'] = transform_pointcloud_points(tmp_PointCloud['points'], TheiaExtrinsics_4by4.T)
        # print("tmp_PointCloud['points'] = ", tmp_PointCloud['points'])
        # tmp_PointCloud['points'] = transform_pc_to_global_coordinate(tmp_PointCloud['points'], TheiaExtrinsics_4by4.T)
        # print("tmp_PointCloud['points'] = ", tmp_PointCloud['points'])

        # transform_pc_to_global_coordinate(points_N_3, MatT_4_4)

        # plot each point cloud in the global coordinate
        pointcloud_actor = create_pointcloud_actor(
           points=tmp_PointCloud['points'],
           colors=tmp_PointCloud['colors'] if 'colors' in tmp_PointCloud else None,
           )
        renderer.AddActor(pointcloud_actor)

        pc_polydata = create_pointcloud_polydata(
                                                points=tmp_PointCloud['points'],
                                                colors=tmp_PointCloud['colors'] if 'colors' in tmp_PointCloud else None,
                                                )
        appendFilterPC.AddInputData(pc_polydata)
        appendFilterModel.AddInputData(pc_polydata)

        # print("tmp_PointCloud['points'].shape = ", tmp_PointCloud['points'].shape)
        # print("tmp_PointCloud = ", tmp_PointCloud)
        # print("type(tmp_PointCloud) = ", type(tmp_PointCloud))
        # print("type(tmp_PointCloud['points']) = ", type(tmp_PointCloud['points']))
        if it==0:
            PointClouds = tmp_PointCloud
        else:
            PointClouds['points'] = np.concatenate((PointClouds['points'],tmp_PointCloud['points']), axis=0)
            PointClouds['colors'] = np.concatenate((PointClouds['colors'],tmp_PointCloud['colors']), axis=0)
        # print("PointClouds['points'].shape = ", PointClouds['points'].shape)
        it += 1
        if it>=viewNum:
            break

        # pc = point_cloud(1/pred_invDepth121)
        # print("pc.shape = ", pc.shape)
        # pc = np.reshape(pc, (pc.shape[0]*pc.shape[1],pc.shape[2]))
        # print("pc.shape = ", pc.shape)
        # # print("np.reshape(pc[:,:,0], (pc.shape[0]*pc.shape[1])).shape = ", np.reshape(pc[:,:,0], (pc.shape[0]*pc.shape[1])).shape)
        # fig = pyplot.figure()
        # ax = Axes3D(fig)
        # ax.scatter(pc[:,0], pc[:,1], pc[:,2])
        # pyplot.show()

    # plot the scatter 2D data of scale records, to find out the correlation between the predicted scales and the calculated scales from global SfM
    plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,1])
    plt.ylabel('scales calculated from global SfM/Colmap')
    plt.xlabel('scales predicted by DeMoN')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # appendFilterPC.Update()
    #
    # # export all point clouds in the same global coordinate to a local .ply file (for external visualization)
    # output_prefix = './'
    # # pointcloud_polydata = create_pointcloud_polydata(
    # #     points=PointClouds['points'],
    # #     colors=PointClouds['colors'] if 'colors' in PointClouds else None,
    # #     )
    # plywriter = vtk.vtkPLYWriter()
    # plywriter.SetFileName(output_prefix + 'points.ply')
    # plywriter.SetInputData(appendFilterPC.GetOutput())
    # # plywriter.SetInputData(pointcloud_polydata)
    # # plywriter.SetFileTypeToASCII()
    # plywriter.SetArrayName('Colors')
    # plywriter.Write()
    #
    # # Append all camera polydata
    # appendFilter = vtk.vtkAppendPolyData()
    # for image_pair12 in data.keys():
    #     if image_pair12 in image_pairs:
    #         image_name1, image_name2 = image_pair12.split("---")
    #         # # colmap
    #         for ids,val in imagesGT.items():
    #         # theia
    #         # for ids,val in TheiaGlobalPosesGT.items():
    #             if val.name==image_name1:
    #                 # # colmap
    #                 cam_actor = create_camera_actor(val.rotmat, val.tvec)
    #                 cam_polydata = create_camera_polydata(val.rotmat,val.tvec, True)
    #
    #                 # theia
    #                 # cam_actor = create_camera_actor(val.rotmat, -np.dot(val.rotmat, val.tvec))
    #                 # cam_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
    #                 renderer.AddActor(cam_actor)
    #                 # cam_polydata = create_camera_polydata(val.rotmat,-np.dot(val.rotmat, val.tvec), True)
    #                 appendFilter.AddInputData(cam_polydata)
    #                 # appendFilterModel.AddInputData(cam_polydata)
    # appendFilter.Update()
    # # appendFilterModel.Update()
    # # appendFilterModel = vtk.vtkAppendPolyData()
    # appendFilterModel.AddInputData(appendFilterPC.GetOutput())
    # appendFilterModel.AddInputData(appendFilter.GetOutput())
    # appendFilterModel.Update()
    # plywriterCam = vtk.vtkPLYWriter()
    # plywriterCam.SetFileName(output_prefix + 'cameras.ply')
    # plywriterCam.SetInputData(appendFilter.GetOutput())
    # # plywriterCam.SetFileTypeToASCII()
    # plywriterCam.Write()
    #
    # plywriterModel = vtk.vtkPLYWriter()
    # plywriterModel.SetFileName(output_prefix + 'fused_point_clouds.ply')
    # plywriterModel.SetInputData(appendFilterModel.GetOutput())
    # # plywriterModel.SetFileTypeToASCII()
    # plywriterModel.SetArrayName('Colors')
    # plywriterModel.Write()
    #
    # axes = vtk.vtkAxesActor()
    # axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    # axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    # axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    # axes.SetCylinderRadius(0.03)
    # axes.SetShaftTypeToCylinder()
    # renderer.AddActor(axes)
    #
    # renwin = vtk.vtkRenderWindow()
    # renwin.SetWindowName("Point Cloud Viewer")
    # renwin.SetSize(800,600)
    # renwin.AddRenderer(renderer)
    #
    #
    # # An interactor
    # interactor = vtk.vtkRenderWindowInteractor()
    # interstyle = vtk.vtkInteractorStyleTrackballCamera()
    # interactor.SetInteractorStyle(interstyle)
    # interactor.SetRenderWindow(renwin)
    #
    # # Start
    # interactor.Initialize()
    # interactor.Start()

if __name__ == "__main__":
    main()
