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

# import tensorflow as tf
from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.helpers import angleaxis_to_rotation_matrix
import colmap_utils as colmap
from PIL import Image
# %matplotlib inline
import math
import cv2


Image = collections.namedtuple(
    "Image", ["id", "camera_id", "name", "qvec", "tvec", "rotmat", "angleaxis"])

ImagePairGT = collections.namedtuple("ImagePairGT", ["id1", "id2", "qvec12", "tvec12", "camera_id1", "name1", "camera_id2", "name2", "rotmat12"])
RotationAngularErrorRecord = collections.namedtuple("RotationAngularErrorRecord", ["id1", "name1", "id2_minErr", "name2_minErr", "RSymAngularError", "tSymAngularError", "scaleError"])

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
    parser.add_argument("--demon_path", required=True)
    parser.add_argument("--output_h5_dir_path", required=True)
    # parser.add_argument("--images_path", required=True)

    args = parser.parse_args()
    return args


def euler2quat(z=0, y=0, x=0):
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



def mat2euler(M, cy_thresh=None):
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
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))

def normalizeQuaternion(qvec1):
    qvec1_norm = np.linalg.norm(qvec1)
    if (qvec1_norm == 0):
        normalised_qvec1 = np.array([1, qvec1[1], qvec1[2], qvec1[3]])
    else:
        normalised_qvec1 = (1.0/qvec1_norm) * qvec1
    return normalised_qvec1

def quaternionMultiplication(qvec1, qvec2):
    qprod = qvec1
    qprod[0] = qvec2[0]*qvec1[0] - qvec2[1]*qvec1[1] - qvec2[2]*qvec1[2] - qvec2[3]*qvec1[3]
    qprod[1] = qvec2[0]*qvec1[1] + qvec2[1]*qvec1[0] - qvec2[2]*qvec1[3] + qvec2[3]*qvec1[2]
    qprod[2] = qvec2[0]*qvec1[2] + qvec2[1]*qvec1[3] + qvec2[2]*qvec1[0] - qvec2[3]*qvec1[1]
    qprod[3] = qvec2[0]*qvec1[3] - qvec2[1]*qvec1[2] + qvec2[2]*qvec1[1] + qvec2[3]*qvec1[0]
    return qprod

def quaternion_mult(q,r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]

def quaternionRotatePoint(q, point):
    #print("point.shape = ", point.shape)
    #r = [0]+point
    r = np.array([0, point[0], point[1], point[2]])
    #print("r.shape = ", r.shape)
    q_conj = [q[0],-1*q[1],-1*q[2],-1*q[3]]
    return quaternion_mult(quaternion_mult(q,r),q_conj)[1:]

def relativePose_from_AbsolutePose(qvec1, tvec1, qvec2, tvec2):
    normalised_qvec1 = normalizeQuaternion(qvec1)
    inv_normalized_qvec1 = np.array([normalised_qvec1[0], -normalised_qvec1[1], -normalised_qvec1[2], -normalised_qvec1[3]])
    qvec12 = quaternionMultiplication(normalizeQuaternion(inv_normalized_qvec1), normalizeQuaternion(qvec2))
    tvec12 = tvec2 - quaternionRotatePoint(normalizeQuaternion(qvec12), tvec1)
    return qvec12, tvec12

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

ImagePairTheia = collections.namedtuple("ImagePair", ["id1", "name1", "id2", "name2", "R_rotmat", "R_angleaxis", "t_vec"])

def read_id_name_pairs_from_theia(path_img_id_map):
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
    return image_id_name_pair

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

def main():
    args = parse_args()


    # images = dict()
    image_pairs = set()

    # recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
    # recondir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/images_demon/dense/'
    # demon_sun3d_train_beijing_hotel_2_010m_to_020m.h5
    recondir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m/hotel_beijing.beijing_hotel_2/demon_prediction/images_demon/dense/'
    cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
    imagesColmap = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
    # views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))

    InConsistentPairNames = True

    # reading theia intermediate output relative poses from textfile
    TheiaRtfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/RelativePoses_after_step7_global_position_estimation.txt'
    TheiaIDNamefilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/viewid_imagename_pairs_file.txt'
    # TheiaRtfilepath = '/home/kevin/ThesisDATA/gerrard-hall/TheiaReconstructionFromImage/fromImages/ExA/intermediate_results_v2/RelativePoses_after_step7_global_position_estimation.txt'
    # TheiaIDNamefilepath = '/home/kevin/ThesisDATA/gerrard-hall/TheiaReconstructionFromImage/fromImages/ExA/intermediate_results_v2/viewid_imagename_pairs_file.txt'
    TheiaRelativePosesGT = read_relative_poses_theia_output(TheiaRtfilepath,TheiaIDNamefilepath)
    image_id_name_pairs_GT = read_id_name_pairs_from_theia(TheiaIDNamefilepath)

    RotationAngularErrors = {}
    TranslationAngularErrors = {}
    ScaleErrors = []
    ReliablePairRecord = {}
    data = h5py.File(args.demon_path)
    # data_ToBeFused = data

    for image_pair12 in data.keys():
        print("Processing", image_pair12)

        if image_pair12 in image_pairs:
            continue

        image_name1, image_name2 = image_pair12.split("---")

        if InConsistentPairNames == True:
            ####### Added for dealing with inconsistent image names stored in .h5 pair-names! Should be commented out when using updated consistent codes
            tempN = image_name1.split('.')
            image_name1 = tempN[0]+'~'+tempN[1]+'.JPG'
            tempN = image_name2.split('.')
            image_name2 = tempN[0]+'~'+tempN[1]+'.JPG'
            converted_image_pair12 = "{}---{}".format(image_name1, image_name2)
            image_pair21 = "{}---{}".format(image_name2, image_name1)
            image_pairs.add(converted_image_pair12)
            image_pairs.add(image_pair21)
            #######
        else:
            image_pair21 = "{}---{}".format(image_name2, image_name1)
            image_pairs.add(image_pair12)
            image_pairs.add(image_pair21)

        # if image_pair21 not in data:
        #     continue

        # for imgIdx, name in image_id_name_pairs_GT.items():
        #     if name == image_name1:
        #         image_ID1 = imgIdx
        #     if name == image_name2:
        #         image_ID2 = imgIdx
        # for imgIdx, val in TheiaRelativePosesGT.items():
        #     if val.name1 == image_name1 and val.name2 == image_name2:
        #         relative_R_GT = val.R_rotmat
        #         relative_t_GT = -np.dot(relative_R_GT, val.t_vec)

        #print(imagesColmap)
        Extrinsic_R1 = np.eye(3)
        Extrinsic_t1 = np.zeros(3)
        Extrinsic_R2 = np.eye(3)
        Extrinsic_t2 = np.zeros(3)
        for imgIdx, val in imagesColmap.items():
            # print(val.name, "; ", image_name1)
            if val.name == image_name1:
                image_ID1 = imgIdx
                Extrinsic_R1 = colmap.quaternion_to_rotation_matrix(val.q)
                Extrinsic_t1 = val.t
            #else:
            #    print("image 1 is not found in colmap record!")
            if val.name == image_name2:
                image_ID2 = imgIdx
                Extrinsic_R2 = colmap.quaternion_to_rotation_matrix(val.q)
                Extrinsic_t2 = val.t
            #else:
            #    print("image 2 is not found in colmap record!")
        #print("Extrinsic_R2 = ", Extrinsic_R2, "; Extrinsic_R1 = ", Extrinsic_R1)
        #print("Extrinsic_t2 = ", Extrinsic_t2, "; Extrinsic_t1 = ", Extrinsic_t1)

        relative_R_GT = np.dot(Extrinsic_R2, Extrinsic_R1.T)
        relative_t_GT = np.dot(Extrinsic_R1, (-np.dot(Extrinsic_R2.T, Extrinsic_t2) + np.dot(Extrinsic_R1.T, Extrinsic_t1)))
        print("relativeGT = ", relative_R_GT, relative_t_GT)
        ### further filtering the image pairs by prediction sym error ### Freiburg's data
        pred_rotmat12 = data[image_pair12]["rotation"].value
        # pred_rotmat12 = data[image_pair12]["rotation_matrix"].value
        # pred_rotmat21 = data[image_pair21]["rotation"].value
        # pred_rotmat21 = data[image_pair21]["rotation_matrix"].value
        pred_trans12 = data[image_pair12]["translation"].value
        # pred_trans21 = data[image_pair21]["translation"].value
        print("prediction = ", pred_rotmat12, pred_trans12)

        pred_rotmat12angleaxis = rotmat_To_angleaxis(pred_rotmat12)
        # pred_rotmat21angleaxis = rotmat_To_angleaxis(pred_rotmat21)
        # theta_err_abs = abs(np.linalg.norm(pred_rotmat12angleaxis) - np.linalg.norm(pred_rotmat21angleaxis))
        # loop_rotation = np.dot(pred_rotmat12.T, pred_rotmat21)
        loop_rotation = np.dot(pred_rotmat12.T, relative_R_GT)
        RotationAngularErr = np.linalg.norm(rotmat_To_angleaxis(loop_rotation))
        RotationAngularErr = RotationAngularErr * 180 / math.pi
        TransMagInput = np.linalg.norm(pred_trans12)
        # TransMagOutput = np.linalg.norm(pred_trans21)
        TransMagOutput = np.linalg.norm(relative_t_GT)
        TransDistErr = TransMagInput - TransMagOutput   # can be different if normalized or not?
        # tmp = TheiaClamp(np.dot(pred_trans12, -pred_trans21)/(TransMagInput*TransMagOutput), -1, 1)   # can be different if normalized or not?
        tmp = TheiaClamp(np.dot(pred_trans12, relative_t_GT)/(TransMagInput*TransMagOutput), -1, 1)   # can be different if normalized or not?
        TransAngularErr = math.acos( tmp )
        TransAngularErr = TransAngularErr * 180 / math.pi
        # if RotationAngularErr > 7.5: # chosen by observing sym_err_hist
        #     print("image_pair12 ", image_pair12, " is skipped because of large sym error!!!")
        #     continue

        # print("image_id_name_pairs_GT.items() = ", image_id_name_pairs_GT.items())

        pred_scale12 = data[image_pair12]["scale"].value
        scaleGTbyTheia = np.linalg.norm(relative_t_GT)
        scale_err = abs(scaleGTbyTheia-pred_scale12)
        ScaleErrors.append(scale_err)
        # print("ReliablePairRecord.keys() = ", ReliablePairRecord.keys())
        if image_name1 not in ReliablePairRecord.keys():
            ReliablePairRecord[image_name1] = RotationAngularErrorRecord(id1=image_ID1, name1=image_name1, id2_minErr=image_ID2, name2_minErr=image_name2, RSymAngularError=RotationAngularErr, tSymAngularError=TransAngularErr, scaleError=scale_err)
        else:
            if ReliablePairRecord[image_name1].scaleError > scale_err:
                ReliablePairRecord[image_name1] = RotationAngularErrorRecord(id1=image_ID1, name1=image_name1, id2_minErr=image_ID2, name2_minErr=image_name2, RSymAngularError=RotationAngularErr, tSymAngularError=TransAngularErr, scaleError=scale_err)
            # if ReliablePairRecord[image_name1].RSymAngularError > RotationAngularErr:
            #     ReliablePairRecord[image_name1] = RotationAngularErrorRecord(id1=image_ID1, name1=image_name1, id2_minErr=image_ID2, name2_minErr=image_name2, RSymAngularError=RotationAngularErr, tSymAngularError=TransAngularErr, scaleError=scale_err)
            # if ReliablePairRecord[image_name1].tSymAngularError > TransAngularErr:
            #     ReliablePairRecord[image_name1] = RotationAngularErrorRecord(id1=image_ID1, name1=image_name1, id2_minErr=image_ID2, name2_minErr=image_name2, RSymAngularError=RotationAngularErr, tSymAngularError=TransAngularErr, scaleError=scale_err)

    output_dir = args.output_h5_dir_path

    print("Write a NeXus HDF5 file")
    # output_h5_filename = u"predSym_RChoice_fuse_southbuilding_demon.h5"
    # output_h5_filename = u"View100_fuse_gerrard_hall_demon.h5"
    # output_h5_filename = u"View128_fuse_southbuilding_demon.h5"
    output_h5_filename = u"View128ColmapFilter_fuse_southbuilding_demon.h5"
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

    image_pairs222 = set()

    for image_pair12 in data.keys():
        # print("Processing", image_pair12)
        if image_pair12 in image_pairs222:
            print("image_pair12 skipped")
            continue

        image_name1, image_name2 = image_pair12.split("---")
        if InConsistentPairNames == True:
            ####### Added for dealing with inconsistent image names stored in .h5 pair-names! Should be commented out when using updated consistent codes
            image_pair21 = "{}---{}".format(image_name2, image_name1)
            tempN = image_name1.split('.')
            image_name1 = tempN[0]+'~'+tempN[1]+'.JPG'
            tempN = image_name2.split('.')
            image_name2 = tempN[0]+'~'+tempN[1]+'.JPG'
            converted_image_pair12 = "{}---{}".format(image_name1, image_name2)
            converted_image_pair21 = "{}---{}".format(image_name2, image_name1)
            image_pairs222.add(image_pair12)
            image_pairs222.add(image_pair21)
            #######
        else:
            image_pair21 = "{}---{}".format(image_name2, image_name1)
            image_pairs222.add(image_pair12)
            image_pairs222.add(image_pair21)

        # if image_pair21 not in data:
        if image_pair21 not in data.keys():
            print("image_pair21 skipped")
            continue

        if image_name1 in ReliablePairRecord.keys():
            print(ReliablePairRecord[image_name1].name2_minErr, "; ", image_name2)
            if ReliablePairRecord[image_name1].name2_minErr == image_name2:
                # h5file.create_dataset((image_name1 + "---" + image_name2 + "/rotation_angleaxis"), data=data[image_pair12]["rotation_angleaxis"].value)
                # h5file.create_dataset((image_name1 + "---" + image_name2 + "/rotation_matrix"), data=data[image_pair12]["rotation_matrix"].value)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/rotation"), data=data[image_pair12]["rotation"].value)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/translation"), data=data[image_pair12]["translation"].value)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/depth"), data=data[image_pair12]["depth"].value)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/depth_upsampled"), data=data[image_pair12]["depth_upsampled"].value)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/flow"), data=data[image_pair12]["flow"].value)
                h5file.create_dataset((image_name1 + "---" + image_name2 + "/scale"), data=data[image_pair12]["scale"].value)
    h5file.close()   # be CERTAIN to close the file
    print("HDF5 file is written successfully:", output_h5_filepath)

    print("ScaleErrors = ", ScaleErrors)
if __name__ == "__main__":
    main()
