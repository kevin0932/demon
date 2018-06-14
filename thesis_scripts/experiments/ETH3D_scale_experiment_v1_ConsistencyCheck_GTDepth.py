import tensorflow as tf
from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.helpers import angleaxis_to_rotation_matrix
import colmap_utils as colmap
from depthmotionnet.vis import *
import collections

from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
import math
import h5py
import os
import cv2

from pyquaternion import Quaternion
import nibabel.quaternions as nq
import vtk

import os
import io
# import Image
from array import array

def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        # return bytearray(f.read())
        return (f.read())

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


ImageTheia = collections.namedtuple(
    "Image", ["id", "camera_id", "name", "qvec", "tvec", "rotmat", "angleaxis"])

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
                images[image_id] = ImageTheia(id=image_id, camera_id=camera_id, name=image_name, qvec=np.array([1,0,0,0]), tvec=tvec, rotmat=rotmat, angleaxis=R_angleaxis)

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


def load_image_pairs_from_consistency_filtering(path):
    image_pair_list = []
    image_pair_cnt = 0
    with open(path, "r") as fid1:
        while True:
            line = fid1.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                if len(elems) == 2:
                    image_name_1 = (elems[0])
                    image_name_2 = (elems[1])
                    print("image_name_1 = ", image_name_1, "; image_name_2 = ", image_name_2)
                    image_pair_list.append((image_name_1,image_name_2))
                    image_pair_cnt += 1
    print(image_pair_list)
    print("valid pair num = ", image_pair_cnt)
    return image_pair_list

def vtkSliderCallback2(obj, event):
    global TheiaOrColmapOrGTPoses, DeMoNOrColmapOrGTDepths, sliderMin, sliderMax, interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    sliderRepres = obj.GetRepresentation()
    pos = sliderRepres.GetValue()
    # contourFilter.SetValue(0, pos)
    alpha=pos

    #close_window(interactor)
    #del renWin, iren

    # renderer = visPointCloudInGlobalFrame(alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT)
    visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, initBool=True)
    #renderer = visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)
    renderer.Modified()
    print("vtkSliderCallback2~~~~~~~~~~~~~~~")
    # # #### vtk slidingbar to adjust some parameters Runtime
    # SliderRepres = vtk.vtkSliderRepresentation2D()
    # SliderRepres.SetMinimumValue(sliderMin)
    # SliderRepres.SetMaximumValue(sliderMax)
    # SliderRepres.SetValue(alpha)
    # SliderRepres.SetTitleText("Slice")
    # SliderRepres.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    # SliderRepres.GetPoint1Coordinate().SetValue(0.2, 0.6)
    # SliderRepres.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    # SliderRepres.GetPoint2Coordinate().SetValue(0.4, 0.6)
    #
    # SliderRepres.SetSliderLength(0.02)
    # SliderRepres.SetSliderWidth(0.03)
    # SliderRepres.SetEndCapLength(0.01)
    # SliderRepres.SetEndCapWidth(0.03)
    # SliderRepres.SetTubeWidth(0.005)
    # SliderRepres.SetLabelFormat("%3.0lf")
    # SliderRepres.SetTitleHeight(0.02)
    # SliderRepres.SetLabelHeight(0.02)
    #
    # SliderWidget = vtk.vtkSliderWidget()
    # SliderWidget.SetInteractor(interactor)
    # SliderWidget.SetRepresentation(SliderRepres)
    # SliderWidget.KeyPressActivationOff()
    # SliderWidget.SetAnimationModeToAnimate()
    # SliderWidget.SetEnabled(True)
    # SliderWidget.AddObserver("EndInteractionEvent", vtkSliderCallback2)

# # # reading theia intermediate output relative poses from textfile
# #TheiaRtfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/RelativePoses_after_step7_global_position_estimation.txt'
TheiaRtfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/RelativePoses_after_step9_BA.txt'
TheiaIDNamefilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/viewid_imagename_pairs_file.txt'
TheiaRelativePosesGT = read_relative_poses_theia_output(TheiaRtfilepath,TheiaIDNamefilepath)
# # # reading theia intermediate output global poses from textfile
# #TheiaGlobalPosesfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/after_step7_global_position_estimation.txt'
TheiaGlobalPosesfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/after_step9_BA.txt'
TheiaGlobalPosesGT = read_global_poses_theia_output(TheiaGlobalPosesfilepath,TheiaIDNamefilepath)


# outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs"
# # infile = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_15_50_050/kevin_southbuilding_demon.h5"
# # # GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2/GT_hotel_beijing~beijing_hotel_2.h5"
# # ExhaustivePairInfile = infile
# ExhaustivePairInfile = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5"
# recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/DenseSIFT/dense_192_256'
#
# infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs/BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_4000_survivorRatio_500_validPairNum_226.txt'
# # infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs/BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_36000_survivorRatio_400_validPairNum_573.txt'
#
# depth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/facade/ground_truth_depth/dslr_images'
#
# cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
# # images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
# images = colmap.read_images_txt(os.path.join(recondir,'sparse_aligned/0','images.txt'))
# #cameras = colmap.read_cameras_txt(os.path.join('/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/DenseSIFT/ground_truth_sparse/0','cameras.txt'))
# #images = colmap.read_images_txt(os.path.join('/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/DenseSIFT/ground_truth_sparse/0','images.txt'))
from collections import namedtuple

ImageGT = namedtuple('ImageGT',['cam_id','name','R','t'])


def read_images_txt_and_return_dict_index_by_name(filename):
    result = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()

        line1 = line
        line2 = f.readline()

        while line1:
            items = line1.split(' ')
            image = ImageGT(
                cam_id = int(items[8]),
                name = items[9].strip(),
                #q = tuple([float(x) for x in items[1:5]]),
                #t = tuple([float(x) for x in items[5:8]])
                R = colmap.quaternion_to_rotation_matrix(tuple([float(x) for x in items[1:5]])).astype(np.float64),
                t = np.array(tuple([float(x) for x in items[5:8]]), dtype=np.float32).astype(np.float64)
            )
            result[items[9].strip()] = image

            line1 = f.readline()
            line2 = f.readline()

    return result

# #### ETH3D Train facade
# outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs"
# ExhaustivePairInfile = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5"
# recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/DenseSIFT/dense_192_256'
#
# # infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_exhaustive_pairs/BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_4000_survivorRatio_500_validPairNum_226.txt'
# infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/demon_prediction_15_50_050/OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_3600000_survivorRatio_0_validPairNum_252.txt'
# depth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/facade/ground_truth_depth/dslr_images'
# ground_truth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/facade/dslr_calibration_undistorted'
# cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
# # images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
# images = colmap.read_images_txt(os.path.join(recondir,'sparse_aligned/0','images.txt'))
# imagesGT = read_images_txt_and_return_dict_index_by_name(os.path.join(ground_truth_dir_path,'images_onlyImageNames.txt'))

# #### ETH3D Train delivery_area
# outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/demon_prediction_exhaustive_pairs"
# ExhaustivePairInfile = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5"
# recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/DenseSIFT/dense_192_256'
#
# # infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/demon_prediction_exhaustive_pairs/BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_16000_survivorRatio_400_validPairNum_97.txt'
# infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/demon_prediction_15_50_050/OrderEnforced_OpticalFlow_OFscale_1_err_10000000000_survivorRatio_0_validPairNum_65.txt'
# depth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/delivery_area/ground_truth_depth/dslr_images'
# ground_truth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/delivery_area/dslr_calibration_undistorted'
# cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
# images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
# #images = colmap.read_images_txt(os.path.join(recondir,'sparse_aligned/0','images.txt'))
# imagesGT = read_images_txt_and_return_dict_index_by_name(os.path.join(ground_truth_dir_path,'images_onlyImageNames.txt'))

#### ETH3D Train relief
outdir = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/demon_prediction_exhaustive_pairs"
ExhaustivePairInfile = "/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5"
recondir = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/DenseSIFT/dense_192_256'

# infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/demon_prediction_exhaustive_pairs/BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_16000_survivorRatio_400_validPairNum_114.txt'
infile = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/demon_prediction_15_50_050/BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_OFscale_1_err_100000000000_survivorRatio_0_validPairNum_71.txt'
depth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/relief/ground_truth_depth/dslr_images'
ground_truth_dir_path = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/relief/dslr_calibration_undistorted'
cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
#images = colmap.read_images_txt(os.path.join(recondir,'sparse_aligned/0','images.txt'))
imagesGT = read_images_txt_and_return_dict_index_by_name(os.path.join(ground_truth_dir_path,'images_onlyImageNames.txt'))


# print(images)
# views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))

knn = 15 # 5
max_angle = 90*math.pi/180  # 60*math.pi/180
min_overlap_ratio = 0.4     # 0.5
w = 256
h = 192
normalized_intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5],np.float32)
target_K = np.eye(3)
target_K[0,0] = w*normalized_intrinsics[0]
target_K[1,1] = h*normalized_intrinsics[1]
target_K[0,2] = w*normalized_intrinsics[2]
target_K[1,2] = h*normalized_intrinsics[3]

# if True:
def get_tf_data_format():
    if tf.test.is_gpu_available(True):
        data_format='channels_first'
    else: # running on cpu requires channels_last data format
        data_format='channels_last'

    return data_format

data_format = get_tf_data_format()

def computePoint2LineDist(pt, lineP1=None, lineP2=None, lineNormal=None):
    if lineNormal is None:
        lineNormal=np.array([1,-1])
    if lineP1 is None:
        lineP1=np.array([0,0])
    if lineP2 is None:
        lineP2=np.array([1,1])

    line = (lineP2-lineP1)/np.linalg.norm(lineP2-lineP1)
    ap = pt - lineP1
    t = np.dot(ap, line)
    x =  lineP1 + t * line #  x is a point on line
    # print("point pt to be checked  :", pt)
    # print("point on line  :", x)
    print("distance from p:", np.linalg.norm(pt - x))

    # # cross product for distance
    # distN = np.linalg.norm(np.dot(ap, lineNormal))
    # print("distN cross prod:", distN)
    # # cross product for distance
    dist = np.linalg.norm(np.cross(ap, line))
    print("dist cross prod:", dist)
    return dist

def computeCorrectionScale(DeMoNPredictionInvDepth, GTDepth, DeMoNDepthThreshold):
    """ scale for correction is based on section 3.2 from paper by Eigen et. al 2014 https://arxiv.org/pdf/1406.2283.pdf"""
    """ don't count the DeMoN prediction depth (1/inv_depth) further than DeMoNDepthThreshold """
    DeMoNDepth = 1/DeMoNPredictionInvDepth
    DeMoNDepth = np.reshape(DeMoNDepth, [DeMoNDepth.shape[0]*DeMoNDepth.shape[1]])
    view1GTDepth = np.reshape(GTDepth, [GTDepth.shape[0]*GTDepth.shape[1]])
    tmpFilter = np.logical_and(view1GTDepth>0, DeMoNDepth<=DeMoNDepthThreshold)
    DeMoNDepth = DeMoNDepth[tmpFilter]
    view1GTDepth = view1GTDepth[tmpFilter]
    print("DeMoNDepth.shape = ", DeMoNDepth.shape)
    print("view1GTDepth.shape = ", view1GTDepth.shape)
    # don't count the inf in the ground truth depth from ETH3D
    tmpFilter = np.isfinite(view1GTDepth)
    DeMoNDepth = DeMoNDepth[tmpFilter]
    view1GTDepth = view1GTDepth[tmpFilter]
    print("DeMoNDepth.shape = ", DeMoNDepth.shape)
    print("view1GTDepth.shape = ", view1GTDepth.shape)
    correctionScale = np.exp(np.mean( (np.log(view1GTDepth) - np.log(DeMoNDepth)) ))
    return correctionScale


### adapted from Ben's code
def adjust_intrinsics_crop_depth(depthNpArr, K_old, K_new, width_new, height_new):
    from PIL import Image
    from skimage.transform import resize
    #from .helpers import safe_crop_image, safe_crop_array2d

    #original parameters
    fx = K_old[0,0]    # 2457.60
    fy = K_old[1,1]    # 2457.60
    cx = K_old[0,2]    # 1536
    cy = K_old[1,2]    # 1152
    width = depthNpArr.shape[1]    # 3072
    height = depthNpArr.shape[0] # 2304
    # print("in lmbspecialops")
    # print("view.K = ", view.K)
    # print(fx, " ", fy, " ", cx, " ", cy, " ", width, " ", height)

    #target param
    fx_new = K_new[0,0] # 0.89115971*256=228.136886
    fy_new = K_new[1,1] # 1.18821287*192=228.136871
    cx_new = K_new[0,2] # 128
    cy_new = K_new[1,2] # 96

    scale_x = fx_new/fx # 228.1369/2457.6=0.09282914
    scale_y = fy_new/fy # 228.1369/2457.6=0.09282914
    # print(fx_new, " ", fy_new, " ", cx_new, " ", cy_new, " ", scale_x, " ", scale_y)

    #resize to get the right focal length
    width_resize = int(width*scale_x)   # 0.09282914*3072=285.17118 => 285
    height_resize = int(height*scale_y) # 0.09282914*2304=213.878339 => 213
    # principal point position in the resized image
    cx_resize = cx*scale_x  # 0.09282914*1536=142.585559
    cy_resize = cy*scale_y  # 0.09282914*1152=106.939169
    # print(width_resize, " ", height_resize, " ", cx_resize, " ", cy_resize)
    # view.image.show()
    # return
    # img_resize = image.resize((width_resize, height_resize), Image.BILINEAR if scale_x > 1 else Image.LANCZOS)
    # img_resize.show()
    # if not view.depth is None:
    # max_depth    = np.max(depthNpArr)
    max_depth    = np.nanmax(depthNpArr)
    depth_resize = depthNpArr / max_depth
    depth_resize[depth_resize < 0.] = 0.
    depth_resize = resize(depth_resize, (height_resize,width_resize), 0,mode='constant') * max_depth

    #crop to get the right principle point and resolution
    # print(cx_resize, " ", cx_new, " ", cy_resize, " ", cy_new)
    x0 = int(round(cx_resize - cx_new)) # int(round(142.585559-128) = 15
    y0 = int(round(cy_resize - cy_new)) # int(round(106.939169-96) = 11
    x1 = x0 + int(width_new)
    y1 = y0 + int(height_new)
    # print(x0, " ", x1, " ", y0, " ", y1)
    if x0 < 0 or y0 < 0 or x1 > width_resize or y1 > height_resize:
        print('Warning: Adjusting intrinsics adds a border to the image')
        print("cropping is outside the new image size")
        # img_new = safe_crop_image(img_resize,(x0,y0,x1,y1),(127,127,127))
        depth_new = safe_crop_array2d(depth_resize,(x0,y0,x1,y1),0).astype(np.float32)

    else:
        # img_new = img_resize.crop((x0,y0,x1,y1))
        print("cropping is within the new image size")
        depth_new = depth_resize[y0:y1,x0:x1].astype(np.float32)
        # img_new.show()

    # print("adjust_intrinsics function return view successfully!")
    return depth_new


def visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource='Theia', DepthSource='DeMoN', initBool=False):
    global PointCloudVisBool
    # data = h5py.File(infile)
    image_pair_list = load_image_pairs_from_consistency_filtering(infile)
    print(image_pair_list)
    print(len(image_pair_list))
    print(image_pair_list[0])
    print(image_pair_list[0][0], " --- ", image_pair_list[0][1])
    # return
    data = h5py.File(ExhaustivePairInfile)

    dataExhaustivePairs = h5py.File(ExhaustivePairInfile)

    #renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    #renderer.RemoveAllViewProps()
    # renderer.ResetCamera()
    actors_to_be_cleared = renderer.GetActors()
    #print("actors_to_be_cleared = ", actors_to_be_cleared)
    print("before: actors_to_be_cleared.GetNumberOfItems() = ", (actors_to_be_cleared.GetNumberOfItems()))
    #or idx, actor in actors_to_be_cleared.items():
    #    renderer.RemoveActor(actor)
    for idx in range(actors_to_be_cleared.GetNumberOfItems()):
        #actors_to_be_cleared.GetNextActor()
        #nextActor = actors_to_be_cleared.GetNextActor()
        nextActor = actors_to_be_cleared.GetLastActor()
        renderer.RemoveActor(nextActor)
        print("remove one actor")
    renderer.Modified()
    actors_currently = renderer.GetActors()
    print("after: actors_currently.GetNumberOfItems() = ", (actors_currently.GetNumberOfItems()))

    appendFilterPC = vtk.vtkAppendPolyData()
    appendFilterModel = vtk.vtkAppendPolyData()

    image_pairs = set()
    it = 0
    inlierCnt = 0

    inlierfile = open(os.path.join(outdir, "inlier_image_pairs.txt"), "w")
    outlierfile = open(os.path.join(outdir, "outlier_image_pairs.txt"), "w")
    rawscalefile = open(os.path.join(outdir, "raw_scale_data_image_pairs.txt"), "w")

    # for image_pair12 in data.keys():
    #     print("Processing", image_pair12)
    #     image_name1, image_name2 = image_pair12.split("---")
    for pair_idx in range(len(image_pair_list)):
        image_name1, image_name2 = image_pair_list[pair_idx]
        image_pair12 = image_name1+"---"+image_name2
        print("Processing", image_pair12)
        # ####### Added for dealing with inconsistent image names stored in .h5 pair-names! Should be commented out when using updated consistent codes
        # tempN = image_name1.split('.')
        # image_name1 = tempN[0]+'~'+tempN[1]+'.JPG'
        # tempN = image_name2.split('.')
        # image_name2 = tempN[0]+'~'+tempN[1]+'.JPG'
        # #######
        image_pair21 = "{}---{}".format(image_name2, image_name1)
        # print(image_name1, "; ", image_name2)

        tmp_dict = {}
        for image_id, image in images.items():
            # print(image.name, "; ", image_name1, "; ", image_name2)
            if image.name == image_name1:
                tmp_dict[image_id] = image

        # tmp_dict = {image_id: image}
        # print("tmp_dict = ", tmp_dict)
        if len(tmp_dict)<1:
            continue
        tmp_views = colmap.create_views(cameras, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
        # print("tmp_views = ", tmp_views)
        tmp_views[0] = adjust_intrinsics(tmp_views[0], target_K, w, h,)
        view1 = tmp_views[0]

        tmp_dict = {}
        for image_id, image in images.items():
            # print(image.name, "; ", image_name1, "; ", image_name2)
            if image.name == image_name2:
                tmp_dict[image_id] = image

        # tmp_dict = {image_id: image}
        # print("tmp_dict = ", tmp_dict)
        if len(tmp_dict)<1:
            continue
        tmp_views = colmap.create_views(cameras, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
        # print("tmp_views = ", tmp_views)
        tmp_views[0] = adjust_intrinsics(tmp_views[0], target_K, w, h,)
        view2 = tmp_views[0]

        ###########################################################################################

        if image_pair12 in image_pairs:
            continue

        # print("view1 = ", view1)
        image_pairs.add(image_pair12)
        # image_pairs.add(image_pair21)

        # ###### Retrieve Theia Global poses for image 1 and 2
        # TheiaExtrinsics1_4by4 = np.eye(4)
        # TheiaExtrinsics2_4by4 = np.eye(4)
        # for ids,val in TheiaGlobalPosesGT.items():
        #     if val.name == image_name1:
        #         TheiaExtrinsics1_4by4[0:3,0:3] = val.rotmat
        #         TheiaExtrinsics1_4by4[0:3,3] = -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t
        #     if val.name == image_name2:
        #         TheiaExtrinsics2_4by4[0:3,0:3] = val.rotmat
        #         TheiaExtrinsics2_4by4[0:3,3] = -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        # # a colormap and a normalization instance
        # cmap = plt.cm.jet
        # # plt.imshow(data[image_pair12]["depth_upsampled"], cmap='Greys')
        # plt.imshow(view1.depth, cmap='Greys')
        ColmapExtrinsics1_4by4 = np.eye(4)
        ColmapExtrinsics1_4by4[0:3,0:3] = view1.R
        ColmapExtrinsics1_4by4[0:3,3] = view1.t# -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        ColmapExtrinsics2_4by4 = np.eye(4)
        ColmapExtrinsics2_4by4[0:3,0:3] = view2.R
        ColmapExtrinsics2_4by4[0:3,3] = view2.t # -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t


        TheiaExtrinsics1_4by4 = ColmapExtrinsics1_4by4
        TheiaExtrinsics2_4by4 = ColmapExtrinsics2_4by4

        #GTExtrinsics1_4by4 = ColmapExtrinsics1_4by4
        #GTExtrinsics2_4by4 = ColmapExtrinsics2_4by4
        GTExtrinsics1_4by4 = np.eye(4)
        GTExtrinsics1_4by4[0:3,0:3] = imagesGT[image_name1].R
        GTExtrinsics1_4by4[0:3,3] = imagesGT[image_name1].t# -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        GTExtrinsics2_4by4 = np.eye(4)
        GTExtrinsics2_4by4[0:3,0:3] = imagesGT[image_name2].R
        GTExtrinsics2_4by4[0:3,3] = imagesGT[image_name2].t # -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t


        view1GT = view1
        view2GT = view2

        print("~~~~~~~ adjust intrinsics for depth data! ~~~~~~~")
        # a rough camera parameter for trial
        depth_K_old = np.eye(3)
        depth_K_old[0,0] = 3414
        depth_K_old[1,1] = 3414
        depth_K_old[0,2] = 3024
        depth_K_old[1,2] = 2016

        tmpGTDepth1 = np.fromfile(os.path.join(depth_dir_path, image_name1),dtype=np.float32)
        # print("tmpGTDepth1.shape = ", tmpGTDepth1.shape)
        # print("tmpGTDepth1[10000:10020] = ", tmpGTDepth1[10000:10020])
        tmpGTDepth1[np.isinf(tmpGTDepth1)]=0.0
        # print("tmpGTDepth1[10000:10020] = ", tmpGTDepth1[10000:10020])
        tmpGTDepth1 = np.reshape(tmpGTDepth1, [4032,6048])
        # print("tmpGTDepth1.shape = ", tmpGTDepth1.shape)
        # plt.imshow(tmpGTDepth1)
        GT_depth_1 = adjust_intrinsics_crop_depth(tmpGTDepth1, depth_K_old, target_K, w, h)
        # print("GT_depth_1[20,100:120] = ", GT_depth_1[20,100:120])
        # plt.imshow(GT_depth_1)
        # plt.show()
        # tmpGTDepth2 = np.fromfile(os.path.join(depth_dir_path, image_name2),dtype=np.float32)
        # tmpGTDepth2 = np.reshape(tmpGTDepth2, [4032,6048])
        # GT_depth_2 = adjust_intrinsics_crop_depth(tmpGTDepth2, depth_K_old, target_K, w, h)

        # ###### Retrieve Ground Truth Global poses for image 1 and 2
        # dataGT1 = h5py.File(GTfile)
        # GTExtrinsics1_4by4 = np.eye(4)
        # K1GT, GTExtrinsics1_4by4[0:3,0:3], GTExtrinsics1_4by4[0:3,3] = read_camera_params(dataGT1[image_name1[:-4]]['camera'], lmuFreiburgFormat=False)
        # tmp_view1 = read_view(dataGT1[image_name1[:-4]], lmuFreiburgFormat=False)
        # view1GT = adjust_intrinsics(tmp_view1, target_K, w, h,)
        #
        # dataGT2 = h5py.File(GTfile)
        # GTExtrinsics2_4by4 = np.eye(4)
        # K2GT, GTExtrinsics2_4by4[0:3,0:3], GTExtrinsics2_4by4[0:3,3] = read_camera_params(dataGT2[image_name2[:-4]]['camera'], lmuFreiburgFormat=False)
        # tmp_view2 = read_view(dataGT2[image_name2[:-4]], lmuFreiburgFormat=False)
        # view2GT = adjust_intrinsics(tmp_view2, target_K, w, h,)

        ##### compute scales and scale correction with GroundTruth/Colmap Depth
        correctionScaleGT = computeCorrectionScale(data[image_pair12]['depth_upsampled'].value, GT_depth_1, 60)
        print("GT_depth_1.shape = ", GT_depth_1.shape)
        print("correctionScaleGT = ", correctionScaleGT)
        correctionScaleColmap = computeCorrectionScale(data[image_pair12]['depth_upsampled'].value, view1.depth, 60)
        transScaleTheia = np.linalg.norm(np.linalg.inv(TheiaExtrinsics2_4by4)[0:3,3] - np.linalg.inv(TheiaExtrinsics1_4by4)[0:3,3])
        transScaleColmap = np.linalg.norm(np.linalg.inv(ColmapExtrinsics2_4by4)[0:3,3] - np.linalg.inv(ColmapExtrinsics1_4by4)[0:3,3])
        transScaleGT = np.linalg.norm(np.linalg.inv(GTExtrinsics2_4by4)[0:3,3] - np.linalg.inv(GTExtrinsics1_4by4)[0:3,3])
        # print("transScaleTheia = ", transScaleTheia, "; transScaleColmap = ", transScaleColmap, "; transScaleGT = ", transScaleGT, "; demon scale = ", data[image_pair12]['scale'].value, "; correctionScaleGT = ", correctionScaleGT, "; correctionScaleColmap = ", correctionScaleColmap)
        pred_scale = data[image_pair12]['scale'].value

        ##### add other characteristics for later inlier investigation
        pred_rotation12 = data[image_pair12]['rotation'].value
        pred_translation12 = data[image_pair12]['translation'].value
        absOrientationErrorInDeg = np.rad2deg(np.arccos((np.trace(pred_rotation12) - 1) / 2))
        absOrientationError = np.arccos((np.trace(pred_rotation12) - 1) / 2)
        view_overlap_ratio = compute_view_overlap( view1, view2 )

        # GTbaselineLength_v2 = np.linalg.norm(imagesGT[image_name2].t-imagesGT[image_name1].t)
        GTbaselineLength = np.linalg.norm(-np.dot(imagesGT[image_name2].R.T, imagesGT[image_name2].t)+np.dot(imagesGT[image_name1].R.T, imagesGT[image_name1].t))
        # print(GTbaselineLength, " ", GTbaselineLength_v2)
        # if GTbaselineLength != GTbaselineLength_v2:
        #     print("Error in baseline calculation!")
        #     return

        # # if computePoint2LineDist(np.array([correctionScaleGT,transScaleGT]))<0.010:
        # #     inlierfile.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(image_pair12, absOrientationErrorInDeg, absOrientationError, view_overlap_ratio, GTbaselineLength, pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap))
        # #     # continue
        # #     inlierCnt += 1
        # #
        # # outlierfile.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(image_pair12, absOrientationErrorInDeg, absOrientationError, view_overlap_ratio, GTbaselineLength, pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap))
        # rawscalefile.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(image_pair12, absOrientationErrorInDeg, absOrientationError, view_overlap_ratio, GTbaselineLength, pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap))

        if it==0:
            scaleRecordMat = np.array([pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap, absOrientationErrorInDeg, absOrientationError, view_overlap_ratio, GTbaselineLength])
            initColmapGTRatio = transScaleColmap/transScaleGT
        else:
            scaleRecordMat = np.vstack((scaleRecordMat, np.array([pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap, absOrientationErrorInDeg, absOrientationError, view_overlap_ratio, GTbaselineLength])))
        # print("scaleRecordMat.shape = ", scaleRecordMat.shape)

        if PointCloudVisBool == True:
            if PoseSource=='Theia':
                GlobalExtrinsics1_4by4 = TheiaExtrinsics1_4by4
                GlobalExtrinsics2_4by4 = TheiaExtrinsics2_4by4
            if PoseSource=='Colmap':
                GlobalExtrinsics1_4by4 = ColmapExtrinsics1_4by4
                GlobalExtrinsics2_4by4 = ColmapExtrinsics2_4by4
            if PoseSource=='GT':
                GlobalExtrinsics1_4by4 = GTExtrinsics1_4by4
                GlobalExtrinsics2_4by4 = GTExtrinsics2_4by4

            ###### scale global poses by a constant (Colmap and Theia may generate 3D reconstruction in different scales, which may differ from the real object depth scale)
            # alpha = 0.28 # 0.3 0.5
            GlobalExtrinsics1_4by4[0:3,3] = alpha * GlobalExtrinsics1_4by4[0:3,3]
            GlobalExtrinsics2_4by4[0:3,3] = alpha * GlobalExtrinsics2_4by4[0:3,3]

            ###### get the first point clouds
            input_data = prepare_input_data(view1.image, view2.image, data_format)
            if DepthSource=='Colmap':
                if PoseSource=='Theia':
                    scale_applied = transScaleTheia/transScaleColmap
                if PoseSource=='Colmap':
                    scale_applied = 1
                if PoseSource=='GT':
                    # scale_applied = transScaleGT/transScaleColmap
                    # scale_applied = 1/initColmapGTRatio
                    scale_applied = 1/1.72921055    # fittedColmapGTRatio = 1.72921055
                tmp_PointCloud1 = visualize_prediction(
                            inverse_depth=1/view1.depth,
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics1_4by4[0:3,0:3],
                            t1=GlobalExtrinsics1_4by4[0:3,3],
                            rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                            translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                            scale=scale_applied)

            elif DepthSource=='DeMoN':
                if PoseSource=='Theia':
                    scale_applied = transScaleTheia
                if PoseSource=='Colmap':
                    # scale_applied = transScaleColmap
                    # scale_applied = data[image_pair12]['scale'].value
                    scale_applied = correctionScaleColmap
                if PoseSource=='GT':
                    scale_applied = transScaleGT
                    # scale_applied = data[image_pair12]['scale'].value
                    # scale_applied = correctionScaleGT
                tmp_PointCloud1 = visualize_prediction(
                            inverse_depth=data[image_pair12]['depth_upsampled'].value,
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics1_4by4[0:3,0:3],
                            t1=GlobalExtrinsics1_4by4[0:3,3],
                            rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                            translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                            scale=scale_applied)
            elif DepthSource=='GT':
                if PoseSource=='Theia':
                    scale_applied = transScaleTheia/transScaleGT
                if PoseSource=='Colmap':
                    # scale_applied = transScaleColmap/transScaleGT
                    # scale_applied = initColmapGTRatio
                    scale_applied = 1.72921055    # fittedColmapGTRatio = 1.72921055
                if PoseSource=='GT':
                    scale_applied = 1
                tmp_PointCloud1 = visualize_prediction(
                            inverse_depth=1/GT_depth_1,
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics1_4by4[0:3,0:3],
                            t1=GlobalExtrinsics1_4by4[0:3,3],
                            rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                            translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                            scale=scale_applied)
            # vis2 = cv2.cvtColor(view1.depth, cv2.COLOR_GRAY2BGR)
            # #Displayed the image
            # cv2.imshow("WindowNameHere", vis2)
            # cv2.waitKey(0)

            pointcloud_actor = create_pointcloud_actor(
               points=tmp_PointCloud1['points'],
               colors=tmp_PointCloud1['colors'] if 'colors' in tmp_PointCloud1 else None,
               )
            renderer.AddActor(pointcloud_actor)

            pc_polydata = create_pointcloud_polydata(
                                                    points=tmp_PointCloud1['points'],
                                                    colors=tmp_PointCloud1['colors'] if 'colors' in tmp_PointCloud1 else None,
                                                    )
            appendFilterPC.AddInputData(pc_polydata)

            if it==0:
                PointClouds = tmp_PointCloud1
            else:
                PointClouds['points'] = np.concatenate((PointClouds['points'],tmp_PointCloud1['points']), axis=0)
                PointClouds['colors'] = np.concatenate((PointClouds['colors'],tmp_PointCloud1['colors']), axis=0)

            cam1_actor = create_camera_actor(GlobalExtrinsics1_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,3])
            # cam1_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
            renderer.AddActor(cam1_actor)
            cam1_polydata = create_camera_polydata(GlobalExtrinsics1_4by4[0:3,0:3],GlobalExtrinsics1_4by4[0:3,3], True)
            appendFilterModel.AddInputData(cam1_polydata)

            if False:   # debug: if the second cam is added for visualization
                ###### get the 2nd point clouds
                input_data = prepare_input_data(view2.image, view1.image, data_format)
                transScale = np.linalg.norm(GlobalExtrinsics1_4by4.T[0:3,3] - GlobalExtrinsics2_4by4.T[0:3,3])
                tmp_PointCloud2 = visualize_prediction(
                            inverse_depth=1/view2.depth,
                            # inverse_depth=1/(view2.depth*transScale),
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics2_4by4[0:3,0:3],
                            t1=GlobalExtrinsics2_4by4[0:3,3],
                            rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics1_4by4[0:3,0:3], GlobalExtrinsics2_4by4[0:3,0:3].T)),
                            translation=GlobalExtrinsics1_4by4[0:3,3],   # should be changed, this is wrong!
                            # scale=data[image_pair12]['scale'].value)
                            # scale=transScale)
                            # scale=1/transScale)
                            # scale=1/data[image_pair12]['scale'].value)
                            # scale=transScale/data[image_pair12]['scale'].value)
                            # scale=data[image_pair12]['scale'].value*transScale)
                            scale=1)

                # vis2 = cv2.cvtColor(view1.depth, cv2.COLOR_GRAY2BGR)
                # #Displayed the image
                # cv2.imshow("WindowNameHere", vis2)
                # cv2.waitKey(0)

                pointcloud_actor = create_pointcloud_actor(
                   points=tmp_PointCloud2['points'],
                   colors=tmp_PointCloud2['colors'] if 'colors' in tmp_PointCloud2 else None,
                   )
                renderer.AddActor(pointcloud_actor)

                pc_polydata = create_pointcloud_polydata(
                                                        points=tmp_PointCloud2['points'],
                                                        colors=tmp_PointCloud2['colors'] if 'colors' in tmp_PointCloud2 else None,
                                                        )
                appendFilterPC.AddInputData(pc_polydata)

                PointClouds['points'] = np.concatenate((PointClouds['points'],tmp_PointCloud2['points']), axis=0)
                PointClouds['colors'] = np.concatenate((PointClouds['colors'],tmp_PointCloud2['colors']), axis=0)

                cam2_actor = create_camera_actor(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics2_4by4[0:3,3])
                # cam2_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
                renderer.AddActor(cam2_actor)
                cam2_polydata = create_camera_polydata(GlobalExtrinsics2_4by4[0:3,0:3],GlobalExtrinsics2_4by4[0:3,3], True)
                appendFilterModel.AddInputData(cam2_polydata)

        it +=1
        # if it>=2000:
        #     break

    if PointCloudVisBool == True:
        appendFilterPC.Update()
    inlierfile.close()
    print("inlier matches num = ", inlierCnt, " out of total pair num = ", it)
    outlierfile.close()
    rawscalefile.close()

    # # ###### Compute the slope of the fitted line to reflect the scale differences among DeMoN, Theia and Colmap
    # # tmpFittingCoef_DeMoNTheia = np.polyfit(scaleRecordMat[:,0], scaleRecordMat[:,1], 1)
    # # print("tmpFittingCoef_DeMoNTheia = ", tmpFittingCoef_DeMoNTheia)
    # # tmpFittingCoef_DeMoNColmap = np.polyfit(scaleRecordMat[:,0], scaleRecordMat[:,2], 1)
    # # print("tmpFittingCoef_DeMoNColmap = ", tmpFittingCoef_DeMoNColmap)
    # # tmpFittingCoef_TheiaColmap = np.polyfit(scaleRecordMat[:,1], scaleRecordMat[:,2], 1)
    # # print("tmpFittingCoef_TheiaColmap = ", tmpFittingCoef_TheiaColmap)
    # tmpFittingCoef_Colmap_GT = np.polyfit(scaleRecordMat[:,3], scaleRecordMat[:,2], 1)
    # print("tmpFittingCoef_Colmap_GT = ", tmpFittingCoef_Colmap_GT)
    # plot the scatter 2D data of scale records, to find out the correlation between the predicted scales and the calculated scales from global SfM
    np.savetxt(os.path.join(outdir,'scale_record_DeMoN_Theia_Colmap_GT_correctionGT_correctionColmap.txt'), scaleRecordMat, fmt='%f')
    if False:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,1])
        plt.ylabel('scales calculated from Theia global SfM')
        plt.xlabel('scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if False:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,2])
        # plt.scatter(1/scaleRecordMat[:,0],scaleRecordMat[:,2])
        plt.ylabel('scales calculated from Colmap')
        plt.xlabel('scales predicted by DeMoN')
        # plt.xlabel('inv_scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,3])
        plt.ylabel('scales calculated from SUN3D Ground Truth')
        plt.xlabel('scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,4])
        plt.ylabel('scales calculated from correction scale (SUN3D Ground Truth)')
        plt.xlabel('scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,5])
        plt.ylabel('scales calculated from correction scale (Colmap)')
        plt.xlabel('scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,4],scaleRecordMat[:,3])
        plt.ylabel('scales calculated from SUN3D Ground Truth')
        plt.xlabel('scales calculated from correction scale (SUN3D Ground Truth)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,5],scaleRecordMat[:,3])
        plt.ylabel('scales calculated from SUN3D Ground Truth')
        plt.xlabel('scales calculated from correction scale (Colmap)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,5],scaleRecordMat[:,2])
        plt.ylabel('scales calculated from Colmap')
        plt.xlabel('scales calculated from correction scale (Colmap)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,5],scaleRecordMat[:,4])
        plt.ylabel('scales calculated from correction scale (SUN3D Ground Truth)')
        plt.xlabel('scales calculated from correction scale (Colmap)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if False:
        plt.scatter(scaleRecordMat[:,2],scaleRecordMat[:,3])
        plt.ylabel('scales calculated from SUN3D Ground Truth')
        plt.xlabel('scales calculated from Colmap')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if False:
        x = np.linspace(np.min(scaleRecordMat[:,3]), np.max(scaleRecordMat[:,3]), 1000)
        # dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
        y = 1.72921055*x+0.02395182
        plt.plot(x, y, 'r-', lw=2)
        plt.text(0.2, 0.15, r'fitted line with slope = 1.72921055')
        plt.scatter(scaleRecordMat[:,3],scaleRecordMat[:,2])
        plt.xlabel('scales calculated from SUN3D Ground Truth')
        plt.ylabel('scales calculated from Colmap')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if False:
        plt.scatter(scaleRecordMat[:,1],scaleRecordMat[:,2])
        plt.ylabel('scales calculated from Colmap')
        plt.xlabel('scales calculated from Theia global SfM')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    if PointCloudVisBool == True:
        appendFilterModel.AddInputData(appendFilterPC.GetOutput())
        appendFilterModel.Update()

        plywriterModel = vtk.vtkPLYWriter()
        plywriterModel.SetFileName(os.path.join(outdir,'fused_point_clouds_colmap_alpha{0}.ply'.format(int(alpha*10000))))
        # plywriterModel.SetInputData(appendFilterModel.GetOutput())
        plywriterModel.SetInputData(appendFilterPC.GetOutput())
        # plywriterModel.SetFileTypeToASCII()
        plywriterModel.SetArrayName('Colors')
        plywriterModel.Write()

        axes = vtk.vtkAxesActor()
        axes.GetXAxisCaptionActor2D().SetHeight(0.05)
        axes.GetYAxisCaptionActor2D().SetHeight(0.05)
        axes.GetZAxisCaptionActor2D().SetHeight(0.05)
        axes.SetCylinderRadius(0.03)
        axes.SetShaftTypeToCylinder()
        renderer.AddActor(axes)

        renderer.Modified()

renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)
interactor = vtk.vtkRenderWindowInteractor()
#SliderRepres = vtk.vtkSliderRepresentation2D()
#SliderWidget = vtk.vtkSliderWidget()

sliderMin = 0 #ImageViewer.GetSliceMin()
sliderMax = 20 #ImageViewer.GetSliceMax()
# TheiaOrColmapOrGTPoses='Colmap'
# TheiaOrColmapOrGTPoses='Theia'
TheiaOrColmapOrGTPoses='GT'
DeMoNOrColmapOrGTDepths='DeMoN'
# DeMoNOrColmapOrGTDepths='Colmap'
# DeMoNOrColmapOrGTDepths='GT'
PointCloudVisBool = True

def main():
    #global SliderRepres, SliderWidget, interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    global PointCloudVisBool, TheiaOrColmapOrGTPoses, DeMoNOrColmapOrGTDepths, sliderMin, sliderMax, interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    # alpha = 0.128
    alpha = 1.0
    # renderer = visPointCloudInGlobalFrame(alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)
    print("alpha is set to ", alpha)

    visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, initBool=True)
    #renderer = visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)

    if PointCloudVisBool == True:
        renwin = vtk.vtkRenderWindow()
        renwin.SetWindowName("Point Cloud Viewer")
        renwin.SetSize(800,600)
        renwin.AddRenderer(renderer)

        # An interactor
        # interactor = vtk.vtkRenderWindowInteractor()
        interstyle = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(interstyle)
        interactor.SetRenderWindow(renwin)

        # #### vtk slidingbar to adjust some parameters Runtime
        SliderRepres = vtk.vtkSliderRepresentation2D()
        # sliderMin = 0 #ImageViewer.GetSliceMin()
        # sliderMax = 10 #ImageViewer.GetSliceMax()
        SliderRepres.SetMinimumValue(sliderMin)
        SliderRepres.SetMaximumValue(sliderMax)
        SliderRepres.SetValue(alpha)
        SliderRepres.SetTitleText("Alpha --- A constant scale added to the translation of global poses from Theia/Colmap")
        SliderRepres.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        SliderRepres.GetPoint1Coordinate().SetValue(0.05, 0.06)
        SliderRepres.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        SliderRepres.GetPoint2Coordinate().SetValue(0.95, 0.06)

        SliderRepres.SetSliderLength(0.02)
        SliderRepres.SetSliderWidth(0.03)
        SliderRepres.SetEndCapLength(0.01)
        SliderRepres.SetEndCapWidth(0.03)
        SliderRepres.SetTubeWidth(0.005)
        SliderRepres.SetLabelFormat("%1.3lf")
        SliderRepres.SetTitleHeight(0.02)
        SliderRepres.SetLabelHeight(0.02)

        SliderWidget = vtk.vtkSliderWidget()
        SliderWidget.SetInteractor(interactor)
        SliderWidget.SetRepresentation(SliderRepres)
        SliderWidget.KeyPressActivationOff()
        SliderWidget.SetAnimationModeToAnimate()
        SliderWidget.SetEnabled(True)
        SliderWidget.AddObserver("EndInteractionEvent", vtkSliderCallback2)

        # Start
        interactor.Initialize()
        interactor.Start()

    # # imgTest = Image.loadImage('/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/facade/ground_truth_depth/dslr_images/DSC_0326.JPG')
    # # imgNpArr = np.array( imgTest.getdata(), np.float ).reshape(imgTest.size[1], imgTest.size[0])
    # # print("imgNpArr.shape = ", imgNpArr.shape)
    #
    # # imgTest = Image.open('/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/facade/ground_truth_depth/dslr_images/DSC_0326.JPG')
    # # imgNpArr = np.array(imgTest)
    # # print("imgNpArr.shape = ", imgNpArr.shape)
    #
    # # from scipy.misc import imread
    # imgPath = '/home/kevin/ThesisDATA/ETH3D/multi_view_training_depth/facade/ground_truth_depth/dslr_images/DSC_0326.JPG'
    # # print(imread(imgPath).shape)
    #
    # tst = '/home/kevin/ThesisDATA/CVG_Capitole/DenseSIFT/resized_images_2304_3072/P1000678.JPG'
    #
    #
    # imgTest = np.fromfile(imgPath,dtype=np.float32)
    # print("imgTest = ", imgTest)
    # print("imgTest.shape = ", imgTest.shape)
    # print("imgTest[0:10] = ", imgTest[0:10])
    # print("imgTest[100000:100020] = ", imgTest[100000:100020])
    # a = imgTest[np.isfinite(imgTest)]
    # print("a = ", a)
    # print("a.shape = ", a.shape)
    # print("a[0:10] = ", a[0:10])
    # print(np.min(a), ", ", np.max(a))
    #
    # # with open(imgPath, 'rb') as f:
    # #     contents = f.read()
    # #     # print("contents = ", contents)
    # # # print("contents = ", contents)
    # # # img = np.fromstring(buffer(contents), dtype='<f2')
    # # # img = np.fromstring((contents), dtype='<f2')
    # # # img = np.fromstring((contents), dtype='<u1')
    # # img = np.fromstring((contents), dtype='>u1')
    # # # img = np.fromstring((contents), dtype='>f2')
    # # # img = np.fromstring((contents))
    # # print("len(contents) = ", len(contents))
    # #
    # # import struct
    # # #image = struct.unpack('f', contents[0:4])
    # #
    # # print("img[0:10] = ", img[0:10])
    # #
    # # print("img.shape = ", img.shape)
    # #
    # # unpackedFloat = []
    # # for i in range(len(contents)):
    # #     #print("contents[4*i:4*i+4] = ", contents[4*i:4*i+4])
    # #     unpackedFloat.append(struct.unpack('f', contents[4*i:4*i+4]))
    # # print("image[0] = ", image[0])
    # # unpackedFloat = np.array(unpackedFloat)
    # # print("unpackedFloat.shape = ", unpackedFloat.shape)


if __name__ == "__main__":
    main()
