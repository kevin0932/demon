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


# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_umd~maryland_hotel3"
# infile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_umd~maryland_hotel3/demon_prediction/demon_hotel_umd~maryland_hotel3.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_umd~maryland_hotel3/GT_hotel_umd~maryland_hotel3.h5"
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85k1~living_room_night"
# infile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85k1~living_room_night/demon_prediction/demon_mit_w85k1~living_room_night.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85k1~living_room_night/GT_mit_w85k1~living_room_night.h5"
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85_lounge1~wg_gym_2"
# infile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85_lounge1~wg_gym_2/demon_prediction/demon_mit_w85_lounge1~wg_gym_2.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85_lounge1~wg_gym_2/GT_mit_w85_lounge1~wg_gym_2.h5"
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_123~classroom_32123_nov_2_2012_scan1_erika"
# infile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_123~classroom_32123_nov_2_2012_scan1_erika/demon_prediction/demon_mit_32_123~classroom_32123_nov_2_2012_scan1_erika.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_123~classroom_32123_nov_2_2012_scan1_erika/GT_mit_32_123~classroom_32123_nov_2_2012_scan1_erika.h5"
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/harvard_conf_big~hv_conf_big_1/demon_prediction_Gist066_kNNOneThird"
# infile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/harvard_conf_big~hv_conf_big_1/demon_prediction_Gist066_kNNOneThird/demon_Gist066_harvard_conf_big~hv_conf_big_1.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/harvard_conf_big~hv_conf_big_1/GT_harvard_conf_big~hv_conf_big_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_pool~pool_1/demon_prediction_Gist066"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_pool~pool_1/GT_mit_32_pool~pool_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_pingpong~pingpong_1"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_32_pingpong~pingpong_1/GT_mit_32_pingpong~pingpong_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/home_ts~apartment_ts_oct_31_2012_scan1_erika"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/home_ts~apartment_ts_oct_31_2012_scan1_erika/GT_home_ts~apartment_ts_oct_31_2012_scan1_erika.h5"


# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_36_ti_lab~tian_lab_1"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_36_ti_lab~tian_lab_1/GT_mit_36_ti_lab~tian_lab_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_pittsburg~hotel_pittsburg_scan1_2012_dec_12"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_pittsburg~hotel_pittsburg_scan1_2012_dec_12/GT_hotel_pittsburg~hotel_pittsburg_scan1_2012_dec_12.h5"

#################################################################################
###### Partially Correct!
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_barcelona~scan1_2012_july_23"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_barcelona~scan1_2012_july_23/GT_hotel_barcelona~scan1_2012_july_23.h5"
#################################################################################
###### Most Correct!

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_sf~scan2"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_sf~scan2/GT_hotel_sf~scan2.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_hkust~hk_hotel_1"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_hkust~hk_hotel_1/GT_hotel_hkust~hk_hotel_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_graz~scan1_2012_aug_29"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_graz~scan1_2012_aug_29/GT_hotel_graz~scan1_2012_aug_29.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2/demon_prediction_knn30_Gist066"
# # GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2/GT_hotel_beijing~beijing_hotel_2.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2/GT_hotel_beijing~beijing_hotel_2_test.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2"
# infile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2/demon_prediction/demon_hotel_beijing~beijing_hotel_2.h5"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/hotel_beijing~beijing_hotel_2/GT_hotel_beijing~beijing_hotel_2.h5"
#################################################################################

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/mit_32_pingpong~pingpong_1"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/mit_32_pingpong~pingpong_1/GT_mit_32_pingpong~pingpong_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/home_ts~apartment_ts_oct_31_2012_scan1_erika"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/home_ts~apartment_ts_oct_31_2012_scan1_erika/GT_home_ts~apartment_ts_oct_31_2012_scan1_erika.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_graz~scan1_2012_aug_29"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_graz~scan1_2012_aug_29/GT_hotel_graz~scan1_2012_aug_29.h5"

outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_beijing~beijing_hotel_2"
GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_beijing~beijing_hotel_2/GT_hotel_beijing~beijing_hotel_2.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/mit_32_pool~pool_1"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/mit_32_pool~pool_1/GT_mit_32_pool~pool_1.h5"

# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_barcelona~scan1_2012_july_23"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_barcelona~scan1_2012_july_23/GT_hotel_barcelona~scan1_2012_july_23.h5"
#
# outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_hkust~hk_hotel_1"
# GTfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D_Python/hotel_hkust~hk_hotel_1/GT_hotel_hkust~hk_hotel_1.h5"

ExhaustivePairInfile = ''
cameras = ''
images = ''


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
    correctionScale = np.exp(np.mean( (np.log(view1GTDepth) - np.log(DeMoNDepth)) ))
    return correctionScale


def visPointCloudInGlobalFrame(renderer, alpha, data_format, target_K, w, h, PoseSource='Theia', DepthSource='DeMoN', initBool=False):
    global PointCloudVisBool

    renderer.SetBackground(0, 0, 0)
    actors_to_be_cleared = renderer.GetActors()
    print("before: actors_to_be_cleared.GetNumberOfItems() = ", (actors_to_be_cleared.GetNumberOfItems()))
    for idx in range(actors_to_be_cleared.GetNumberOfItems()):
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
    dataGT1 = h5py.File(GTfile)

    for image_name1 in dataGT1.keys():
        print("adding ", image_name1)

        ###### Retrieve Ground Truth Global poses for image 1 and 2

        GTExtrinsics1_4by4 = np.eye(4)
        K1GT, GTExtrinsics1_4by4[0:3,0:3], GTExtrinsics1_4by4[0:3,3] = read_camera_params(dataGT1[image_name1]['camera'], lmuFreiburgFormat=True)
        tmp_view1 = read_view(dataGT1[image_name1], lmuFreiburgFormat=True)
        view1GT = adjust_intrinsics(tmp_view1, target_K, w, h,)

        if PointCloudVisBool == True:
            if PoseSource=='GT':
                GlobalExtrinsics1_4by4 = GTExtrinsics1_4by4
                # GlobalExtrinsics2_4by4 = GTExtrinsics2_4by4

            ###### scale global poses by a constant (Colmap and Theia may generate 3D reconstruction in different scales, which may differ from the real object depth scale)
            # alpha = 0.28 # 0.3 0.5
            GlobalExtrinsics1_4by4[0:3,3] = alpha * GlobalExtrinsics1_4by4[0:3,3]
            # GlobalExtrinsics2_4by4[0:3,3] = alpha * GlobalExtrinsics2_4by4[0:3,3]

            ###### get the first point clouds
            input_data = prepare_input_data(view1GT.image, view1GT.image, data_format)
            if DepthSource=='GT':
                if PoseSource=='GT':
                    scale_applied = 1
                tmp_PointCloud1 = visualize_prediction(
                            inverse_depth=1/view1GT.depth,
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics1_4by4[0:3,0:3],
                            t1=GlobalExtrinsics1_4by4[0:3,3],
                            rotation=rotmat_To_angleaxis(GlobalExtrinsics1_4by4[0:3,0:3]),
                            translation=GlobalExtrinsics1_4by4[0:3,3],   # should be changed, this is wrong!
                            scale=scale_applied)

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

        it +=1
        if it>=125:
            break

    if PointCloudVisBool == True:
        appendFilterPC.Update()

    if PointCloudVisBool == True:
        appendFilterModel.AddInputData(appendFilterPC.GetOutput())
        appendFilterModel.Update()

        plywriterModel = vtk.vtkPLYWriter()
        plywriterModel.SetFileName(os.path.join(outdir,'fused_point_clouds_colmap_alpha{0}.ply'.format(int(alpha*10000))))
        plywriterModel.SetInputData(appendFilterModel.GetOutput())
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

sliderMin = 0 #ImageViewer.GetSliceMin()
sliderMax = 20 #ImageViewer.GetSliceMax()
TheiaOrColmapOrGTPoses='GT'
DeMoNOrColmapOrGTDepths='GT'
PointCloudVisBool = True

def main():
    global PointCloudVisBool, TheiaOrColmapOrGTPoses, DeMoNOrColmapOrGTDepths, sliderMin, sliderMax, interactor, renderer, data_format, target_K, w, h
    # alpha = 0.128
    alpha = 1.0
    print("alpha is set to ", alpha)

    visPointCloudInGlobalFrame(renderer, alpha, data_format, target_K, w, h, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, initBool=True)

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

if __name__ == "__main__":
    main()
