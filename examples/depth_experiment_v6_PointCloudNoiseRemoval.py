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

from scipy.spatial import distance
from scipy import spatial
import math
import sys
examples_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(examples_dir, '..', 'lmbspecialops', 'python'))

def compute_normals_from_depth(depth_numpyArr):
    h = depth_numpyArr.shape[0]
    w = depth_numpyArr.shape[1]
    normal_map = np.empty([h,w,3],dtype=np.float32)
    for x in range(w):
        for y in range(h):
            if x==0 or y==0 or x==w-1 or y==h-1:
                n = np.array([0,0,1])
            else:
                dzdx = (depth_numpyArr[y, x+1] - depth_numpyArr[y, x-1]) / 2.0;
                dzdy = (depth_numpyArr[y+1, x] - depth_numpyArr[y-1, x]) / 2.0;
                n = np.array([-dzdx, -dzdy, 1])
                # n = np.array([-dzdx, -dzdy, depth_numpyArr[y, x]])
                # n = np.array([-dzdx, -dzdy, 1/depth_numpyArr[y, x]])
                # print(n)
                n = n/np.linalg.norm(n)
            normal_map[y,x,:] = n

    # print(normal_map)
    # np.savetxt('test.txt', normal_map[:,:,0])
    print(normal_map.shape)
    # # plt.imshow((normal_map-np.min(normal_map))/(np.max(normal_map)-np.min(normal_map)))
    # plt.imshow(normal_map/2+0.5)
    # print(np.min(normal_map), " ", np.max(normal_map))
    # plt.imshow(normal_map/2+0.5)
    # plt.show()

    return normal_map

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


# # reading theia intermediate output relative poses from textfile
#TheiaRtfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/RelativePoses_after_step7_global_position_estimation.txt'
TheiaRtfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/RelativePoses_after_step9_BA.txt'
TheiaIDNamefilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/viewid_imagename_pairs_file.txt'
TheiaRelativePosesGT = read_relative_poses_theia_output(TheiaRtfilepath,TheiaIDNamefilepath)
# # reading theia intermediate output global poses from textfile
#TheiaGlobalPosesfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/after_step7_global_position_estimation.txt'
TheiaGlobalPosesfilepath = '/home/kevin/JohannesCode/theia_trial_demon/intermediate_results_southbuilding_01012018/after_step9_BA.txt'
TheiaGlobalPosesGT = read_global_poses_theia_output(TheiaGlobalPosesfilepath,TheiaIDNamefilepath)




# # weights_dir = '/home/ummenhof/projects/demon/weights'
# weights_dir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/weights'
# # outdir = '/home/kevin/DeMoN_Prediction/south_building'
# # infile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions.h5'
# ## infile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions_v1_05012018.h5'

# outdir = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction'
# outdir = '/home/kevin/ThesisDATA/person-hall/demon_prediction'
# outdir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction"
# outdir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction"
# outdir = "/home/kevin/JohannesCode/ws1/demon_prediction/freiburgSettingBak"
# outdir = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m/hotel_beijing.beijing_hotel_2/demon_prediction"
# infile = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction/gerrard_hall_predictions.h5'
# infile = '/home/kevin/ThesisDATA/person-hall/demon_prediction/person_hall_predictions.h5'
# infile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction/CVG_barcelona_predictions.h5"
# infile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction/CVG_redmond_predictions.h5"
# infile = "/home/kevin/JohannesCode/ws1/demon_prediction/kevin_southbuilding_predictions_08012018.h5"
# infile = "/home/kevin/JohannesCode/ws1/demon_prediction/freiburgSettingBak/View128_fuse_southbuilding_demon.h5"
# infile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m/hotel_beijing.beijing_hotel_2/demon_prediction/View128ColmapFilter_demon_sun3d_train_beijing_hotel_2_010m_to_020m.h5"
# infile = "/home/kevin/JohannesCode/ws1/demon_prediction/freiburgSettingBak/View128ColmapFilter_fuse_southbuilding_demon.h5"
# ExhaustivePairInfile = "/home/kevin/JohannesCode/ws1/demon_prediction/freiburgSettingBak/kevin_southbuilding_predictions_06012018.h5"
# ExhaustivePairInfile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m/hotel_beijing.beijing_hotel_2/demon_prediction/demon_sun3d_train_beijing_hotel_2_010m_to_020m.h5"

# outimagedir_small = os.path.join(outdir,'images_demon_small')
# outimagedir_large = os.path.join(outdir,'images_demon')
# os.makedirs(outdir, exist_ok=True)
# os.makedirs(outimagedir_small, exist_ok=True)
# os.makedirs(outimagedir_large, exist_ok=True)
# os.makedirs(os.path.join(outdir,'graydepthmap'), exist_ok=True)
# os.makedirs(os.path.join(outdir,'vizdepthmap'), exist_ok=True)

# recondir = '/misc/lmbraid12/depthmotionnet/datasets/mvs_colmap/south-building/mvs/'
# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# recondir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m/hotel_beijing.beijing_hotel_2/demon_prediction/images_demon/dense/'
# recondir = '/home/kevin/ThesisDATA/ToyDataset_Desk/dense/'
# recondir = '/home/kevin/ThesisDATA/gerrard-hall/dense/'
# recondir = '/home/kevin/ThesisDATA/person-hall/dense/'
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/dense/"
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/dense/"

# outdir = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction"
# # infile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/View128ColmapFilter_demon_sun3d_train_hotel_beijing~beijing_hotel_2.h5"
# # ExhaustivePairInfile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/demon_sun3d_train_hotel_beijing~beijing_hotel_2.h5"
# #infile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/demon_sun3d_train_hotel_beijing~beijing_hotel_2.h5"
# infile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/kevin_southbuilding_demon.h5"
# ExhaustivePairInfile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/kevin_southbuilding_demon.h5"
# recondir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/images_demon/dense/'

outdir = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_mit_w85_lounge1~wg_lounge1_1/demon_prediction/images_demon/dense/1"
# infile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_mit_w85_lounge1~wg_lounge1_1/demon_prediction/images_demon/dense/1/kevin_exhaustive_demon.h5"
infile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_mit_w85_lounge1~wg_lounge1_1/demon_prediction/demon_sun3d_train_mit_w85_lounge1~wg_lounge1_1.h5"
ExhaustivePairInfile = "/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_mit_w85_lounge1~wg_lounge1_1/demon_prediction/images_demon/dense/1/kevin_exhaustive_demon.h5"
recondir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_mit_w85_lounge1~wg_lounge1_1/demon_prediction/images_demon/dense/1'


cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
# print(images)
# views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))

####### add the source data from SUN3D Ground Truth
inputSUN3D_trainFilePaths = []
inputSUN3D_trainFilePaths.append('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.01m_to_0.1m.h5')
inputSUN3D_trainFilePaths.append('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m.h5')
inputSUN3D_trainFilePaths.append('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.2m_to_0.4m.h5')
inputSUN3D_trainFilePaths.append('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.4m_to_0.8m.h5')
inputSUN3D_trainFilePaths.append('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.8m_to_1.6m.h5')
inputSUN3D_trainFilePaths.append('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_1.6m_to_infm.h5')


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

def visMultiViewsPointCloudInGlobalFrame(rendererNotUsed, alpha, image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource='Theia', DepthSource='DeMoN', initBool=True, setColmapGTRatio=False):
    global pointclouds_beforefiltering, initColmapGTRatio, renderer, appendFilterPC, appendFilterModel, curIteration, image_pairs, scaleRecordMat, tmpFittingCoef_Colmap_GT
    data = h5py.File(infile)
    dataExhaustivePairs = h5py.File(ExhaustivePairInfile)

    ######### Clean all actors viz in the renderer before
    if initBool == False:
        renderer.SetBackground(0, 0, 0)
        #renderer.RemoveAllViewProps()
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

    if initBool == False:
        # print("before: inputs in appendFilterPC = ", appendFilterPC.GetInput())
        for removeIdx in range(curIteration*2):
            appendFilterPC.RemoveInputData(appendFilterPC.GetInput(removeIdx))
            appendFilterModel.RemoveInputData(appendFilterModel.GetInput(removeIdx))
        # print("after: inputs in appendFilterPC = ", appendFilterPC.GetInput())
        # print("no. of inputs in appendFilterPC = ", len(appendFilterPC.GetInput()))
        # print("number of items in appendFilterPC = ", appendFilterPC.GetNumberOfItems())

    if initBool == False:
        image_pairs.clear()
        print("image_pairs is cleared!")

    # appendFilterPC = vtk.vtkAppendPolyData()
    # appendFilterModel = vtk.vtkAppendPolyData()

    # image_pairs = set()
    # it = 0
    it = curIteration

    # for image_pair12 in (data.keys()):
    # print(list(data.keys()))
    candidate_pool = []
    if initBool==False:
        print("it = ", it)
        for i in range(it):
            candidate_pool.append(list(One2MultiImagePairs_DeMoN.keys())[i])
        print("recalculate all trajectory after adjusting alpha; the length of candidate_pool is ", len(candidate_pool))
    else:
        candidate_pool.append(list(One2MultiImagePairs_DeMoN.keys())[it])


    # pointclouds_beforefiltering = {}

    for image_pair12 in candidate_pool:
        print("Processing", image_pair12)

        image_name1, image_name2 = image_pair12.split("---")
        image_pair21 = "{}---{}".format(image_name2, image_name1)
        # print(image_name1, "; ", image_name2)

        if image_pair12 in image_pairs:
            if initBool==False:
                continue
            print("Warning: a pair is skipped because its view0 has been processed!")
            curIteration += 1
            return

        # print("view1 = ", view1)
        image_pairs.add(image_pair12)
        # image_pairs.add(image_pair21)

        ##### compute scales
        transScaleTheia = 0
        transScaleColmap = One2MultiImagePairs_Colmap[image_pair12].scale12
        transScaleGT = One2MultiImagePairs_GT[image_pair12].scale12
        correctionScaleColmap = One2MultiImagePairs_correctionColmap[image_pair12].scale12
        correctionScaleGT = One2MultiImagePairs_correctionGT[image_pair12].scale12
        print("transScaleTheia = ", transScaleTheia, "; transScaleColmap = ", transScaleColmap, "; transScaleGT = ", transScaleGT, "; demon scale = ",  One2MultiImagePairs_DeMoN[image_pair12].scale12)
        pred_scale =  One2MultiImagePairs_DeMoN[image_pair12].scale12

        if setColmapGTRatio==True:
            initColmapGTRatio = transScaleColmap/transScaleGT
        if initBool == True:
            if it==0:
                scaleRecordMat = np.array([pred_scale, transScaleTheia, transScaleColmap, transScaleGT])
                initColmapGTRatio = transScaleColmap/transScaleGT
            else:
                scaleRecordMat = np.vstack((scaleRecordMat, np.array([pred_scale, transScaleTheia, transScaleColmap, transScaleGT])))
        print("scaleRecordMat.shape = ", scaleRecordMat.shape)


        # if PoseSource=='Theia':
        #     GlobalExtrinsics1_4by4 = TheiaExtrinsics1_4by4
        #     GlobalExtrinsics2_4by4 = TheiaExtrinsics2_4by4
        if PoseSource=='Colmap':
            GlobalExtrinsics1_4by4 = One2MultiImagePairs_Colmap[image_pair12].Extrinsic1_4by4
            GlobalExtrinsics2_4by4 = One2MultiImagePairs_Colmap[image_pair12].Extrinsic2_4by4
        if PoseSource=='GT':
            GlobalExtrinsics1_4by4 = One2MultiImagePairs_GT[image_pair12].Extrinsic1_4by4
            GlobalExtrinsics2_4by4 = One2MultiImagePairs_GT[image_pair12].Extrinsic2_4by4

        # ###### scale global poses by a constant (Colmap and Theia may generate 3D reconstruction in different scales, which may differ from the real object depth scale)
        # # alpha = 0.28 # 0.3 0.5
        # GlobalExtrinsics1_4by4[0:3,3] = alpha * GlobalExtrinsics1_4by4[0:3,3]
        # GlobalExtrinsics2_4by4[0:3,3] = alpha * GlobalExtrinsics2_4by4[0:3,3]

        ###### get the first point clouds
        input_data = prepare_input_data(One2MultiImagePairs_GT[image_pair12].image1, One2MultiImagePairs_GT[image_pair12].image2, data_format)
        if DepthSource=='Colmap':
            if PoseSource=='Theia':
                scale_applied = transScaleTheia/transScaleColmap
            if PoseSource=='Colmap':
                scale_applied = 1
            if PoseSource=='GT':
                # scale_applied = transScaleGT/transScaleColmap
                # scale_applied = 1/initColmapGTRatio
                # scale_applied = 1/1.72921055    # fittedColmapGTRatio = 1.72921055
                scale_applied = correctionScaleGT/correctionScaleColmap
            # tmp_PointCloud1 = visualize_prediction(
            tmp_PointCloud1 = organize_data_for_noise_removal_stage1(
                        inverse_depth=1/One2MultiImagePairs_Colmap[image_pair12].depth1,
                        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                        image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                        R1=GlobalExtrinsics1_4by4[0:3,0:3],
                        t1=GlobalExtrinsics1_4by4[0:3,3],
                        # rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                        # translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                        scale=scale_applied)

        elif DepthSource=='DeMoN':
            if PoseSource=='Theia':
                scale_applied = transScaleTheia
            if PoseSource=='Colmap':
                # scale_applied = transScaleColmap
                # scale_applied = pred_scale
                # scale_applied = 1
                scale_applied = correctionScaleColmap
            if PoseSource=='GT':
                # scale_applied = transScaleGT
                # scale_applied = 1/pred_scale
                # scale_applied = pred_scale
                # scale_applied = 1
                scale_applied = correctionScaleGT

            # normal_map = compute_normals_from_depth(One2MultiImagePairs_DeMoN[image_pair12].depth1*scale_applied*100)
            normal_map = compute_normals_from_depth(One2MultiImagePairs_DeMoN[image_pair12].depth1*scale_applied)
            normal_map = np.transpose(normal_map, (2, 0, 1))
            # tmp_PointCloud1 = visualize_prediction(
            tmp_PointCloud1 = organize_data_for_noise_removal_stage1(
                        inverse_depth=1/One2MultiImagePairs_DeMoN[image_pair12].depth1,### in previous data retrieval part, the predicted inv_depth has been inversed to depth for later access!
                        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                        image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                        R1=GlobalExtrinsics1_4by4[0:3,0:3],
                        t1=GlobalExtrinsics1_4by4[0:3,3],
                        normals=normal_map,
                        # rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                        # translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
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
            # tmp_PointCloud1 = visualize_prediction(
            tmp_PointCloud1 = organize_data_for_noise_removal_stage1(
                        inverse_depth=1/One2MultiImagePairs_GT[image_pair12].depth1,
                        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                        image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                        R1=GlobalExtrinsics1_4by4[0:3,0:3],
                        t1=GlobalExtrinsics1_4by4[0:3,3],
                        # rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                        # translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
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

        # if it==0:
        #     PointClouds = tmp_PointCloud1
        # else:
        #     PointClouds['points'] = np.concatenate((PointClouds['points'],tmp_PointCloud1['points']), axis=0)
        #     PointClouds['colors'] = np.concatenate((PointClouds['colors'],tmp_PointCloud1['colors']), axis=0)

        cam1_actor = create_camera_actor(GlobalExtrinsics1_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,3])
        # cam1_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        renderer.AddActor(cam1_actor)
        cam1_polydata = create_camera_polydata(GlobalExtrinsics1_4by4[0:3,0:3],GlobalExtrinsics1_4by4[0:3,3], True)
        appendFilterModel.AddInputData(cam1_polydata)

        renderer.Modified()

        pointclouds_beforefiltering[image_pair12] = tmp_PointCloud1


        if True:   # debug: if the second cam is added for visualization
            ##### compute scales
            transScaleTheia = 0
            transScaleColmap = One2MultiImagePairs_Colmap[image_pair12].scale21
            transScaleGT = One2MultiImagePairs_GT[image_pair12].scale21
            correctionScaleColmap = One2MultiImagePairs_correctionColmap[image_pair12].scale21
            correctionScaleGT = One2MultiImagePairs_correctionGT[image_pair12].scale21
            print("transScaleTheia = ", transScaleTheia, "; transScaleColmap = ", transScaleColmap, "; transScaleGT = ", transScaleGT, "; demon scale = ",  One2MultiImagePairs_DeMoN[image_pair12].scale21)
            pred_scale =  One2MultiImagePairs_DeMoN[image_pair12].scale21

            ###### get the second point clouds
            input_data = prepare_input_data(One2MultiImagePairs_GT[image_pair12].image2, One2MultiImagePairs_GT[image_pair12].image1, data_format)
            if DepthSource=='Colmap':
                if PoseSource=='Theia':
                    scale_applied = transScaleTheia/transScaleColmap
                if PoseSource=='Colmap':
                    scale_applied = 1
                if PoseSource=='GT':
                    # scale_applied = transScaleGT/transScaleColmap
                    # scale_applied = 1/initColmapGTRatio
                    # scale_applied = 1/1.72921055    # fittedColmapGTRatio = 1.72921055
                    scale_applied = correctionScaleGT/correctionScaleColmap
                # tmp_PointCloud2 = visualize_prediction(
                tmp_PointCloud2 = organize_data_for_noise_removal_stage1(
                            inverse_depth=1/One2MultiImagePairs_Colmap[image_pair12].depth2,
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics2_4by4[0:3,0:3],
                            t1=GlobalExtrinsics2_4by4[0:3,3],
                            # rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                            # translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                            scale=scale_applied)

            elif DepthSource=='DeMoN':
                if PoseSource=='Theia':
                    scale_applied = transScaleTheia
                if PoseSource=='Colmap':
                    # scale_applied = transScaleColmap
                    # scale_applied = pred_scale
                    # scale_applied = 1
                    scale_applied = correctionScaleColmap
                if PoseSource=='GT':
                    # scale_applied = transScaleGT
                    # scale_applied = 1/pred_scale
                    # scale_applied = pred_scale
                    # scale_applied = 1
                    scale_applied = correctionScaleGT

                # normal_map = compute_normals_from_depth(One2MultiImagePairs_DeMoN[image_pair12].depth1*scale_applied*100)
                normal_map = compute_normals_from_depth(One2MultiImagePairs_DeMoN[image_pair12].depth2*scale_applied)
                # print("normal_map = ", normal_map)
                # return
                normal_map = np.transpose(normal_map, (2, 0, 1))
                # tmp_PointCloud2 = visualize_prediction(
                tmp_PointCloud2 = organize_data_for_noise_removal_stage1(
                            inverse_depth=1/One2MultiImagePairs_DeMoN[image_pair12].depth2,### in previous data retrieval part, the predicted inv_depth has been inversed to depth for later access!
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics2_4by4[0:3,0:3],
                            t1=GlobalExtrinsics2_4by4[0:3,3],
                            normals=normal_map,
                            # rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics1_4by4[0:3,0:3], GlobalExtrinsics2_4by4[0:3,0:3].T)),
                            # translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
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
                # tmp_PointCloud2 = visualize_prediction(
                tmp_PointCloud2 = organize_data_for_noise_removal_stage1(
                            inverse_depth=1/One2MultiImagePairs_GT[image_pair12].depth2,
                            intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                            image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                            R1=GlobalExtrinsics2_4by4[0:3,0:3],
                            t1=GlobalExtrinsics2_4by4[0:3,3],
                            # rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                            # translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                            scale=scale_applied)

            pointcloud_actor = create_pointcloud_actor(
               points=tmp_PointCloud2['points'],
               colors=tmp_PointCloud2['colors'] if 'colors' in tmp_PointCloud1 else None,
               )
            renderer.AddActor(pointcloud_actor)

            pc_polydata = create_pointcloud_polydata(
                                                    points=tmp_PointCloud2['points'],
                                                    colors=tmp_PointCloud2['colors'] if 'colors' in tmp_PointCloud2 else None,
                                                    )
            appendFilterPC.AddInputData(pc_polydata)

            # if it==0:
            #     PointClouds = tmp_PointCloud2
            # else:
            #     PointClouds['points'] = np.concatenate((PointClouds['points'],tmp_PointCloud2['points']), axis=0)
            #     PointClouds['colors'] = np.concatenate((PointClouds['colors'],tmp_PointCloud2['colors']), axis=0)

            cam2_actor = create_camera_actor(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics2_4by4[0:3,3])
            # cam2_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
            renderer.AddActor(cam2_actor)
            cam2_polydata = create_camera_polydata(GlobalExtrinsics2_4by4[0:3,0:3],GlobalExtrinsics2_4by4[0:3,3], True)
            appendFilterModel.AddInputData(cam2_polydata)

            renderer.Modified()

            pointclouds_beforefiltering[image_pair21] = tmp_PointCloud2


        it +=1
        if it>=len(One2MultiImagePairs_DeMoN.keys()):
            print("Warning: No more views(image pairs) can be added for visualization!")
            # break


    appendFilterPC.Update()
    renderer.Modified()
    # ###### Compute the slope of the fitted line to reflect the scale differences among DeMoN, Theia and Colmap
    # tmpFittingCoef_DeMoNTheia = np.polyfit(scaleRecordMat[:,0], scaleRecordMat[:,1], 1)
    # print("tmpFittingCoef_DeMoNTheia = ", tmpFittingCoef_DeMoNTheia)
    # tmpFittingCoef_DeMoNColmap = np.polyfit(scaleRecordMat[:,0], scaleRecordMat[:,2], 1)
    # print("tmpFittingCoef_DeMoNColmap = ", tmpFittingCoef_DeMoNColmap)
    # tmpFittingCoef_TheiaColmap = np.polyfit(scaleRecordMat[:,1], scaleRecordMat[:,2], 1)
    # print("tmpFittingCoef_TheiaColmap = ", tmpFittingCoef_TheiaColmap)
    if it > 1:
        tmpFittingCoef_Colmap_GT = np.polyfit(scaleRecordMat[:,3], scaleRecordMat[:,2], 1)
    print("tmpFittingCoef_Colmap_GT = ", tmpFittingCoef_Colmap_GT)
    # plot the scatter 2D data of scale records, to find out the correlation between the predicted scales and the calculated scales from global SfM
    np.savetxt(os.path.join(outdir,'scale_record_DeMoN_Theia_Colmap.txt'), scaleRecordMat, fmt='%f')
    if scaleRecordMat.ndim>1:
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
        if False:
            plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,3])
            plt.ylabel('scales calculated from SUN3D Ground Truth')
            plt.xlabel('scales predicted by DeMoN')
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
            # # dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
            # y = 1.72921055*x+0.02395182
            # plt.plot(x, y, 'r-', lw=2)
            # plt.text(0.2, 0.15, r'fitted line with slope = 1.72921055')
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


    appendFilterModel.AddInputData(appendFilterPC.GetOutput())
    appendFilterModel.Update()

    plywriterModel = vtk.vtkPLYWriter()
    plywriterModel.SetFileName(os.path.join(outdir,'One2Multi_point_clouds_iteration{0}_.ply'.format(int(curIteration))))
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
    if initBool==True:
        curIteration += 1


# TheiaOrColmapOrGTPoses='Colmap'
# TheiaOrColmapOrGTPoses='Theia'
TheiaOrColmapOrGTPoses='GT'
# DeMoNOrColmapOrGTDepths='GT'
# DeMoNOrColmapOrGTDepths='Colmap'
DeMoNOrColmapOrGTDepths='DeMoN'

tarImageFileName = 'mit_w85_lounge1~wg_lounge1_1-0000055_baseline_2_v1.JPG'
# tarImageFileName = 'hotel_beijing~beijing_hotel_2-0000103_baseline_1_v0.JPG'

renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)
interactor = vtk.vtkRenderWindowInteractor()

curIteration = 0
image_pairs = set()
# scaleRecordMat = []
scaleRecordMat = np.empty((1,4))
tmpFittingCoef_Colmap_GT = 0
alpha = 1.0
appendFilterPC = vtk.vtkAppendPolyData()
appendFilterModel = vtk.vtkAppendPolyData()
initColmapGTRatio = 0

class MyKeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    global pointclouds_beforefiltering, initColmapGTRatio, appendFilterPC, appendFilterModel, alpha, tmpFittingCoef_Colmap_GT, scaleRecordMat, image_pairs, TheiaOrColmapOrGTPoses, DeMoNOrColmapOrGTDepths, sliderMin, sliderMax, interactor, renderer, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT

    def __init__(self,parent=None):
        self.parent = interactor

        self.AddObserver("KeyPressEvent",self.keyPressEvent)

    def keyPressEvent(self,obj,event):
        key = self.parent.GetKeySym()
        if key == 'n':
            print("Key n is pressed: one more view will be added!")
            visMultiViewsPointCloudInGlobalFrame(renderer, alpha, image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, initBool=True)
            renderer.Modified()
        return


# One2MultiImagePair = collections.namedtuple("One2MultiImagePair", ["name1", "name2", "image1", "image2", "depth1", "depth2", "Extrinsic1_4by4_Colmap", "Extrinsic2_4by4_Colmap", "Extrinsic1_4by4_GT", "Extrinsic2_4by4_GT", "Extrinsic1_4by4_DeMoN", "Extrinsic2_4by4_DeMoN", "Relative12_4by4_Colmap", "Relative12_4by4_GT", "Relative12_4by4_DeMoN", "scale_Colmap", "scale_GT", "scale_DeMoN"])
One2MultiImagePair = collections.namedtuple("One2MultiImagePair", ["name1", "name2", "image1", "image2", "depth1", "depth2", "Extrinsic1_4by4", "Extrinsic2_4by4", "Relative12_4by4", "Relative21_4by4", "scale12", "scale21"])

def findMultiPairs(infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, image1_filename='hotel_beijing~beijing_hotel_2-0000181_baseline_1_v0.JPG'):
    # global initColmapGTRatio, renderer, appendFilterPC, appendFilterModel, curIteration, image_pairs, scaleRecordMat, tmpFittingCoef_Colmap_GT

    One2MultiImagePairs_Colmap = {}
    One2MultiImagePairs_correctionColmap = {}
    One2MultiImagePairs_GT = {}
    One2MultiImagePairs_correctionGT = {}
    One2MultiImagePairs_DeMoN = {}

    data = h5py.File(infile)
    dataExhaustivePairs = h5py.File(ExhaustivePairInfile)

    image_pairs_One2Multi = set()
    it = 0
    # it = curIteration

    inlierfile = open(os.path.join(outdir, "inlier_image_pairs.txt"), "w")
    outlierfile = open(os.path.join(outdir, "outlier_image_pairs.txt"), "w")

    for image_pair12 in (data.keys()):
        image_name1, image_name2 = image_pair12.split("---")
        if image_name1 != image1_filename:
            continue
        print("add ", image_pair12, " to the candidate pool!")

        image_pair21 = "{}---{}".format(image_name2, image_name1)
        print(image_name1, "; ", image_name2)
        if image_pair21 not in data.keys():
            continue

        tmp_dict = {}
        for image_id, image in images.items():
            # print(image.name, "; ", image_name1, "; ", image_name2)
            if image.name == image_name1:
                tmp_dict[image_id] = image

        # tmp_dict = {image_id: image}
        print("tmp_dict = ", tmp_dict)
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
        print("tmp_dict = ", tmp_dict)
        if len(tmp_dict)<1:
            continue
        tmp_views = colmap.create_views(cameras, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
        # print("tmp_views = ", tmp_views)
        tmp_views[0] = adjust_intrinsics(tmp_views[0], target_K, w, h,)
        view2 = tmp_views[0]

        ###########################################################################################

        # print("view1 = ", view1)
        image_pairs_One2Multi.add(image_pair12)
        # image_pairs_One2Multi.add(image_pair21)

        ###### Retrieve Theia Global poses for image 1 and 2
        TheiaExtrinsics1_4by4 = np.eye(4)
        TheiaExtrinsics2_4by4 = np.eye(4)
        for ids,val in TheiaGlobalPosesGT.items():
            if val.name == image_name1:
                TheiaExtrinsics1_4by4[0:3,0:3] = val.rotmat
                TheiaExtrinsics1_4by4[0:3,3] = -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t
            if val.name == image_name2:
                TheiaExtrinsics2_4by4[0:3,0:3] = val.rotmat
                TheiaExtrinsics2_4by4[0:3,3] = -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        # # a colormap and a normalization instance
        # cmap = plt.cm.jet
        # # plt.imshow(data[image_pair12]["depth_upsampled"], cmap=plt.get_cmap('gray'))
        # plt.imshow(view1.depth, cmap=plt.get_cmap('gray'))
        ColmapExtrinsics1_4by4 = np.eye(4)
        ColmapExtrinsics1_4by4[0:3,0:3] = view1.R
        ColmapExtrinsics1_4by4[0:3,3] = view1.t# -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        ColmapExtrinsics2_4by4 = np.eye(4)
        ColmapExtrinsics2_4by4[0:3,0:3] = view2.R
        ColmapExtrinsics2_4by4[0:3,3] = view2.t # -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        ###### Retrieve Ground Truth Global poses for image 1 and 2
        BaselineRange1 = image_name1[-8:-7]
        tmpName1 = image_name1[:-18].split('~')
        name1InSUN3D_H5 = tmpName1[0]+'.'+tmpName1[1]
        vId1 = image_name1[-6:-4]
        # print(BaselineRange1, name1InSUN3D_H5, vId1)
        name1InSUN3D_H5 = name1InSUN3D_H5+'/frames/t0/' +vId1
        # print(name1InSUN3D_H5)
        # print(inputSUN3D_trainFilePaths[0])
        dataGT1 = h5py.File(inputSUN3D_trainFilePaths[int(BaselineRange1)])
        GTExtrinsics1_4by4 = np.eye(4)
        K1GT, GTExtrinsics1_4by4[0:3,0:3], GTExtrinsics1_4by4[0:3,3] = read_camera_params(dataGT1[name1InSUN3D_H5]['camera'])
        tmp_view1 = read_view(dataGT1[name1InSUN3D_H5])
        view1GT = adjust_intrinsics(tmp_view1, target_K, w, h,)

        BaselineRange2 = image_name2[-8:-7]
        tmpName2 = image_name2[:-18].split('~')
        name2InSUN3D_H5 = tmpName2[0]+'.'+tmpName2[1]
        vId2 = image_name2[-6:-4]
        print(BaselineRange2, name2InSUN3D_H5, vId2)
        name2InSUN3D_H5 = name2InSUN3D_H5+'/frames/t0/' +vId2
        print(name2InSUN3D_H5)
        dataGT2 = h5py.File(inputSUN3D_trainFilePaths[int(BaselineRange2)])
        GTExtrinsics2_4by4 = np.eye(4)
        K2GT, GTExtrinsics2_4by4[0:3,0:3], GTExtrinsics2_4by4[0:3,3] = read_camera_params(dataGT2[name2InSUN3D_H5]['camera'])
        tmp_view2 = read_view(dataGT2[name2InSUN3D_H5])
        view2GT = adjust_intrinsics(tmp_view2, target_K, w, h,)

        # # print(view1GT.depth)
        # fig, ax = plt.subplots()
        # cax = ax.imshow(view1GT.depth[0:3,0:4], cmap=plt.get_cmap('gray'))
        # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
        # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
        # # plt.imshow(view1GT.depth, cmap=plt.get_cmap('gray'))
        # plt.show()
        # plt.imshow(view1GT.depth, cmap=plt.get_cmap('gray'))
        # plt.show()
        # return

        ##### compute scales
        correctionScaleGT12 = computeCorrectionScale(data[image_pair12]['depth_upsampled'].value, view1GT.depth, 60)
        correctionScaleColmap12 = computeCorrectionScale(data[image_pair12]['depth_upsampled'].value, view1.depth, 60)
        correctionScaleGT21 = computeCorrectionScale(data[image_pair21]['depth_upsampled'].value, view2GT.depth, 60)
        correctionScaleColmap21 = computeCorrectionScale(data[image_pair21]['depth_upsampled'].value, view2.depth, 60)
        transScaleTheia = np.linalg.norm(np.linalg.inv(TheiaExtrinsics2_4by4)[0:3,3] - np.linalg.inv(TheiaExtrinsics1_4by4)[0:3,3])
        transScaleColmap = np.linalg.norm(np.linalg.inv(ColmapExtrinsics2_4by4)[0:3,3] - np.linalg.inv(ColmapExtrinsics1_4by4)[0:3,3])
        transScaleGT = np.linalg.norm(np.linalg.inv(GTExtrinsics2_4by4)[0:3,3] - np.linalg.inv(GTExtrinsics1_4by4)[0:3,3])
        print("transScaleTheia = ", transScaleTheia, "; transScaleColmap = ", transScaleColmap, "; transScaleGT = ", transScaleGT, "; demon scale = ", data[image_pair12]['scale'].value)
        pred_scale = data[image_pair12]['scale'].value

        GTbaselineLength = np.linalg.norm(-np.dot(view2GT.R.T, view2GT.t)+np.dot(view1GT.R.T, view1GT.t))
        # if computePoint2LineDist(np.array([correctionScaleGT,transScaleGT]))>0.010:
        #     outlierfile.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(image_pair12, GTbaselineLength, pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap))
        #     continue
        # inlierfile.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(image_pair12, GTbaselineLength, pred_scale, transScaleTheia, transScaleColmap, transScaleGT, correctionScaleGT, correctionScaleColmap))

        ###### record data in corresponding data structure for later access
        One2MultiImagePairs_Colmap[image_pair12] = One2MultiImagePair(name1=image_name1, name2=image_name2, image1=view1.image, image2=view2.image, depth1=view1.depth, depth2=view2.depth, Extrinsic1_4by4=ColmapExtrinsics1_4by4, Extrinsic2_4by4=ColmapExtrinsics2_4by4, Relative12_4by4=np.dot(ColmapExtrinsics2_4by4, np.linalg.inv(ColmapExtrinsics1_4by4)), Relative21_4by4=np.dot(ColmapExtrinsics1_4by4, np.linalg.inv(ColmapExtrinsics2_4by4)), scale12=transScaleColmap, scale21=transScaleColmap)
        One2MultiImagePairs_correctionColmap[image_pair12] = One2MultiImagePair(name1=image_name1, name2=image_name2, image1=view1.image, image2=view2.image, depth1=view1.depth, depth2=view2.depth, Extrinsic1_4by4=ColmapExtrinsics1_4by4, Extrinsic2_4by4=ColmapExtrinsics2_4by4, Relative12_4by4=np.dot(ColmapExtrinsics2_4by4, np.linalg.inv(ColmapExtrinsics1_4by4)), Relative21_4by4=np.dot(ColmapExtrinsics1_4by4, np.linalg.inv(ColmapExtrinsics2_4by4)), scale12=correctionScaleColmap12, scale21=correctionScaleColmap21)
        One2MultiImagePairs_GT[image_pair12] = One2MultiImagePair(name1=image_name1, name2=image_name2, image1=view1GT.image, image2=view2GT.image, depth1=view1GT.depth, depth2=view2GT.depth, Extrinsic1_4by4=GTExtrinsics1_4by4, Extrinsic2_4by4=GTExtrinsics2_4by4, Relative12_4by4=np.dot(GTExtrinsics2_4by4, np.linalg.inv(GTExtrinsics1_4by4)), Relative21_4by4=np.dot(GTExtrinsics1_4by4, np.linalg.inv(GTExtrinsics2_4by4)), scale12=transScaleGT, scale21=transScaleGT)
        One2MultiImagePairs_correctionGT[image_pair12] = One2MultiImagePair(name1=image_name1, name2=image_name2, image1=view1GT.image, image2=view2GT.image, depth1=view1GT.depth, depth2=view2GT.depth, Extrinsic1_4by4=GTExtrinsics1_4by4, Extrinsic2_4by4=GTExtrinsics2_4by4, Relative12_4by4=np.dot(GTExtrinsics2_4by4, np.linalg.inv(GTExtrinsics1_4by4)), Relative21_4by4=np.dot(GTExtrinsics1_4by4, np.linalg.inv(GTExtrinsics2_4by4)), scale12=correctionScaleGT12, scale21=correctionScaleGT21)
        DeMoNRelative12_4by4 = np.eye(4)
        DeMoNRelative12_4by4[0:3,0:3] = data[image_pair12]['rotation'].value
        DeMoNRelative12_4by4[0:3,3] = data[image_pair12]['translation'].value
        One2MultiImagePairs_DeMoN[image_pair12] = One2MultiImagePair(name1=image_name1, name2=image_name2, image1=view1.image, image2=view2.image, depth1=1/data[image_pair12]['depth_upsampled'].value, depth2=1/data[image_pair21]['depth_upsampled'].value, Extrinsic1_4by4=np.eye(4), Extrinsic2_4by4=DeMoNRelative12_4by4, Relative12_4by4=DeMoNRelative12_4by4, Relative21_4by4=np.linalg.inv(DeMoNRelative12_4by4), scale12=data[image_pair12]['scale'].value, scale21=data[image_pair21]['scale'].value)

        ###### record reverse pair data in corresponding data structure for later access
        One2MultiImagePairs_Colmap[image_pair21] = One2MultiImagePair(name1=image_name2, name2=image_name1, image1=view2.image, image2=view1.image, depth1=view2.depth, depth2=view1.depth, Extrinsic1_4by4=ColmapExtrinsics2_4by4, Extrinsic2_4by4=ColmapExtrinsics1_4by4, Relative12_4by4=np.dot(ColmapExtrinsics1_4by4, np.linalg.inv(ColmapExtrinsics2_4by4)), Relative21_4by4=np.dot(ColmapExtrinsics2_4by4, np.linalg.inv(ColmapExtrinsics1_4by4)), scale12=transScaleColmap, scale21=transScaleColmap)
        One2MultiImagePairs_correctionColmap[image_pair21] = One2MultiImagePair(name1=image_name2, name2=image_name1, image1=view2.image, image2=view1.image, depth1=view2.depth, depth2=view1.depth, Extrinsic1_4by4=ColmapExtrinsics2_4by4, Extrinsic2_4by4=ColmapExtrinsics1_4by4, Relative12_4by4=np.dot(ColmapExtrinsics1_4by4, np.linalg.inv(ColmapExtrinsics2_4by4)), Relative21_4by4=np.dot(ColmapExtrinsics2_4by4, np.linalg.inv(ColmapExtrinsics1_4by4)), scale12=correctionScaleColmap21, scale21=correctionScaleColmap12)
        One2MultiImagePairs_GT[image_pair21] = One2MultiImagePair(name1=image_name2, name2=image_name1, image1=view2GT.image, image2=view1GT.image, depth1=view2GT.depth, depth2=view1GT.depth, Extrinsic1_4by4=GTExtrinsics2_4by4, Extrinsic2_4by4=GTExtrinsics1_4by4, Relative12_4by4=np.dot(GTExtrinsics1_4by4, np.linalg.inv(GTExtrinsics2_4by4)), Relative21_4by4=np.dot(GTExtrinsics2_4by4, np.linalg.inv(GTExtrinsics1_4by4)), scale12=transScaleGT, scale21=transScaleGT)
        One2MultiImagePairs_correctionGT[image_pair21] = One2MultiImagePair(name1=image_name2, name2=image_name1, image1=view2GT.image, image2=view1GT.image, depth1=view2GT.depth, depth2=view1GT.depth, Extrinsic1_4by4=GTExtrinsics2_4by4, Extrinsic2_4by4=GTExtrinsics1_4by4, Relative12_4by4=np.dot(GTExtrinsics1_4by4, np.linalg.inv(GTExtrinsics2_4by4)), Relative21_4by4=np.dot(GTExtrinsics2_4by4, np.linalg.inv(GTExtrinsics1_4by4)), scale12=correctionScaleGT21, scale21=correctionScaleGT12)
        DeMoNRelative12_4by4 = np.eye(4)
        DeMoNRelative12_4by4[0:3,0:3] = data[image_pair12]['rotation'].value
        DeMoNRelative12_4by4[0:3,3] = data[image_pair12]['translation'].value
        One2MultiImagePairs_DeMoN[image_pair21] = One2MultiImagePair(name1=image_name2, name2=image_name1, image1=view2.image, image2=view1.image, depth1=1/data[image_pair21]['depth_upsampled'].value, depth2=1/data[image_pair12]['depth_upsampled'].value, Extrinsic1_4by4=np.eye(4), Extrinsic2_4by4=np.linalg.inv(DeMoNRelative12_4by4), Relative12_4by4=np.linalg.inv(DeMoNRelative12_4by4), Relative21_4by4=DeMoNRelative12_4by4, scale12=data[image_pair21]['scale'].value, scale21=data[image_pair12]['scale'].value)


    print("Colmap image pairs retrieved = ", (One2MultiImagePairs_Colmap))
    print("GT image pairs retrieved = ", (One2MultiImagePairs_GT))
    print("correctionColmap image pairs retrieved = ", (One2MultiImagePairs_correctionColmap))
    print("correctionGT image pairs retrieved = ", (One2MultiImagePairs_correctionGT))
    print("DeMoN image pairs retrieved = ", (One2MultiImagePairs_DeMoN))
    print("number of image pairs retrieved = ", len(image_pairs_One2Multi))
    print("number of image pairs retrieved = ", len(One2MultiImagePairs_Colmap))
    print("number of image pairs retrieved = ", len(One2MultiImagePairs_GT))
    print("number of image pairs retrieved = ", len(One2MultiImagePairs_DeMoN))

    inlierfile.close()
    print("inlier matches num = ", it)
    outlierfile.close()

    if False:
        if len(image_pairs_One2Multi) >= 5:
            print("pair 1 (img1+img2): DeMoN scale12 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[0]].scale12, ", while DeMoN scale21 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[0]].scale21)
            print("pair 1 (img1+img2): GT scale12 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[0]].scale12, ", while GT scale21 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[0]].scale21)
            print("pair 1 (img1+img2): Colmap scale12 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[0]].scale12, ", while Colmap scale21 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[0]].scale21)
            print("pair 1 (img1+img2): correctionGT scale12 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[0]].scale12, ", while correctionGT scale21 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[0]].scale21)
            print("pair 1 (img1+img2): correctionColmap scale12 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[0]].scale12, ", while correctionColmap scale21 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[0]].scale21)

            print("pair 2 (img1+img3): DeMoN scale12 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[1]].scale12, ", while DeMoN scale21 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[1]].scale21)
            print("pair 2 (img1+img3): GT scale12 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[1]].scale12, ", while GT scale21 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[1]].scale21)
            print("pair 2 (img1+img3): Colmap scale12 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[1]].scale12, ", while Colmap scale21 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[1]].scale21)
            print("pair 2 (img1+img3): correctionGT scale12 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[1]].scale12, ", while correctionGT scale21 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[1]].scale21)
            print("pair 2 (img1+img3): correctionColmap scale12 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[1]].scale12, ", while correctionColmap scale21 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[1]].scale21)

            print("pair 3 (img1+img4): DeMoN scale12 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[2]].scale12, ", while DeMoN scale21 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[2]].scale21)
            print("pair 3 (img1+img4): GT scale12 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[2]].scale12, ", while GT scale21 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[2]].scale21)
            print("pair 3 (img1+img4): Colmap scale12 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[2]].scale12, ", while Colmap scale21 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[2]].scale21)
            print("pair 3 (img1+img4): correctionGT scale12 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[2]].scale12, ", while correctionGT scale21 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[2]].scale21)
            print("pair 3 (img1+img4): correctionColmap scale12 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[2]].scale12, ", while correctionColmap scale21 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[2]].scale21)

            print("pair 4 (img1+img5): DeMoN scale12 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[3]].scale12, ", while DeMoN scale21 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[3]].scale21)
            print("pair 4 (img1+img5): GT scale12 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[3]].scale12, ", while GT scale21 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[3]].scale21)
            print("pair 4 (img1+img5): Colmap scale12 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[3]].scale12, ", while Colmap scale21 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[3]].scale21)
            print("pair 4 (img1+img5): correctionGT scale12 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[3]].scale12, ", while correctionGT scale21 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[3]].scale21)
            print("pair 4 (img1+img5): correctionColmap scale12 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[3]].scale12, ", while correctionColmap scale21 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[3]].scale21)

            print("pair 5 (img1+img6): DeMoN scale12 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[4]].scale12, ", while DeMoN scale21 = ", One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[4]].scale21)
            print("pair 5 (img1+img6): GT scale12 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[4]].scale12, ", while GT scale21 = ", One2MultiImagePairs_GT[list(image_pairs_One2Multi)[4]].scale21)
            print("pair 5 (img1+img6): Colmap scale12 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[4]].scale12, ", while Colmap scale21 = ", One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[4]].scale21)
            print("pair 5 (img1+img6): correctionGT scale12 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[4]].scale12, ", while correctionGT scale21 = ", One2MultiImagePairs_correctionGT[list(image_pairs_One2Multi)[4]].scale21)
            print("pair 5 (img1+img6): correctionColmap scale12 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[4]].scale12, ", while correctionColmap scale21 = ", One2MultiImagePairs_correctionColmap[list(image_pairs_One2Multi)[4]].scale21)


            plt.figure()

            plt.subplot(4,5,1) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[0]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,11) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[0]].image2)
            plt.title('RGB Image 2')
            plt.subplot(4,5,6)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[0]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 1')
            plt.subplot(4,5,16)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[0]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 2')

            plt.subplot(4,5,2) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[1]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,12) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[1]].image2)
            plt.title('RGB Image 3')
            plt.subplot(4,5,7)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[1]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 1')
            plt.subplot(4,5,17)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[1]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 2')

            plt.subplot(4,5,3) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[2]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,13) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[2]].image2)
            plt.title('RGB Image 4')
            plt.subplot(4,5,8)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[2]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 1')
            plt.subplot(4,5,18)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[2]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 2')

            plt.subplot(4,5,4) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[3]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,14) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[3]].image2)
            plt.title('RGB Image 5')
            plt.subplot(4,5,9)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[3]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 1')
            plt.subplot(4,5,19)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[3]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 2')

            plt.subplot(4,5,5) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[4]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,15) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[4]].image2)
            plt.title('RGB Image 6')
            plt.subplot(4,5,10)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[4]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 1')
            plt.subplot(4,5,20)
            plt.imshow(One2MultiImagePairs_DeMoN[list(image_pairs_One2Multi)[4]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('DeMoN Depth Image 2')

            plt.show()


            plt.figure()

            plt.subplot(4,5,1) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[0]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,11) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[0]].image2)
            plt.title('RGB Image 2')
            plt.subplot(4,5,6)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[0]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 1')
            plt.subplot(4,5,16)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[0]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 2')

            plt.subplot(4,5,2) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[1]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,12) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[1]].image2)
            plt.title('RGB Image 3')
            plt.subplot(4,5,7)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[1]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 1')
            plt.subplot(4,5,17)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[1]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 2')

            plt.subplot(4,5,3) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[2]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,13) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[2]].image2)
            plt.title('RGB Image 4')
            plt.subplot(4,5,8)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[2]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 1')
            plt.subplot(4,5,18)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[2]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 2')

            plt.subplot(4,5,4) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[3]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,14) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[3]].image2)
            plt.title('RGB Image 5')
            plt.subplot(4,5,9)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[3]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 1')
            plt.subplot(4,5,19)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[3]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 2')

            plt.subplot(4,5,5) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[4]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,15) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[4]].image2)
            plt.title('RGB Image 6')
            plt.subplot(4,5,10)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[4]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 1')
            plt.subplot(4,5,20)
            plt.imshow(One2MultiImagePairs_GT[list(image_pairs_One2Multi)[4]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('GT Depth Image 2')

            plt.show()

            plt.figure()

            plt.subplot(4,5,1) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[0]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,11) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[0]].image2)
            plt.title('RGB Image 2')
            plt.subplot(4,5,6)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[0]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 1')
            plt.subplot(4,5,16)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[0]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 2')

            plt.subplot(4,5,2) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[1]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,12) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[1]].image2)
            plt.title('RGB Image 3')
            plt.subplot(4,5,7)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[1]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 1')
            plt.subplot(4,5,17)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[1]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 2')

            plt.subplot(4,5,3) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[2]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,13) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[2]].image2)
            plt.title('RGB Image 4')
            plt.subplot(4,5,8)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[2]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 1')
            plt.subplot(4,5,18)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[2]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 2')

            plt.subplot(4,5,4) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[3]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,14) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[3]].image2)
            plt.title('RGB Image 5')
            plt.subplot(4,5,9)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[3]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 1')
            plt.subplot(4,5,19)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[3]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 2')

            plt.subplot(4,5,5) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[4]].image1)
            plt.title('RGB Image 1')
            plt.subplot(4,5,15) # equivalent to: plt.subplot(2, 2, 1)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[4]].image2)
            plt.title('RGB Image 6')
            plt.subplot(4,5,10)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[4]].depth1, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 1')
            plt.subplot(4,5,20)
            plt.imshow(One2MultiImagePairs_Colmap[list(image_pairs_One2Multi)[4]].depth2, cmap=plt.get_cmap('gray'))
            plt.title('Colmap Depth Image 2')

            plt.show()

    return image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap



def checkDepthConsistencyPixelwise_One2Multi_OldInefficientNestedLoops(image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, PoseSource='GT', DepthSource='DeMoN', w=256, h=192):
    ###### Check Depth & InvDepth Stats
    pixelwiseStats = np.zeros([h,w,4])
    it = 0
    scaleRecorderMat = np.empty([1,3])
    for row in range(h):
        for col in range(w):
            # tmpDepthRecorder = np.zeros([len(list(One2MultiImagePairs_DeMoN.keys()))])
            tmpDepthRecorder = []
            tmpInvDepthRecorder = []

            # scaleRecorderMat = np.empty([1,3])
            for image_pair12 in One2MultiImagePairs_DeMoN.keys():
                ### check correlation between different scales regarding one-to-multi image pairs
                if it==0:
                    scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale12, One2MultiImagePairs_GT[image_pair12].scale12, One2MultiImagePairs_Colmap[image_pair12].scale12])))
                    scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale21, One2MultiImagePairs_GT[image_pair12].scale21, One2MultiImagePairs_Colmap[image_pair12].scale21])))
                    print("scaleRecorderMat.shape = ", scaleRecorderMat.shape)
                    #np.savetxt(os.path.join(outdir, 'scale_matrix.txt'),scaleRecorderMat)
                    if One2MultiImagePairs_Colmap[image_pair12].scale12 <0:
                        print(One2MultiImagePairs_DeMoN[image_pair12].scale12, " ". One2MultiImagePairs_GT[image_pair12].scale12, " ", One2MultiImagePairs_Colmap[image_pair12].scale12)
                        print(image_pair12, " has negative scale value: figure out the reason")
                        return
                    if One2MultiImagePairs_Colmap[image_pair12].scale21 <0:
                        print(One2MultiImagePairs_DeMoN[image_pair12].scale21, " ". One2MultiImagePairs_GT[image_pair12].scale21, " ", One2MultiImagePairs_Colmap[image_pair12].scale21)
                        print(image_pair21, " has negative scale value: figure out the reason")
                        return
                scale_choice = 1
                if DepthSource=='DeMoN':
                    depth_choice = One2MultiImagePairs_DeMoN[image_pair12].depth1[row,col]
                    if PoseSource=='Colmap':
                        # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale12
                        scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12
                    if PoseSource=='GT':
                        # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale12
                        scale_choice = One2MultiImagePairs_GT[image_pair12].scale12
                if DepthSource=='Colmap':
                    depth_choice = One2MultiImagePairs_Colmap[image_pair12].depth1[row,col]
                    if PoseSource=='GT':
                        scale_choice = One2MultiImagePairs_GT[image_pair12].scale12/One2MultiImagePairs_Colmap[image_pair12].scale12    # or a fixed value
                if DepthSource=='GT':
                    depth_choice = One2MultiImagePairs_GT[image_pair12].depth1[row,col]
                    if PoseSource=='Colmap':
                        scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12/One2MultiImagePairs_GT[image_pair12].scale12    # or a fixed value
                if depth_choice<0.001:
                    continue
                tmpScaledDepthValue = depth_choice * scale_choice
                tmpDepthRecorder.append(tmpScaledDepthValue)
                tmpScaledInvDepthValue = 1/tmpScaledDepthValue
                tmpInvDepthRecorder.append(tmpScaledInvDepthValue)
                # print("scale_choice = ", scale_choice)

            # if it==0:
            #     plt.figure()
            #     plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,1])
            #     plt.xlabel('DeMoN Scale')
            #     plt.ylabel('GT Scale')
            #     plt.axis('equal')
            #     plt.grid(True)
            #     plt.show()
            #     plt.figure()
            #     plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,2])
            #     plt.xlabel('DeMoN Scale')
            #     plt.ylabel('Colmap Scale')
            #     plt.axis('equal')
            #     plt.grid(True)
            #     plt.show()
            #     plt.figure()
            #     plt.scatter(scaleRecorderMat[:,2],scaleRecorderMat[:,1])
            #     plt.xlabel('Colmap Scale')
            #     plt.ylabel('GT Scale')
            #     plt.axis('equal')
            #     plt.grid(True)
            #     plt.show()
            it += 1
            if len(tmpDepthRecorder) == 0:
                pixelwiseStats[row,col,0] = np.nan
                pixelwiseStats[row,col,1] = np.nan
                pixelwiseStats[row,col,2] = np.nan
                pixelwiseStats[row,col,3] = np.nan
            # print("len(tmpDepthRecorder) = ", len(tmpDepthRecorder))
            pixelwiseStats[row,col,0] = np.mean(tmpDepthRecorder)
            pixelwiseStats[row,col,1] = np.std(tmpDepthRecorder)
            pixelwiseStats[row,col,2] = np.mean(tmpInvDepthRecorder)
            pixelwiseStats[row,col,3] = np.std(tmpInvDepthRecorder)
    #print("pixelwiseStats = ", pixelwiseStats)
    scaleRecorderMat = scaleRecorderMat[1:,:]
    print("scaleRecorderMat.shape = ", scaleRecorderMat.shape)
    np.savetxt(os.path.join(outdir, 'scale_matrix.txt'),scaleRecorderMat)
    if False:
        plt.figure()
        plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,1])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,0]),0,1.2*np.max(scaleRecorderMat[:,1])))
        #plt.axis((0,1,0,1.2*np.max(scaleRecorderMat[:,1])))
        plt.xlabel('DeMoN Scale')
        plt.ylabel('GT Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,2])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,0]),0,1.2*np.max(scaleRecorderMat[:,2])))
        #plt.axis((0,1,0,1.2*np.max(scaleRecorderMat[:,2])))
        plt.xlabel('DeMoN Scale')
        plt.ylabel('Colmap Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.scatter(scaleRecorderMat[:,2],scaleRecorderMat[:,1])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,2]),0,1.2*np.max(scaleRecorderMat[:,1])))
        plt.xlabel('Colmap Scale')
        plt.ylabel('GT Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
    # ClosePairs_Depth_DeMoNDepth_GTScale_GTPose_pixelwise_std_CDF.png
    # ClosePairs_Depth_DeMoNDepth_ColmapScale_ColmapPose_pixelwise_std_CDF.png

    data = pixelwiseStats[:,:,1].flatten()
    print(data.shape)
    data = data[data!=np.nan]
    print(data.shape)
    plt.figure()
    num_bins = 2000
    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.title('CDF of pixelwise std of depth values: mu={0}, sigma={1}, from {2} pairs'.format(np.mean(data), np.std(data), int(len(list(One2MultiImagePairs_DeMoN.keys())))))
    plt.xlabel('pixelwise standard deviation of depth values')
    plt.ylabel('Probability')
    plt.show()

    # for q in [50, 90, 95, 100]:
    #     print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
    #
    # plt.figure()
    # hist = plt.hist(data, normed=False, bins=1000)
    # # plt.plot(bin_edges[1:], counts)
    # # plt.bar(bin_edges[1:], height=counts)
    # plt.xlabel('pixelwise standard deviation of depth values')
    # plt.ylabel('counts number')
    # plt.title('Histogram of pixelwise standard deviation of depth values: mu={0}, sigma={1}'.format(np.mean(data), np.std(data)))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.show()
    #######################################################
    data = pixelwiseStats[:,:,3].flatten()
    print(data.shape)
    data = data[data!=np.nan]
    print(data.shape)
    plt.figure()
    num_bins = 2000
    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.title('CDF of pixelwise std of inv_depth values: mu={0}, sigma={1}, from {2} pairs'.format(np.mean(data), np.std(data), int(len(list(One2MultiImagePairs_DeMoN.keys())))))
    plt.xlabel('pixelwise standard deviation of inv_depth values')
    plt.ylabel('Probability')
    plt.show()

    # for q in [50, 90, 95, 100]:
    #     print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
    #
    # plt.figure()
    # hist = plt.hist(data, normed=False, bins=1000)
    # # plt.plot(bin_edges[1:], counts)
    # # plt.bar(bin_edges[1:], height=counts)
    # plt.xlabel('pixelwise standard deviation of inv_depth values')
    # plt.ylabel('counts number')
    # plt.title('Histogram of pixelwise standard deviation of inv_depth values: mu={0}, sigma={1}'.format(np.mean(data), np.std(data)))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.show()


def checkDepthConsistencyPixelwise_One2Multi(image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, PoseSource='GT', DepthSource='DeMoN', w=256, h=192):
    ###### Check Depth & InvDepth Stats
    pixelwiseStats = np.zeros([h*w,4])
    # scaleRecorderMat = np.empty([1,3])
    viewCnt=0
    for image_pair12 in One2MultiImagePairs_DeMoN.keys():
        ### check correlation between different scales regarding one-to-multi image pairs
        if viewCnt==0:
            scaleRecorderMat = np.array([One2MultiImagePairs_DeMoN[image_pair12].scale12, One2MultiImagePairs_GT[image_pair12].scale12, One2MultiImagePairs_Colmap[image_pair12].scale12, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])
            scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale21, One2MultiImagePairs_GT[image_pair12].scale21, One2MultiImagePairs_Colmap[image_pair12].scale21, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])))
        else:
            scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale12, One2MultiImagePairs_GT[image_pair12].scale12, One2MultiImagePairs_Colmap[image_pair12].scale12, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])))
            scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale21, One2MultiImagePairs_GT[image_pair12].scale21, One2MultiImagePairs_Colmap[image_pair12].scale21, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])))
        print("scaleRecorderMat.shape = ", scaleRecorderMat.shape)

        scale_choice = 1
        if DepthSource=='DeMoN':
            depth_choice = One2MultiImagePairs_DeMoN[image_pair12].depth1
            if PoseSource=='Colmap':
                # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale12
                # scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12
                scale_choice = One2MultiImagePairs_correctionColmap[image_pair12].scale12
            if PoseSource=='GT':
                # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale12
                # scale_choice = One2MultiImagePairs_GT[image_pair12].scale12
                scale_choice = One2MultiImagePairs_correctionGT[image_pair12].scale12
        if DepthSource=='Colmap':
            depth_choice = One2MultiImagePairs_Colmap[image_pair12].depth1
            if PoseSource=='GT':
                # scale_choice = One2MultiImagePairs_GT[image_pair12].scale12/One2MultiImagePairs_Colmap[image_pair12].scale12    # or a fixed value
                scale_choice = One2MultiImagePairs_correctionGT[image_pair12].scale12/One2MultiImagePairs_correctionColmap[image_pair12].scale12    # or a fixed value
        if DepthSource=='GT':
            depth_choice = One2MultiImagePairs_GT[image_pair12].depth1
            if PoseSource=='Colmap':
                # scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12/One2MultiImagePairs_GT[image_pair12].scale12    # or a fixed value
                scale_choice = One2MultiImagePairs_correctionColmap[image_pair12].scale12/One2MultiImagePairs_correctionGT[image_pair12].scale12    # or a fixed value

        tmpScaledDepthMap =  depth_choice * scale_choice
        tmpScaledInvDepthMap = 1/tmpScaledDepthMap
        if viewCnt==0:
            DepthMapRecorder = tmpScaledDepthMap
            InvDepthMapRecorder = tmpScaledInvDepthMap
        else:
            DepthMapRecorder = np.dstack((DepthMapRecorder,tmpScaledDepthMap))
            InvDepthMapRecorder = np.dstack((InvDepthMapRecorder,tmpScaledInvDepthMap))
        viewCnt += 1

        print("viewCnt = ", viewCnt, ": DepthMapRecorder.shape = ", DepthMapRecorder.shape)

    DepthMapRecorder = np.reshape(DepthMapRecorder, [DepthMapRecorder.shape[0]*DepthMapRecorder.shape[1], DepthMapRecorder.shape[2]])
    InvDepthMapRecorder = np.reshape(InvDepthMapRecorder, [InvDepthMapRecorder.shape[0]*InvDepthMapRecorder.shape[1], InvDepthMapRecorder.shape[2]])
    pixelwiseStats[:,0] = np.mean(DepthMapRecorder, axis=1)
    pixelwiseStats[:,1] = np.std(DepthMapRecorder, axis=1)
    pixelwiseStats[:,2] = np.mean(InvDepthMapRecorder, axis=1)
    pixelwiseStats[:,3] = np.std(InvDepthMapRecorder, axis=1)
    print("pixelwiseStats = ", pixelwiseStats)
    print("scaleRecorderMat.shape = ", scaleRecorderMat.shape)
    np.savetxt(os.path.join(outdir, 'scale_matrix.txt'),scaleRecorderMat)
    if True:
        plt.figure()
        plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,1])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,0]),0,1.2*np.max(scaleRecorderMat[:,1])))
        #plt.axis((0,1,0,1.2*np.max(scaleRecorderMat[:,1])))
        plt.xlabel('DeMoN Scale')
        plt.ylabel('GT Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,2])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,0]),0,1.2*np.max(scaleRecorderMat[:,2])))
        #plt.axis((0,1,0,1.2*np.max(scaleRecorderMat[:,2])))
        plt.xlabel('DeMoN Scale')
        plt.ylabel('Colmap Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.scatter(scaleRecorderMat[:,2],scaleRecorderMat[:,1])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,2]),0,1.2*np.max(scaleRecorderMat[:,1])))
        plt.xlabel('Colmap Scale')
        plt.ylabel('GT Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
    # ClosePairs_Depth_DeMoNDepth_GTScale_GTPose_pixelwise_std_CDF.png
    # ClosePairs_Depth_DeMoNDepth_ColmapScale_ColmapPose_pixelwise_std_CDF.png

    data = pixelwiseStats[:,1].flatten()
    print(data.shape)
    # data = data[data!=np.nan]
    # data = data[data!=np.inf]
    data = data[np.isfinite(data)]
    print(data.shape)
    plt.figure()
    num_bins = 2000
    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.title('CDF of pixelwise std of depth values: mu={0}, sigma={1}, from {2} pairs'.format(np.mean(data), np.std(data), int(len(list(One2MultiImagePairs_DeMoN.keys())))))
    plt.xlabel('pixelwise standard deviation of depth values')
    plt.ylabel('Probability')
    plt.show()

    # for q in [50, 90, 95, 100]:
    #     print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
    #
    # plt.figure()
    # hist = plt.hist(data, normed=False, bins=1000)
    # # plt.plot(bin_edges[1:], counts)
    # # plt.bar(bin_edges[1:], height=counts)
    # plt.xlabel('pixelwise standard deviation of depth values')
    # plt.ylabel('counts number')
    # plt.title('Histogram of pixelwise standard deviation of depth values: mu={0}, sigma={1}'.format(np.mean(data), np.std(data)))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.show()
    #######################################################
    data = pixelwiseStats[:,3].flatten()
    print(data.shape)
    # data = data[data!=np.nan]
    # data = data[data!=np.inf]
    data = data[np.isfinite(data)]
    print(data.shape)
    plt.figure()
    num_bins = 2000
    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.title('CDF of pixelwise std of inv_depth values: mu={0}, sigma={1}, from {2} pairs'.format(np.mean(data), np.std(data), int(len(list(One2MultiImagePairs_DeMoN.keys())))))
    plt.xlabel('pixelwise standard deviation of inv_depth values')
    plt.ylabel('Probability')
    plt.show()

    # for q in [50, 90, 95, 100]:
    #     print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
    #
    # plt.figure()
    # hist = plt.hist(data, normed=False, bins=1000)
    # # plt.plot(bin_edges[1:], counts)
    # # plt.bar(bin_edges[1:], height=counts)
    # plt.xlabel('pixelwise standard deviation of inv_depth values')
    # plt.ylabel('counts number')
    # plt.title('Histogram of pixelwise standard deviation of inv_depth values: mu={0}, sigma={1}'.format(np.mean(data), np.std(data)))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.show()


def flow_from_depth(depth, R, T, K):
    # K = np.eye(3)
    # K[0, 0] = K[1, 1] = F
    # K[0, 2] = depth.shape[1] / 2
    # K[1, 2] = depth.shape[0] / 2
    y, x = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = 1 / depth[:]
    coords1 = np.column_stack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))
    coords2 = np.dot(coords1, np.linalg.inv(K).T)
    coords2[:,0] *= z.ravel()
    coords2[:,1] *= z.ravel()
    coords2[:,2] = z.ravel()
    coords2 = np.dot(coords2, R[:].T) + T[:]
    coords2 = np.dot(coords2, K.T)
    coords2 /= coords2[:, 2][:, None]
    flow12 = coords2 - coords1
    flowx = flow12[:, 0].reshape(depth.shape) / depth.shape[1]
    flowy = flow12[:, 1].reshape(depth.shape) / depth.shape[0]
    return np.concatenate((flowx[None], flowy[None]), axis=0)


def flow_to_matches(flow):
    fx = np.round(flow[0] * flow.shape[2]).astype(np.int)
    fy = np.round(flow[1] * flow.shape[1]).astype(np.int)
    y1, x1 = np.mgrid[0:flow.shape[1], 0:flow.shape[2]]
    x2 = x1.ravel() + fx.ravel()
    y2 = y1.ravel() + fy.ravel()
    mask = (x2 >= 0) & (x2 < flow.shape[2]) & (y2 >= 0) & (y2 < flow.shape[1])
    matches = np.zeros((mask.size, 2), dtype=np.uint32)
    matches[:, 0] = np.arange(mask.size)
    matches[:, 1] = y2 * flow.shape[2] + x2
    # matches = matches[mask].copy()
    # coords1 = np.column_stack((x1.ravel(), y1.ravel()))[mask]
    # coords2 = np.column_stack((x2, y2))[mask]
    coords1 = np.column_stack((x1.ravel(), y1.ravel()))
    coords2 = np.column_stack((x2, y2))
    return matches, coords1, coords2, mask

def checkDepthConsistencyPixelwise_MultiViews(image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, PoseSource='GT', DepthSource='DeMoN', w=256, h=192):
    global target_K
    ###### Check Depth & InvDepth Stats
    pixelwiseStats = np.zeros([h*w,4])
    # scaleRecorderMat = np.empty([1,3])
    viewCnt=0
    for image_pair12 in One2MultiImagePairs_DeMoN.keys():
        ### check correlation between different scales regarding one-to-multi image pairs
        if viewCnt==0:
            scaleRecorderMat = np.array([One2MultiImagePairs_DeMoN[image_pair12].scale12, One2MultiImagePairs_GT[image_pair12].scale12, One2MultiImagePairs_Colmap[image_pair12].scale12, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])
            scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale21, One2MultiImagePairs_GT[image_pair12].scale21, One2MultiImagePairs_Colmap[image_pair12].scale21, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])))
        else:
            scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale12, One2MultiImagePairs_GT[image_pair12].scale12, One2MultiImagePairs_Colmap[image_pair12].scale12, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])))
            scaleRecorderMat = np.vstack((scaleRecorderMat,np.array([One2MultiImagePairs_DeMoN[image_pair12].scale21, One2MultiImagePairs_GT[image_pair12].scale21, One2MultiImagePairs_Colmap[image_pair12].scale21, One2MultiImagePairs_correctionGT[image_pair12].scale12, One2MultiImagePairs_correctionColmap[image_pair12].scale12])))
        print("scaleRecorderMat.shape = ", scaleRecorderMat.shape)

        ###### Add view1 in image_pair12
        scale_choice = 1
        if DepthSource=='DeMoN':
            depth_choice = One2MultiImagePairs_DeMoN[image_pair12].depth1
            if PoseSource=='Colmap':
                # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale12
                # scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12
                scale_choice = One2MultiImagePairs_correctionColmap[image_pair12].scale12
            if PoseSource=='GT':
                # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale12
                # scale_choice = One2MultiImagePairs_GT[image_pair12].scale12
                scale_choice = One2MultiImagePairs_correctionGT[image_pair12].scale12
        if DepthSource=='Colmap':
            depth_choice = One2MultiImagePairs_Colmap[image_pair12].depth1
            if PoseSource=='GT':
                # scale_choice = One2MultiImagePairs_GT[image_pair12].scale12/One2MultiImagePairs_Colmap[image_pair12].scale12    # or a fixed value
                scale_choice = One2MultiImagePairs_correctionGT[image_pair12].scale12/One2MultiImagePairs_correctionColmap[image_pair12].scale12    # or a fixed value
        if DepthSource=='GT':
            depth_choice = One2MultiImagePairs_GT[image_pair12].depth1
            if PoseSource=='Colmap':
                # scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12/One2MultiImagePairs_GT[image_pair12].scale12    # or a fixed value
                scale_choice = One2MultiImagePairs_correctionColmap[image_pair12].scale12/One2MultiImagePairs_correctionGT[image_pair12].scale12    # or a fixed value

        tmpScaledDepthMap =  depth_choice * scale_choice
        tmpScaledInvDepthMap = 1/tmpScaledDepthMap
        if viewCnt==0:
            DepthMapRecorder = tmpScaledDepthMap
            InvDepthMapRecorder = tmpScaledInvDepthMap
        else:
            DepthMapRecorder = np.dstack((DepthMapRecorder,tmpScaledDepthMap))
            InvDepthMapRecorder = np.dstack((InvDepthMapRecorder,tmpScaledInvDepthMap))
        viewCnt += 1


        ###### Add view2 in image_pair12, i.e. view1 in image_pair21
        scale_choice = 1
        if DepthSource=='DeMoN':
            depth_choice = One2MultiImagePairs_DeMoN[image_pair12].depth2
            if PoseSource=='Colmap':
                # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale21
                # scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale21
                scale_choice = One2MultiImagePairs_correctionColmap[image_pair12].scale21
            if PoseSource=='GT':
                # scale_choice = One2MultiImagePairs_DeMoN[image_pair12].scale21
                # scale_choice = One2MultiImagePairs_GT[image_pair12].scale21
                scale_choice = One2MultiImagePairs_correctionGT[image_pair12].scale21
        if DepthSource=='Colmap':
            depth_choice = One2MultiImagePairs_Colmap[image_pair12].depth2
            if PoseSource=='GT':
                # scale_choice = One2MultiImagePairs_GT[image_pair12].scale12/One2MultiImagePairs_Colmap[image_pair12].scale21    # or a fixed value
                scale_choice = One2MultiImagePairs_correctionGT[image_pair12].scale12/One2MultiImagePairs_correctionColmap[image_pair12].scale21    # or a fixed value
        if DepthSource=='GT':
            depth_choice = One2MultiImagePairs_GT[image_pair12].depth2
            if PoseSource=='Colmap':
                # scale_choice = One2MultiImagePairs_Colmap[image_pair12].scale12/One2MultiImagePairs_GT[image_pair12].scale21    # or a fixed value
                scale_choice = One2MultiImagePairs_correctionColmap[image_pair12].scale12/One2MultiImagePairs_correctionGT[image_pair12].scale21    # or a fixed value

        tmpScaledDepthMap2 =  depth_choice * scale_choice
        tmpScaledInvDepthMa2 = 1/tmpScaledDepthMap2

        ## retrieve corresponding depth values in other views
        if PoseSource=='Colmap':
            flow12 = flow_from_depth(tmpScaledDepthMap, One2MultiImagePairs_Colmap[image_pair12].Relative12_4by4[0:3,0:3], One2MultiImagePairs_Colmap[image_pair12].Relative12_4by4[0:3,3], target_K)
        if PoseSource=='GT':
            flow12 = flow_from_depth(tmpScaledDepthMap, One2MultiImagePairs_GT[image_pair12].Relative12_4by4[0:3,0:3], One2MultiImagePairs_GT[image_pair12].Relative12_4by4[0:3,3], target_K)
        matches12, coords121, coords122, mask12 = flow_to_matches(flow12)
        print("flow12.shape = ", flow12.shape)
        print("coords122.shape = ", coords122.shape)
        print("coords121 x max = ", np.max(coords121[:,0]), "; coords121 y max = ", np.max(coords121[:,1]))

        matchedDepthMapView2 = np.empty((tmpScaledDepthMap2.shape[0],tmpScaledDepthMap2.shape[1]))
        matchedDepthMapView2[:] = np.nan
        matchedInvDepthMapView2 = np.empty((tmpScaledDepthMap2.shape[0],tmpScaledDepthMap2.shape[1]))
        matchedInvDepthMapView2[:] = np.nan
        for pixIdx in range(len(mask12)):
            if mask12[pixIdx] == True:
                # print("(", coords121[pixIdx,0], coords121[pixIdx,1], ") (", coords122[pixIdx,0], coords122[pixIdx,1], ")")
                matchedDepthMapView2[coords121[pixIdx,1], coords121[pixIdx,0]] = tmpScaledDepthMap2[coords122[pixIdx,1], coords122[pixIdx,0]]
                matchedInvDepthMapView2[coords121[pixIdx,1], coords121[pixIdx,0]] = tmpScaledInvDepthMa2[coords122[pixIdx,1], coords122[pixIdx,0]]

        DepthMapRecorder = np.dstack((DepthMapRecorder,matchedDepthMapView2))
        InvDepthMapRecorder = np.dstack((InvDepthMapRecorder,matchedInvDepthMapView2))

        viewCnt += 1

        print("viewCnt = ", viewCnt, ": DepthMapRecorder.shape = ", DepthMapRecorder.shape)

    DepthMapRecorder = np.reshape(DepthMapRecorder, [DepthMapRecorder.shape[0]*DepthMapRecorder.shape[1], DepthMapRecorder.shape[2]])
    InvDepthMapRecorder = np.reshape(InvDepthMapRecorder, [InvDepthMapRecorder.shape[0]*InvDepthMapRecorder.shape[1], InvDepthMapRecorder.shape[2]])
    pixelwiseStats[:,0] = np.nanmean(DepthMapRecorder, axis=1)
    pixelwiseStats[:,1] = np.nanstd(DepthMapRecorder, axis=1)
    pixelwiseStats[:,2] = np.nanmean(InvDepthMapRecorder, axis=1)
    pixelwiseStats[:,3] = np.nanstd(InvDepthMapRecorder, axis=1)
    print("pixelwiseStats = ", pixelwiseStats)
    print("scaleRecorderMat.shape = ", scaleRecorderMat.shape)
    np.savetxt(os.path.join(outdir, 'scale_matrix.txt'),scaleRecorderMat)
    if True:
        plt.figure()
        plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,1])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,0]),0,1.2*np.max(scaleRecorderMat[:,1])))
        #plt.axis((0,1,0,1.2*np.max(scaleRecorderMat[:,1])))
        plt.xlabel('DeMoN Scale')
        plt.ylabel('GT Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.scatter(scaleRecorderMat[:,0],scaleRecorderMat[:,2])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,0]),0,1.2*np.max(scaleRecorderMat[:,2])))
        #plt.axis((0,1,0,1.2*np.max(scaleRecorderMat[:,2])))
        plt.xlabel('DeMoN Scale')
        plt.ylabel('Colmap Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.scatter(scaleRecorderMat[:,2],scaleRecorderMat[:,1])
        plt.axis((0,1.2*np.max(scaleRecorderMat[:,2]),0,1.2*np.max(scaleRecorderMat[:,1])))
        plt.xlabel('Colmap Scale')
        plt.ylabel('GT Scale')
        #plt.axis('equal')
        plt.grid(True)
        plt.show()
    # ClosePairs_Depth_DeMoNDepth_GTScale_GTPose_pixelwise_std_CDF.png
    # ClosePairs_Depth_DeMoNDepth_ColmapScale_ColmapPose_pixelwise_std_CDF.png

    data = pixelwiseStats[:,1].flatten()
    print(data.shape)
    # data = data[data!=np.nan]
    # data = data[data!=np.inf]
    data = data[np.isfinite(data)]
    print(data.shape)
    plt.figure()
    num_bins = 2000
    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.title('CDF of pixelwise std of depth values: mu={0}, sigma={1}, from {2} pairs'.format(np.mean(data), np.std(data), int(len(list(One2MultiImagePairs_DeMoN.keys())))))
    plt.xlabel('pixelwise standard deviation of depth values')
    plt.ylabel('Probability')
    plt.show()

    # for q in [50, 90, 95, 100]:
    #     print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
    #
    # plt.figure()
    # hist = plt.hist(data, normed=False, bins=1000)
    # # plt.plot(bin_edges[1:], counts)
    # # plt.bar(bin_edges[1:], height=counts)
    # plt.xlabel('pixelwise standard deviation of depth values')
    # plt.ylabel('counts number')
    # plt.title('Histogram of pixelwise standard deviation of depth values: mu={0}, sigma={1}'.format(np.mean(data), np.std(data)))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.show()
    #######################################################
    data = pixelwiseStats[:,3].flatten()
    print(data.shape)
    # data = data[data!=np.nan]
    # data = data[data!=np.inf]
    data = data[np.isfinite(data)]
    print(data.shape)
    plt.figure()
    num_bins = 2000
    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.title('CDF of pixelwise std of inv_depth values: mu={0}, sigma={1}, from {2} pairs'.format(np.mean(data), np.std(data), int(len(list(One2MultiImagePairs_DeMoN.keys())))))
    plt.xlabel('pixelwise standard deviation of inv_depth values')
    plt.ylabel('Probability')
    plt.show()

    # for q in [50, 90, 95, 100]:
    #     print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
    #
    # plt.figure()
    # hist = plt.hist(data, normed=False, bins=1000)
    # # plt.plot(bin_edges[1:], counts)
    # # plt.bar(bin_edges[1:], height=counts)
    # plt.xlabel('pixelwise standard deviation of inv_depth values')
    # plt.ylabel('counts number')
    # plt.title('Histogram of pixelwise standard deviation of inv_depth values: mu={0}, sigma={1}'.format(np.mean(data), np.std(data)))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.show()


image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap = findMultiPairs(infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, tarImageFileName)

pointclouds_beforefiltering = {}


def PCA(data, correlation = False, sort = True):
    """ Applies Principal Component Analysis to the data
    https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
    Parameters
    ----------
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix.

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors


def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])

    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]

    #: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal

def write_pointcloud_to_PLY():
    pointcloud_polydata = create_pointcloud_polydata(
        points=tmpPC['points'],
        # colors=tmpPC['colors'] if 'colors' in tmpPC else None,
        colors=tmpPC['colors'],
        )

    appendFilterModel = vtk.vtkAppendPolyData()
    cam1_polydata = create_camera_polydata(np.eye(3), [0,0,0], True)
    cam2_polydata = create_camera_polydata(angleaxis_to_rotation_matrix(rotation[0]), translation[0], True)
    appendFilterModel.AddInputData(pointcloud_polydata)
    appendFilterModel.AddInputData(cam1_polydata)
    appendFilterModel.AddInputData(cam2_polydata)
    appendFilterModel.Update()

    plywriter = vtk.vtkPLYWriter()
    plywriter.SetFileName('DeMoN_Sculpture_Example_pointcloud.ply')
    # plywriter.SetInputData(pointcloud_polydata)
    plywriter.SetInputData(appendFilterModel.GetOutput())
    # plywriter.SetFileTypeToASCII()
    plywriter.SetArrayName('colors')
    plywriter.Write()

def main():
    global pointclouds_beforefiltering, curIteration, initColmapGTRatio, appendFilterPC, appendFilterModel, alpha, tmpFittingCoef_Colmap_GT, scaleRecordMat, image_pairs, TheiaOrColmapOrGTPoses, DeMoNOrColmapOrGTDepths, sliderMin, sliderMax, interactor, renderer, image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT

    # ###### check the statistics of generated point clouds
    # checkDepthConsistencyPixelwise_MultiViews(image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, w=256, h=192)

    # ###### visualize the view one by one
    # visMultiViewsPointCloudInGlobalFrame(renderer, alpha, image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, initBool=True, setColmapGTRatio=True)
    ###### visualize all retrieved the views
    curIteration = len(One2MultiImagePairs_DeMoN)
    print("curIteration = ", curIteration)
    visMultiViewsPointCloudInGlobalFrame(renderer, alpha, image_pairs_One2Multi, One2MultiImagePairs_DeMoN, One2MultiImagePairs_GT, One2MultiImagePairs_Colmap, One2MultiImagePairs_correctionGT, One2MultiImagePairs_correctionColmap, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, PoseSource=TheiaOrColmapOrGTPoses, DepthSource=DeMoNOrColmapOrGTDepths, initBool=False, setColmapGTRatio=True)

    print("len(pointclouds_beforefiltering) = ", len(pointclouds_beforefiltering))

    ###### loop over point clouds to compute normals and weights
    patchSize = 7*7
    for image_pair12 in pointclouds_beforefiltering.keys():
        print("pointclouds_beforefiltering[image_pair12]['points'].shape = ", pointclouds_beforefiltering[image_pair12]['points'].shape)
        # pts = pointclouds_beforefiltering[image_pair12]['points']
        curPC = pointclouds_beforefiltering[image_pair12]['points']
        curNormals = pointclouds_beforefiltering[image_pair12]['normals']
        cam_v = np.linalg.inv(One2MultiImagePairs_GT[image_pair12].Extrinsic1_4by4)[0:3,3]
        print("cam_v = ", cam_v)
        # distance.cdist(pointclouds_beforefiltering[image_pair12]['points'], pointclouds_beforefiltering[image_pair12]['points']).argmin()
        # curPC[spatial.KDTree(curPC).query(pts[0])[10]]
        if True:
            fitted_normals = []
            weights_from_fitted_normals = []
            for ptIdx in range(curPC.shape[0]):
                queryPt = curPC[ptIdx]
                dist,indices = spatial.KDTree(curPC).query(queryPt, k=3)
                # print(dist, " ", indices)
                p1 = curPC[indices[1],:]
                p2 = curPC[indices[2],:]
                fitted_normal = np.cross((p1-queryPt), (p2-queryPt))
                fitted_normal = fitted_normal/np.linalg.norm(fitted_normal)
                # pt, fitted_normal = best_fitting_plane(patchPts)
                # print(pt, " ", fitted_normal, "; query point = ", queryPt)
                fitted_normals.append(fitted_normal)

                ###### Calculate weights
                weight_ptIdx = np.dot(fitted_normal, (queryPt-cam_v)/np.linalg.norm(queryPt-cam_v))
                weights_from_fitted_normals.append(weight_ptIdx)
            print("np.array(fitted_normals).shape = ", np.array(fitted_normals).shape)
            print("np.array(weights_from_fitted_normals).shape = ", np.array(weights_from_fitted_normals).shape)
            pointclouds_beforefiltering[image_pair12]['fitted_normals'] = np.array(fitted_normals)
            pointclouds_beforefiltering[image_pair12]['weights_from_fitted_normals'] = np.array(weights_from_fitted_normals)
            pointclouds_beforefiltering[image_pair12]['filtering_flag'] = []    # used for keeping a boolean value indicating if the pt should be kept (1) or filtered (0)!

        if False:
            fitted_normals = []
            weights_from_fitted_normals = []
            for ptIdx in range(curPC.shape[0]):
                queryPt = curPC[ptIdx]
                dist,indices = spatial.KDTree(curPC).query(queryPt, k=patchSize)
                # print(dist, " ", indices)
                patchPts = curPC[indices,:]

                pt, fitted_normal = best_fitting_plane(patchPts)
                # print(pt, " ", fitted_normal, "; query point = ", queryPt)
                fitted_normals.append(fitted_normal)

                ###### Calculate weights
                weight_ptIdx = np.dot(fitted_normal, (queryPt-cam_v)/np.linalg.norm(queryPt-cam_v))
                weights_from_fitted_normals.append(weight_ptIdx)
            print("np.array(fitted_normals).shape = ", np.array(fitted_normals).shape)
            print("np.array(weights_from_fitted_normals).shape = ", np.array(weights_from_fitted_normals).shape)
            pointclouds_beforefiltering[image_pair12]['fitted_normals'] = np.array(fitted_normals)
            pointclouds_beforefiltering[image_pair12]['weights_from_fitted_normals'] = np.array(weights_from_fitted_normals)
            pointclouds_beforefiltering[image_pair12]['filtering_flag'] = []    # used for keeping a boolean value indicating if the pt should be kept (1) or filtered (0)!

        if False:
            fitted_normals = []
            weights_from_fitted_normals = []
            for ptIdx in range(curPC.shape[0]):
                queryPt = curPC[ptIdx]

                ###### Calculate weights
                # print(pointclouds_beforefiltering[image_pair12].keys())
                fitted_normal = pointclouds_beforefiltering[image_pair12]['normals'][ptIdx,:]
                # print(pointclouds_beforefiltering[image_pair12]['normals'][ptIdx,:])
                fitted_normals.append(fitted_normal)
                weight_ptIdx = np.dot(fitted_normal, (queryPt-cam_v)/np.linalg.norm(queryPt-cam_v))
                weights_from_fitted_normals.append(weight_ptIdx)
            # print("np.array(fitted_normals).shape = ", np.array(fitted_normals).shape)
            print("np.array(weights_from_fitted_normals).shape = ", np.array(weights_from_fitted_normals).shape)
            pointclouds_beforefiltering[image_pair12]['fitted_normals'] = np.array(fitted_normals)
            pointclouds_beforefiltering[image_pair12]['weights_from_fitted_normals'] = np.array(weights_from_fitted_normals)
            pointclouds_beforefiltering[image_pair12]['filtering_flag'] = []    # used for keeping a boolean value indicating if the pt should be kept (1) or filtered (0)!
        # print(pointclouds_beforefiltering[image_pair12].keys())
        # return

    # sigma = 0.1 * (np.max(pointclouds_beforefiltering[list(pointclouds_beforefiltering.keys())[0]]['scaled_depth']) - np.min(pointclouds_beforefiltering[list(pointclouds_beforefiltering.keys())[0]]['scaled_depth']))   # σ should be chosen according to the scale of the scene, so we set it to 1% of the depth range (e.g., the length of the bounding box along the z-axis);
    sigma = 1.25 * 0.1 * (np.max(pointclouds_beforefiltering[list(pointclouds_beforefiltering.keys())[0]]['scaled_depth']) - np.min(pointclouds_beforefiltering[list(pointclouds_beforefiltering.keys())[0]]['scaled_depth']))   # σ should be chosen according to the scale of the scene, so we set it to 1% of the depth range (e.g., the length of the bounding box along the z-axis);
    td = 0.1 * sigma
    tp = 0.2
    tv = 0.075 * len(list(pointclouds_beforefiltering.keys()))
    print("sigma = ", sigma, ", td = ", td, ", tp = ", tp, ", tv = ", tv)
    filtered_3D_points_positions = []
    filtered_3D_points_colors = []
    ###### loop over points and different views to do the filtering
    for image_pair12_i in pointclouds_beforefiltering.keys():
        print("len(pointclouds_beforefiltering[image_pair12_i]['points']) = ", len(pointclouds_beforefiltering[image_pair12_i]['points']))
        depth_after_filtering_image_pair12_i = np.zeros([h,w])
        RGBimg_after_filtering_image_pair12_i = One2MultiImagePairs_GT[image_pair12_i].image1
        for ptIdx in range(len(pointclouds_beforefiltering[image_pair12_i]['points'])):
            d_pt = 0
            w_pt = 0
            v_pt = 0
            s = 0
            s2 = 0
            cur_xy_coordinate = pointclouds_beforefiltering[image_pair12_i]['coordinates'][ptIdx,:]
            cur_3D_point_position = pointclouds_beforefiltering[image_pair12_i]['points'][ptIdx,:]
            cur_3D_point_color = pointclouds_beforefiltering[image_pair12_i]['colors'][ptIdx,:]
            cur_3D_point_depth = pointclouds_beforefiltering[image_pair12_i]['scaled_depth'][cur_xy_coordinate[1],cur_xy_coordinate[0]]
            for image_pair12_j in pointclouds_beforefiltering.keys():
                # if image_pair12_i != image_pair12_j:
                if True:
                    cam_vi = np.linalg.inv(One2MultiImagePairs_GT[image_pair12_i].Extrinsic1_4by4)[0:3,3]
                    cam_vj = np.linalg.inv(One2MultiImagePairs_GT[image_pair12_j].Extrinsic1_4by4)[0:3,3]
                    if np.dot(cam_vi, cam_vj) > 0:
                        # print("skipped camera centers = ", cam_vi, "; ", cam_vj)
                        continue
                    ###### weight and depth interpolations are skipped here because mesh triangle is not involved in our setup
                    ###### retrieve corresponding depth values in other views and compute the depth diff
                    # if PoseSource=='Colmap':
                    #     flow12 = flow_from_depth(tmpScaledDepthMap, One2MultiImagePairs_Colmap[image_pair12].Relative12_4by4[0:3,0:3], One2MultiImagePairs_Colmap[image_pair12].Relative12_4by4[0:3,3], target_K)
                    if PoseSource=='GT':
                        RelativeTransformation_4by4 = One2MultiImagePairs_GT[image_pair12_j].Extrinsic1_4by4 * np.linalg.inv(One2MultiImagePairs_GT[image_pair12_i].Extrinsic1_4by4)
                        flow12 = flow_from_depth(pointclouds_beforefiltering[image_pair12_i]['scaled_depth'], RelativeTransformation_4by4[0:3,0:3], RelativeTransformation_4by4[0:3,3], target_K)
                    matches12, coords121, coords122, mask12 = flow_to_matches(flow12)
                    # print("flow12.shape = ", flow12.shape)
                    # print("coords122.shape = ", coords122.shape)
                    # # print("coords121 x max = ", np.max(coords121[:,0]), "; coords121 y max = ", np.max(coords121[:,1]))
                    idx1D = cur_xy_coordinate[0] + cur_xy_coordinate[1]*w
                    print("coords121[idx1D] = ", coords121[idx1D], ";? (", cur_xy_coordinate[0],", ", cur_xy_coordinate[1], ")")
                    interpolatedWeight = pointclouds_beforefiltering[image_pair12_j]['weights_from_fitted_normals'][idx1D]
                    if mask12[idx1D] == True:
                        d_diff = pointclouds_beforefiltering[image_pair12_j]['scaled_depth'][coords122[idx1D,1], coords122[idx1D,0]] - pointclouds_beforefiltering[image_pair12_i]['scaled_depth'][cur_xy_coordinate[1],cur_xy_coordinate[0]]
                    else:
                        continue
                    if d_diff < -sigma:
                        continue
                    if d_diff > sigma:
                        d_diff = sigma

                    d_pt = ( w_pt*d_pt + interpolatedWeight*d_diff/sigma ) / (w_pt + interpolatedWeight)
                    w_pt = w_pt + interpolatedWeight

                    if d_diff != sigma: #update photoconsistency only for range surfaces close to p
                        s = s + pointclouds_beforefiltering[image_pair12_j]['colors'][idx1D,:]
                        s2 = s2 + np.dot(pointclouds_beforefiltering[image_pair12_j]['colors'][idx1D,:], pointclouds_beforefiltering[image_pair12_j]['colors'][idx1D,:])
                        v_pt = v_pt + 1
            if v_pt > 0:
                p_pt = math.sqrt( (s2 - np.dot(s,s)/v_pt) / v_pt ) * (2/(255*math.sqrt(3)))
                if  d_pt > -td and d_pt < 0 and p_pt < tp and v_pt > tv:
                    filtered_3D_points_positions.append(cur_3D_point_position)
                    filtered_3D_points_colors.append(cur_3D_point_color)
            # else:
            #     print("skip the 3d point because there are no corresponding points in other views!")
    pointclouds_afterfiltering = {}
    pointclouds_afterfiltering['points'] = np.array(filtered_3D_points_positions)
    print("pointclouds_afterfiltering['points'].shape = ", pointclouds_afterfiltering['points'].shape)
    pointclouds_afterfiltering['colors'] = np.array(filtered_3D_points_colors)

    ## save the filtered point cloud
    ###############################################################
    appendFilterModel = vtk.vtkAppendPolyData()
    # cam1_polydata = create_camera_polydata(R1, t1, True)
    # cam2_polydata = create_camera_polydata(R2, t2, True)
    pointcloud1_polydata = create_pointcloud_polydata(
    points=pointclouds_afterfiltering['points'],
    colors=pointclouds_afterfiltering['colors'] if 'colors' in pointclouds_afterfiltering else None,
    )
    appendFilterModel.AddInputData(pointcloud1_polydata)
    # appendFilterModel.AddInputData(pointcloud2_polydata)
    # appendFilterModel.AddInputData(cam1_polydata)
    # appendFilterModel.AddInputData(cam2_polydata)
    appendFilterModel.Update()

    plywriter = vtk.vtkPLYWriter()
    plywriter.SetFileName(('test_filtered_pointcloud.ply'))
    # plywriter.SetFileName(('SUN3D_GT_pair_pointcloud_'+inputSUN3D_trainingdata.split('/')[-1][:-3]+'_'+SUN3D_datasetname+'_'+str(predict_scale)+'_'+str(predict_scale21)+'.ply'))
    # plywriter.SetInputData(pointcloud_polydata)
    plywriter.SetInputData(appendFilterModel.GetOutput())
    plywriter.SetFileTypeToASCII()
    plywriter.SetArrayName('Colors')
    plywriter.Write()
    ###############################################################


    ###### VTK Visualization
    # renwin = vtk.vtkRenderWindow()
    # renwin.SetWindowName("Point Cloud Viewer")
    # renwin.SetSize(800,600)
    # renwin.AddRenderer(renderer)
    #
    # # An interactor
    # interactor.SetInteractorStyle(MyKeyPressInteractorStyle())
    # interactor.SetRenderWindow(renwin)
    #
    # # Start
    # interactor.Initialize()
    # interactor.Start()

if __name__ == "__main__":
    main()
