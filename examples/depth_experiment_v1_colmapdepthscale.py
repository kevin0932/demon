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
    #global SliderRepres, SliderWidget, interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    global interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    sliderRepres = obj.GetRepresentation()
    pos = sliderRepres.GetValue()
    # contourFilter.SetValue(0, pos)
    alpha=pos

    #close_window(interactor)
    #del renWin, iren

    # renderer = visPointCloudInGlobalFrame(alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT)
    visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)
    #renderer = visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)
    renderer.Modified()
    print("vtkSliderCallback2~~~~~~~~~~~~~~~")
    # #### vtk slidingbar to adjust some parameters Runtime
    SliderRepres = vtk.vtkSliderRepresentation2D()
    min = 0 #ImageViewer.GetSliceMin()
    max = 2 #ImageViewer.GetSliceMax()
    SliderRepres.SetMinimumValue(min)
    SliderRepres.SetMaximumValue(max)
    SliderRepres.SetValue(alpha)
    SliderRepres.SetTitleText("Slice")
    SliderRepres.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    SliderRepres.GetPoint1Coordinate().SetValue(0.2, 0.6)
    SliderRepres.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    SliderRepres.GetPoint2Coordinate().SetValue(0.4, 0.6)

    SliderRepres.SetSliderLength(0.02)
    SliderRepres.SetSliderWidth(0.03)
    SliderRepres.SetEndCapLength(0.01)
    SliderRepres.SetEndCapWidth(0.03)
    SliderRepres.SetTubeWidth(0.005)
    SliderRepres.SetLabelFormat("%3.0lf")
    SliderRepres.SetTitleHeight(0.02)
    SliderRepres.SetLabelHeight(0.02)

    SliderWidget = vtk.vtkSliderWidget()
    SliderWidget.SetInteractor(interactor)
    SliderWidget.SetRepresentation(SliderRepres)
    SliderWidget.KeyPressActivationOff()
    SliderWidget.SetAnimationModeToAnimate()
    SliderWidget.SetEnabled(True)
    SliderWidget.AddObserver("EndInteractionEvent", vtkSliderCallback2)

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
outdir = "/home/kevin/JohannesCode/ws1/demon_prediction"
# infile = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction/gerrard_hall_predictions.h5'
# infile = '/home/kevin/ThesisDATA/person-hall/demon_prediction/person_hall_predictions.h5'
# infile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction/CVG_barcelona_predictions.h5"
# infile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction/CVG_redmond_predictions.h5"
# infile = "/home/kevin/JohannesCode/ws1/demon_prediction/kevin_southbuilding_predictions_08012018.h5"
infile = "/home/kevin/JohannesCode/ws1/demon_prediction/freiburgSettingBak/View128_fuse_southbuilding_demon.h5"
ExhaustivePairInfile = "/home/kevin/JohannesCode/ws1/demon_prediction/freiburgSettingBak/kevin_southbuilding_predictions_06012018.h5"

# outimagedir_small = os.path.join(outdir,'images_demon_small')
# outimagedir_large = os.path.join(outdir,'images_demon')
# os.makedirs(outdir, exist_ok=True)
# os.makedirs(outimagedir_small, exist_ok=True)
# os.makedirs(outimagedir_large, exist_ok=True)
# os.makedirs(os.path.join(outdir,'graydepthmap'), exist_ok=True)
# os.makedirs(os.path.join(outdir,'vizdepthmap'), exist_ok=True)

# recondir = '/misc/lmbraid12/depthmotionnet/datasets/mvs_colmap/south-building/mvs/'
recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# recondir = '/home/kevin/ThesisDATA/ToyDataset_Desk/dense/'
# recondir = '/home/kevin/ThesisDATA/gerrard-hall/dense/'
# recondir = '/home/kevin/ThesisDATA/person-hall/dense/'
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/dense/"
# recondir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/dense/"

cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
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

    # gpu_options = tf.GPUOptions()
    # gpu_options.per_process_gpu_memory_fraction=0.8
    # session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    #
    # # init networks
    # bootstrap_net = BootstrapNet(session, data_format)
    # iterative_net = IterativeNet(session, data_format)
    # refine_net = RefinementNet(session, data_format)
    #
    # session.run(tf.global_variables_initializer())
    #
    # # load weights
    # saver = tf.train.Saver()
    # saver.restore(session,os.path.join(weights_dir,'demon_original'))


# if True:
def visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, initBool=False):
#def visPointCloudInGlobalFrame(NULLrenderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, initBool=False):
    data = h5py.File(infile)
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

# def visPointCloudInGlobalFrame(data, dataExhaustivePairs, data_format, renderer, appendFilterPC, appendFilterModel):
    image_pairs = set()
    it = 0

    for image_pair12 in data.keys():
        print("Processing", image_pair12)

        image_name1, image_name2 = image_pair12.split("---")
        image_pair21 = "{}---{}".format(image_name2, image_name1)

        tmp_dict = {}
        for image_id, image in images.items():
            if image.name == image_name1:
                tmp_dict[image_id] = image
            if image.name == image_name2:
                tmp_dict[image_id] = image

        # tmp_dict = {image_id: image}
        print("tmp_dict = ", tmp_dict)
        tmp_views = colmap.create_views(cameras, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
        tmp_views[0] = adjust_intrinsics(tmp_views[0], target_K, w, h,)
        tmp_views[1] = adjust_intrinsics(tmp_views[1], target_K, w, h,)

        view1 = tmp_views[0]
        view2 = tmp_views[1]

        if image_pair12 in image_pairs:
            continue

        # print("view1 = ", view1)
        image_pairs.add(image_pair12)
        # image_pairs.add(image_pair21)

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
        # # plt.imshow(data[image_pair12]["depth_upsampled"], cmap='Greys')
        # plt.imshow(view1.depth, cmap='Greys')
        ColmapExtrinsics1_4by4 = np.eye(4)
        ColmapExtrinsics1_4by4[0:3,0:3] = view1.R
        ColmapExtrinsics1_4by4[0:3,3] = view1.t# -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        ColmapExtrinsics2_4by4 = np.eye(4)
        ColmapExtrinsics2_4by4[0:3,0:3] = view2.R
        ColmapExtrinsics2_4by4[0:3,3] = view2.t # -np.dot(val.rotmat, val.tvec) # theia output camera position in world frame instead of extrinsic t

        ##### compute scales
        transScaleTheia = np.linalg.norm(np.linalg.inv(TheiaExtrinsics2_4by4)[0:3,3] - np.linalg.inv(TheiaExtrinsics1_4by4)[0:3,3])
        transScaleColmap = np.linalg.norm(np.linalg.inv(ColmapExtrinsics2_4by4)[0:3,3] - np.linalg.inv(ColmapExtrinsics1_4by4)[0:3,3])
        print("transScaleTheia = ", transScaleTheia, "; transScaleColmap = ", transScaleColmap, "; demon scale = ", data[image_pair12]['scale'].value)
        pred_scale = data[image_pair12]['scale'].value
        if it==0:
            scaleRecordMat = np.array([pred_scale, transScaleTheia, transScaleColmap])
        else:
            scaleRecordMat = np.vstack((scaleRecordMat, np.array([pred_scale, transScaleTheia, transScaleColmap])))
        print("scaleRecordMat.shape = ", scaleRecordMat.shape)


        GlobalExtrinsics1_4by4 = TheiaExtrinsics1_4by4
        GlobalExtrinsics2_4by4 = TheiaExtrinsics2_4by4
        # GlobalExtrinsics1_4by4 = ColmapExtrinsics1_4by4
        # GlobalExtrinsics2_4by4 = ColmapExtrinsics2_4by4

        ###### scale global poses by a constant (Colmap and Theia may generate 3D reconstruction in different scales, which may differ from the real object depth scale)
        # alpha = 0.28 # 0.3 0.5
        GlobalExtrinsics1_4by4[0:3,3] = alpha * GlobalExtrinsics1_4by4[0:3,3]
        GlobalExtrinsics2_4by4[0:3,3] = alpha * GlobalExtrinsics2_4by4[0:3,3]

        ###### get the first point clouds
        input_data = prepare_input_data(view1.image, view2.image, data_format)
        tmp_PointCloud1 = visualize_prediction(
                    inverse_depth=data[image_pair12]['depth_upsampled'].value,
                    # inverse_depth=1/view1.depth,
                    intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]), # sun3d intrinsics
                    image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
                    R1=GlobalExtrinsics1_4by4[0:3,0:3],
                    t1=GlobalExtrinsics1_4by4[0:3,3],
                    rotation=rotmat_To_angleaxis(np.dot(GlobalExtrinsics2_4by4[0:3,0:3], GlobalExtrinsics1_4by4[0:3,0:3].T)),
                    translation=GlobalExtrinsics2_4by4[0:3,3],   # should be changed, this is wrong!
                    ###### inverse_depth=data[image_pair12]['depth_upsampled'].value, by DeMoN depth---colmap poses
                    # scale=data[image_pair12]['scale'].value)    # apply scales estimated by DeMoN;
                    # scale=(data[image_pair12]['scale'].value+dataExhaustivePairs[image_pair21]['scale'].value)/2)    # apply scales estimated by DeMoN, average value by exchanging pair order;
                    # scale=transScaleColmap)   # apply scales calculated from colmap global/relative poses
                    # # scale=transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    # # scale=1/transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    ###### inverse_depth=data[image_pair12]['depth_upsampled'].value, by DeMoN depth---theia poses
                    # scale=1/data[image_pair12]['scale'].value)    # apply scales estimated by DeMoN;
                    scale=(data[image_pair12]['scale'].value+dataExhaustivePairs[image_pair21]['scale'].value)/2)    # apply scales estimated by DeMoN, average value by exchanging pair order;
                    # scale=transScaleColmap)   # apply scales calculated from colmap global/relative poses
                    # # scale=transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    # # scale=1/transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    ###### inverse_depth=1/view1.depth  by colmap depth---colmap poses
                    # scale=data[image_pair12]['scale'].value)    # apply scales estimated by DeMoN;
                    # scale=(data[image_pair12]['scale'].value+dataExhaustivePairs[image_pair21]['scale'].value)/2)    # apply scales estimated by DeMoN, average value by exchanging pair order;
                    # scale=transScaleColmap)   # apply scales calculated from colmap global/relative poses
                    # # scale=transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    # # scale=1/transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    ###### inverse_depth=1/view1.depth  by colmap depth---theia poses
                    # scale=1/data[image_pair12]['scale'].value)    # apply scales estimated by DeMoN;
                    # scale=2/(data[image_pair12]['scale'].value+dataExhaustivePairs[image_pair21]['scale'].value))    # apply scales estimated by DeMoN, average value by exchanging pair order;
                    # scale=1/transScaleColmap)   # apply scales calculated from colmap global/relative poses
                    # # scale=transScaleTheia)   # apply scales calculated from colmap global/relative poses
                    # scale=1/transScaleTheia)   # apply scales calculated from colmap global/relative poses

                    # scale=1)  # a complete model; no scaling because the depths are already in the correct scale

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
        if it>=2000:
            break


    appendFilterPC.Update()

    # plot the scatter 2D data of scale records, to find out the correlation between the predicted scales and the calculated scales from global SfM
    np.savetxt(os.path.join(outdir,'scale_record_DeMoN_Theia_Colmap.txt'), scaleRecordMat, fmt='%f')
    if True:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,1])
        # plt.scatter(1/scaleRecordMat[:,0],scaleRecordMat[:,1])
        plt.ylabel('scales calculated from Theia global SfM')
        plt.xlabel('scales predicted by DeMoN')
        # plt.xlabel('inv_scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,0],scaleRecordMat[:,2])
        # plt.scatter(1/scaleRecordMat[:,0],scaleRecordMat[:,1])
        plt.ylabel('scales calculated from Colmap')
        plt.xlabel('scales predicted by DeMoN')
        # plt.xlabel('inv_scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    if True:
        plt.scatter(scaleRecordMat[:,2],scaleRecordMat[:,1])
        # plt.scatter(1/scaleRecordMat[:,0],scaleRecordMat[:,1])
        plt.ylabel('scales calculated from Colmap')
        plt.xlabel('scales calculated from Theia global SfM')
        # plt.xlabel('inv_scales predicted by DeMoN')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

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

    # if initBool == False:
    #     renwin = vtk.vtkRenderWindow()
    #     renwin.SetWindowName("Point Cloud Viewer")
    #     renwin.SetSize(800,600)
    #     renwin.AddRenderer(renderer)
    #
    #     # An interactor
    #     interactor = vtk.vtkRenderWindowInteractor()
    #     interstyle = vtk.vtkInteractorStyleTrackballCamera()
    #     interactor.SetInteractorStyle(interstyle)
    #     interactor.SetRenderWindow(renwin)
    #
    #     # Start
    #     interactor.Initialize()
    #     interactor.Start()

    # SliderWidget.SetInteractor(interactor)
    # SliderWidget.SetRepresentation(SliderRepres)
    # # SliderWidget.KeyPressActivationOff()
    # # SliderWidget.SetAnimationModeToAnimate()
    # # SliderWidget.SetEnabled(True)
    # SliderWidget.AddObserver("EndInteractionEvent", vtkSliderCallback2)

    # return renderer
    renderer.Modified()

renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)
interactor = vtk.vtkRenderWindowInteractor()
#SliderRepres = vtk.vtkSliderRepresentation2D()
#SliderWidget = vtk.vtkSliderWidget()

def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()

def main():
    #global SliderRepres, SliderWidget, interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    global interactor, renderer, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT
    alpha = 0.18
    # renderer = visPointCloudInGlobalFrame(alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)
    print("alpha is set to ", alpha)

    visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)
    #renderer = visPointCloudInGlobalFrame(renderer, alpha, infile, ExhaustivePairInfile, data_format, target_K, w, h, cameras, images, TheiaGlobalPosesGT, TheiaRelativePosesGT, True)

    # axes = vtk.vtkAxesActor()
    # axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    # axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    # axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    # axes.SetCylinderRadius(0.03)
    # axes.SetShaftTypeToCylinder()
    # renderer.AddActor(axes)

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
    min = 0 #ImageViewer.GetSliceMin()
    max = 10 #ImageViewer.GetSliceMax()
    SliderRepres.SetMinimumValue(min)
    SliderRepres.SetMaximumValue(max)
    SliderRepres.SetValue(alpha)
    SliderRepres.SetTitleText("Alpha")
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
