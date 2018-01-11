import tensorflow as tf
from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.helpers import angleaxis_to_rotation_matrix
import colmap_utils as colmap
from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
import math
import h5py
import os
import cv2

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


# weights_dir = '/home/ummenhof/projects/demon/weights'
weights_dir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/weights'
# outdir = '/home/kevin/DeMoN_Prediction/south_building'
# infile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions.h5'
## infile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions_v1_05012018.h5'

# outdir = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction'
# outdir = '/home/kevin/ThesisDATA/person-hall/demon_prediction'
# outdir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction"
# outdir = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction"
outdir = "/home/kevin/JohannesCode/ws1/demon_prediction"
# infile = '/home/kevin/ThesisDATA/gerrard-hall/demon_prediction/gerrard_hall_predictions.h5'
# infile = '/home/kevin/ThesisDATA/person-hall/demon_prediction/person_hall_predictions.h5'
# infile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/barcelona_Dataset/demon_prediction/CVG_barcelona_predictions.h5"
# infile = "/home/kevin/ThesisDATA/CVG_Datasets_3Dsymmetric/redmond_Dataset/demon_prediction/CVG_redmond_predictions.h5"
infile = "/home/kevin/JohannesCode/ws1/demon_prediction/kevin_southbuilding_predictions_08012018.h5"


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

views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))

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

# # w_large = 8*w
# # h_large = 8*h
# # w_large = 12.1*w
# # h_large = 12.05*h
#
# # w_large = 7.8125*w
# # h_large = 7.8125*h
# # #
#
# # # gerrard-hall, person-hall
# w_large = 7.8125*w
# h_large = 6.84896*h
#
# # # # barcelona cvg
# # w_large = 7.8125*w
# # h_large = 7.8125*h  # barcelona cvg
#
# # # # redmon cvg
# # w_large = 7.8125*w
# # h_large = 5.7917*h  # cvg redmond
#
# target_K_large = np.eye(3)
# target_K_large[0,0] = w_large*normalized_intrinsics[0]
# target_K_large[1,1] = h_large*normalized_intrinsics[1]
# target_K_large[0,2] = w_large*normalized_intrinsics[2]
# target_K_large[1,2] = h_large*normalized_intrinsics[3]

if False:
    views = []
    id_image_list = []
    cnt = 0
    for image_id, image in images.items():
        tmp_dict = {image_id: image}
        tmp_views = colmap.create_views(cameras, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
        # print("type(tmp_views) = ", type(tmp_views))
        # print("(tmp_views) = ", (tmp_views))
        new_v = adjust_intrinsics(tmp_views[0], target_K, w, h,)
        # print("type(new_v) = ", type(new_v))
        # new_v_large = adjust_intrinsics(tmp_views[0], target_K_large, w_large, h_large,)
        # new_v_large.image.save(os.path.join(outimagedir_large,image.name))
        new_v.image.save(os.path.join(outimagedir_small,image.name))
        if not new_v is None:
            id_image_list.append((image_id,image))
            views.append(new_v)
            # Kevin: visualization
            # visualize_views(tmp_views)
            cnt += 1
        # if cnt >= 2:
        #     break
    distances = compute_view_distances(views)

    pairs_to_compute = set()

    for idx, view in enumerate(views):
        print(idx, len(views), end=' ')
        view_dists = distances[idx]

        # find k nearest neighbours
        neighbours = sorted([(d,i) for i,d in enumerate(view_dists)])[1:knn+1]
        good_neighbours = []
        for _, neighbour_idx in neighbours:
            if not (idx, neighbour_idx) in pairs_to_compute:
                neighbour_view = views[neighbour_idx]
                if compute_view_angle(view, neighbour_view) < max_angle:
                    overlap = compute_view_overlap(view, neighbour_view)
                    if overlap > min_overlap_ratio:
                        good_neighbours.append(neighbour_idx)

        print(len(good_neighbours))
        for neighbour_idx in good_neighbours:
            pairs_to_compute.add((idx, neighbour_idx))
            pass

    len(pairs_to_compute)

if True:
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

    data = h5py.File(infile)

    # for image_id, image in images.items():
    #     print(image_id, image)
    # for i, pair in enumerate(pairs_to_compute):
    #     print(i, len(pairs_to_compute))
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
        tmp_views = adjust_intrinsics(tmp_views, target_K, w, h,)

        view1 = tmp_views[0]
        view2 = tmp_views[1]

        # print("view1 = ", view1)

        # input_data = prepare_input_data(view1.image, view2.image, data_format)
        #
        # # run the network
        # result = bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])
        # for i in range(3):
        #     result = iterative_net.eval(
        #         input_data['image_pair'],
        #         input_data['image2_2'],
        #         result['predict_depth2'],
        #         result['predict_normal2'],
        #         result['predict_rotation'],
        #         result['predict_translation']
        #     )
        #
        # result = refine_net.eval(input_data['image1'],result['predict_depth2'])

        # # a colormap and a normalization instance
        # cmap = plt.cm.jet
        # # plt.imshow(data[image_pair12]["depth_upsampled"], cmap='Greys')
        # plt.imshow(view1.depth, cmap='Greys')

        vis2 = cv2.cvtColor(view1.depth, cv2.COLOR_GRAY2BGR)
        #Displayed the image
        cv2.imshow("WindowNameHere", vis2)
        cv2.waitKey(0)

        # vis2 = cv2.cvtColor(view2.depth, cv2.COLOR_GRAY2BGR)
        # #Displayed the image
        # cv2.imshow("WindowNameHere", vis2)
        # cv2.waitKey(0)

        break
