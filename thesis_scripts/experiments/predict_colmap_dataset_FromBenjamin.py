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
# outfile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions.h5'
## outfile = '/home/kevin/DeMoN_Prediction/south_building/south_building_predictions_v1_05012018.h5'

outdir = '/home/kevin/ThesisDATA/ToyDataset_Desk/demon_predictions'
outfile = '/home/kevin/ThesisDATA/ToyDataset_Desk/demon_predictions/toyexample_labIndoor_predictions.h5'

outimagedir_small = os.path.join(outdir,'images_demon_small')
outimagedir_large = os.path.join(outdir,'images_demon')
os.makedirs(outdir, exist_ok=True)
os.makedirs(outimagedir_small, exist_ok=True)
os.makedirs(outimagedir_large, exist_ok=True)
os.makedirs(os.path.join(outdir,'graydepthmap'), exist_ok=True)
os.makedirs(os.path.join(outdir,'vizdepthmap'), exist_ok=True)




# recondir = '/misc/lmbraid12/depthmotionnet/datasets/mvs_colmap/south-building/mvs/'
# recondir = '/home/kevin/JohannesCode/ws1/'
recondir = '/home/kevin/ThesisDATA/ToyDataset_Desk/dense/'

cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))

images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))

views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))

knn = 5
max_angle = 60*math.pi/180
min_overlap_ratio = 0.5
w = 256
h = 192
normalized_intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5],np.float32)
target_K = np.eye(3)
target_K[0,0] = w*normalized_intrinsics[0]
target_K[1,1] = h*normalized_intrinsics[1]
target_K[0,2] = w*normalized_intrinsics[2]
target_K[1,2] = h*normalized_intrinsics[3]

# w_large = 8*w
# h_large = 8*h
# w_large = 12.1*w
# h_large = 12.05*h
w_large = 7.8125*w
h_large = 7.8125*h
target_K_large = np.eye(3)
target_K_large[0,0] = w_large*normalized_intrinsics[0]
target_K_large[1,1] = h_large*normalized_intrinsics[1]
target_K_large[0,2] = w_large*normalized_intrinsics[2]
target_K_large[1,2] = h_large*normalized_intrinsics[3]

if True:
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
        new_v_large = adjust_intrinsics(tmp_views[0], target_K_large, w_large, h_large,)
        new_v_large.image.save(os.path.join(outimagedir_large,image.name))
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

    out = h5py.File(outfile)

    for i, pair in enumerate(pairs_to_compute):
        print(i, len(pairs_to_compute))
        view1 = views[pair[0]]
        view2 = views[pair[1]]
        input_data = prepare_input_data(view1.image, view2.image, data_format)

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
        group = out.require_group('{0}---{1}'.format(id_image_list[pair[0]][1].name, id_image_list[pair[1]][1].name))
        group['normalized_intrinsics'] = normalized_intrinsics
        group['depth'] = result['predict_depth2'].squeeze()
        group['rotation'] = angleaxis_to_rotation_matrix(result['predict_rotation'].squeeze()).astype(np.float32)
        group['translation'] = result['predict_translation'].squeeze().astype(np.float32)
        group['flow'] = result['predict_flow2'].squeeze()

        result = refine_net.eval(input_data['image1'],result['predict_depth2'])

        group['depth_upsampled'] = result['predict_depth0'].squeeze()

        # # a colormap and a normalization instance
        # cmap = plt.cm.jet
        # plt.imsave(os.path.join(outdir, "graydepthmap", id_image_list[pair[0]][1].name + "---" + id_image_list[pair[1]][1].name), result['predict_depth0'].squeeze(), cmap='Greys')
        # plt.imsave(os.path.join(outdir, "vizdepthmap", id_image_list[pair[0]][1].name + "---" + id_image_list[pair[1]][1].name), result['predict_depth0'].squeeze(), cmap=cmap)
        # plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
        # plt.imshow(result['predict_depth0'].squeeze(), cmap=cmap)

        # vis2 = cv2.cvtColor(result['predict_depth0'].squeeze(), cv2.COLOR_GRAY2BGR)
        # #Displayed the image
        # cv2.imshow("WindowNameHere", vis2)
        # cv2.waitKey(0)

    out.close()

if False:
# if True:
    print("visualize some depth maps")
    f = h5py.File(outfile, 'r')

    # visualize some depth maps
    d = None
    d_list = []
    # for k in f:
    for k in f.keys():
        if d is None:
            d = f[k]['depth'][:]
            print(d)
            print(d.shape)
            # print(type(d))
            # # plt.figure(figsize=(20,18))
            # plt.imshow(d, cmap='Greys')
            imgplot = plt.imshow(d)
            plt.colorbar()
        elif d.shape[1] < 64*16:
            d = np.concatenate((d, f[k]['depth'][:]),axis=1)
        else:
            d_list.append(d.copy())
            d = None
            if len(d_list) > 20:
                break

    print("len(d_list) = ", len(d_list))
    d = None
    for dd in d_list:
        if d is None:
            d = dd
        else:
            d = np.concatenate((d,dd),axis=0)

    # plt.figure(figsize=(20,18))
    # plt.imshow(d)
    # # plt.imsave(d)

    f.close()
