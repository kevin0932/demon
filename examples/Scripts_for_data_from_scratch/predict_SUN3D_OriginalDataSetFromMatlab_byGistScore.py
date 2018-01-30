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

from skimage.measure import compare_ssim as ssim #structural_similarity as ssim
import gist


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


weights_dir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/weights'

outdir = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85k1~living_room_night/demon_prediction"
outfile = "/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85k1~living_room_night/demon_prediction/demon_mit_w85k1~living_room_night.h5"


outimagedir_small = os.path.join(outdir,'images_demon_small')
outimagedir_large = os.path.join(outdir,'images_demon')
os.makedirs(outdir, exist_ok=True)
os.makedirs(outimagedir_small, exist_ok=True)
os.makedirs(outimagedir_large, exist_ok=True)
os.makedirs(os.path.join(outdir,'graydepthmap'), exist_ok=True)
os.makedirs(os.path.join(outdir,'vizdepthmap'), exist_ok=True)


# recondir = '/home/kevin/JohannesCode/ws1/dense/0/'
# cameras = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
# images = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))
# views = colmap.create_views(cameras, images, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))


# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m.h5'
# data = h5py.File(inputSUN3D_trainingdata)
# # h5_group_v1 = data[SUN3D_datasetname+'/frames/t0/v0']
# # h5_group_v2 = data[SUN3D_datasetname+'/frames/t0/v1']
inputSUN3D_trainFilePaths = []
inputSUN3D_trainFilePaths.append('/media/kevin/SamsungT5_F/ThesisDATA/SUN3D/mit_w85k1~living_room_night/GT_mit_w85k1~living_room_night.h5')

# knn = 15 # 5
# max_angle = 90*math.pi/180  # 60*math.pi/180
# min_overlap_ratio = 0.4     # 0.5
knn = 5
max_angle = 60*math.pi/180  # 60*math.pi/180
min_overlap_ratio = 0.5     # 0.5
w = 256
h = 192
normalized_intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5],np.float32)
target_K = np.eye(3)
target_K[0,0] = w*normalized_intrinsics[0]
target_K[1,1] = h*normalized_intrinsics[1]
target_K[0,2] = w*normalized_intrinsics[2]
target_K[1,2] = h*normalized_intrinsics[3]

def computePairGistDistance(image1, image2):
    GistDescriptor1 = gist.extract(np.array(image1))
    GistDescriptor2 = gist.extract(np.array(image2))
    # print("gist descriptor = ", GistDescriptor)
    # print("gist descriptor.shape = ", GistDescriptor.shape)
    return np.linalg.norm(GistDescriptor1-GistDescriptor2)

if True:
    views = []
    id_image_list = []
    cnt = 0
    for fileIdx in range(len(inputSUN3D_trainFilePaths)):
        inputSUN3D_trainingdata = inputSUN3D_trainFilePaths[fileIdx]
        print("processing ", inputSUN3D_trainingdata)
        data = h5py.File(inputSUN3D_trainingdata)
        for h5key in data.keys():
            if h5key.split('-')[0] == 'mit_w85k1~living_room_night':
                image_name = h5key
                print(h5key, " ====> ", image_name)
                h5_group_tmp = data[h5key]
                tmp_view = read_view(h5_group_tmp, lmuFreiburgFormat=False)
                new_v = adjust_intrinsics(tmp_view, target_K, w, h,)
                # print("type(new_v) = ", type(new_v))
                # GistDescriptor = gist.extract(np.array(new_v.image))
                # print("gist descriptor = ", GistDescriptor)
                # print("gist descriptor.shape = ", GistDescriptor.shape)
                if not new_v is None:
                    dupFlag = False
                    for prevIdx in range(len(views)):
                        opencvImage1 = cv2.cvtColor(np.array(views[prevIdx].image), cv2.COLOR_RGB2BGR)
                        opencvImage2 = cv2.cvtColor(np.array(new_v.image), cv2.COLOR_RGB2BGR)
                        pair_ssim_val = ssim(opencvImage1,opencvImage2, multichannel=True)
                        # new_v.image.show()
                        # views[prevIdx].image.show()
                        # print("pair_ssim_val = ", pair_ssim_val)
                        if pair_ssim_val ==1:
                            # print("one duplicated image pair is detected and the new_v will be skipped!")
                            dupFlag = True
                        # else:
                        #     print("ssim = ", pair_ssim_val)

                    if dupFlag == False:
                        v0_save_name_forIndexing = image_name+'.JPG'
                        tmp_view.image.save(os.path.join(outimagedir_large,(v0_save_name_forIndexing)))
                        new_v.image.save(os.path.join(outimagedir_small,(v0_save_name_forIndexing)))
                        # id_image_list.append((cnt,image))
                        id_image_list.append((v0_save_name_forIndexing))
                        views.append(new_v)
                        # Kevin: visualization
                        # visualize_views(tmp_views)
                        cnt += 1

                    dupFlag = False
                # if cnt >= 10:
                #     break

    distances = compute_view_distances(views)

    pairs_to_compute = set()
    allpairsRecord = set()
    gistScore_allNeighbours_record = []
    gistScore_goodNeighbours_record = []


    for idx, view in enumerate(views):
        for idx2, view2 in enumerate(views):
            if not (idx, idx2) in allpairsRecord:
                gistScore_allNeighbours = computePairGistDistance(views[idx].image, views[idx2].image)
                gistScore_allNeighbours_record.append(gistScore_allNeighbours)
                allpairsRecord.add((idx, idx2))

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
            gistScore_goodNeighbours = computePairGistDistance(views[idx].image, views[neighbour_idx].image)
            gistScore_goodNeighbours_record.append(gistScore_goodNeighbours)
            pass

    print(len(allpairsRecord))
    print(len(pairs_to_compute))
    gistScore_allNeighbours_record = np.array(gistScore_allNeighbours_record)
    gistScore_goodNeighbours_record = np.array(gistScore_goodNeighbours_record)
    print((gistScore_allNeighbours_record.shape))
    print(np.nanmean(gistScore_allNeighbours_record)," ",np.nanmedian(gistScore_allNeighbours_record)," ",np.nanstd(gistScore_allNeighbours_record))
    print((gistScore_goodNeighbours_record.shape))
    print(np.nanmean(gistScore_goodNeighbours_record)," ",np.nanmedian(gistScore_goodNeighbours_record)," ",np.nanstd(gistScore_goodNeighbours_record))

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
    pair_ssim_values = []
    for i, pair in enumerate(pairs_to_compute):
        print(i, len(pairs_to_compute))
        view1 = views[pair[0]]
        view2 = views[pair[1]]

        ### also show the SSIM score
        opencvImage1 = cv2.cvtColor(np.array(view1.image), cv2.COLOR_RGB2BGR)
        opencvImage2 = cv2.cvtColor(np.array(view2.image), cv2.COLOR_RGB2BGR)
        pair_ssim_val = ssim(opencvImage1,opencvImage2, multichannel=True)
        pair_ssim_values.append(pair_ssim_val)
        print("pair_ssim_val = ", pair_ssim_val)

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
        group = out.require_group('{0}---{1}'.format(id_image_list[pair[0]], id_image_list[pair[1]]))
        group['normalized_intrinsics'] = normalized_intrinsics
        group['depth'] = result['predict_depth2'].squeeze()
        group['rotation'] = angleaxis_to_rotation_matrix(result['predict_rotation'].squeeze()).astype(np.float32)
        group['translation'] = result['predict_translation'].squeeze().astype(np.float32)
        group['flow'] = result['predict_flow2'].squeeze()
        ### save the learned scale to the output as well
        group['scale'] = result['predict_scale'].squeeze().astype(np.float32)

        result = refine_net.eval(input_data['image1'],result['predict_depth2'])

        group['depth_upsampled'] = result['predict_depth0'].squeeze()

        # # a colormap and a normalization instance
        # cmap = plt.cm.jet
        # plt.imsave(os.path.join(outdir, "graydepthmap", id_image_list[pair[0]][1].name + "---" + id_image_list[pair[1]][1].name), result['predict_depth0'].squeeze(), cmap='Greys')
        # plt.imsave(os.path.join(outdir, "vizdepthmap", id_image_list[pair[0]][1].name + "---" + id_image_list[pair[1]][1].name), result['predict_depth0'].squeeze(), cmap=cmap)
        # plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
        # plt.imshow(result['predict_depth0'].squeeze(), cmap=cmap)

        # view1.image.show()
        # view2.image.show()
        # vis2 = cv2.cvtColor(result['predict_depth0'].squeeze(), cv2.COLOR_GRAY2BGR)
        # #Displayed the image
        # cv2.imshow("WindowNameHere", vis2)
        # cv2.waitKey(0)

    out.close()

    np.savetxt(os.path.join(outdir,'pair_ssim_values.txt'), np.array(pair_ssim_values))

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
