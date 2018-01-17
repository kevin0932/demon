import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import h5py

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
#sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

from depthmotionnet.networks_original import *
from depthmotionnet.dataset_tools.view_io import *
from depthmotionnet.dataset_tools.view_tools import *
import colmap_utils as colmap
# from depthmotionnet.helpers import angleaxis_to_rotation_matrix
from depthmotionnet.vis import *
# from .view import View
# from .view_io import *
# from .view_tools import *

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


if tf.test.is_gpu_available(True):
    data_format='channels_first'
else: # running on cpu requires channels_last data format
    data_format='channels_last'

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

# ##################################################################
# ######## Add Colmap result as ground truth alternatives ##########
# ##################################################################
recondir = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_pedraza~hotel_room_pedraza_2012_nov_25/demon_prediction/images_demon/dense/'

camerasColmap = colmap.read_cameras_txt(os.path.join(recondir,'sparse','cameras.txt'))
imagesColmap = colmap.read_images_txt(os.path.join(recondir,'sparse','images.txt'))

print("imagesColmap = ", imagesColmap)

# ##################################################################
# ############Add input data from training data SUN3D###############
# ##################################################################
### SUN3D Training Data
# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.01m_to_0.1m.h5'
# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.1m_to_0.2m.h5'
inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.2m_to_0.4m.h5'
# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.4m_to_0.8m.h5'
# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_0.8m_to_1.6m.h5'
# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/sun3d_train_1.6m_to_infm.h5'


# SUN3D_datasetname = 'hotel_pedraza.hotel_room_pedraza_2012_nov_25-0000100'
SUN3D_datasetname = 'hotel_pedraza.hotel_room_pedraza_2012_nov_25-0000248'

# ### SUN3D testing data
# inputSUN3D_trainingdata = '/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/sun3d_test.h5'
#
# # SUN3D_datasetname = 'mit_w85h.h2_1-0006100_10to15cm'
# # SUN3D_datasetname = 'harvard_corridor_lounge.hv_lounge1_2-0000687_30to50cm'
# SUN3D_datasetname = 'harvard_robotics_lab.hv_s1_2-0000427_15to30cm'
# # SUN3D_datasetname = 'hotel_florence_jx.florence_hotel_stair_room_all-0017880_50toinfcm'
# # SUN3D_datasetname = 'hotel_florence_jx.florence_hotel_stair_room_all-0000710_5to10cm'
#
# # SUN3D_datasetname = 'hotel_florence_jx.florence_hotel_stair_room_all-0000070_30to50cm'
# # SUN3D_datasetname = 'hotel_florence_jx.florence_hotel_stair_room_all-0000070_50toinfcm'


data = h5py.File(inputSUN3D_trainingdata)

h5_group_v1 = data[SUN3D_datasetname+'/frames/t0/v0']
h5_group_v2 = data[SUN3D_datasetname+'/frames/t0/v1']



K1 = np.zeros((3,3),dtype=np.float64)
R1 = np.zeros((3,3),dtype=np.float64)
t1 = np.zeros((3,),dtype=np.float64)
K2 = np.zeros((3,3),dtype=np.float64)
R2 = np.zeros((3,3),dtype=np.float64)
t2 = np.zeros((3,),dtype=np.float64)
###### baseline 0.01m_0.1m ######
# h5_group_v1 = data['mit_w85_basement.wg_big_lounge_1-0000055/frames/t0/v0']
# h5_group_v2 = data['mit_w85_basement.wg_big_lounge_1-0000055/frames/t0/v1']
# h5_group_v1 = data['mit_w85_playroom.westgate_playroom_1-0000102/frames/t0/v0']
# h5_group_v2 = data['mit_w85_playroom.westgate_playroom_1-0000102/frames/t0/v1']
# h5_group_v1 = data['mit_w85_4.4_1-0000180/frames/t0/v0']
# h5_group_v2 = data['mit_w85_4.4_1-0000180/frames/t0/v1']
###### baseline 0.2m_0.4m ######
# h5_group_v1 = data['hotel_pittsburg.hotel_pittsburg_scan1_2012_dec_12-0000008/frames/t0/v0']
# h5_group_v2 = data['hotel_pittsburg.hotel_pittsburg_scan1_2012_dec_12-0000008/frames/t0/v1']
# h5_group_v1 = data['hotel_pedraza.hotel_room_pedraza_2012_nov_25-0000100/frames/t0/v0']
# h5_group_v2 = data['hotel_pedraza.hotel_room_pedraza_2012_nov_25-0000100/frames/t0/v1']
###### baseline 0.4m_0.8m ######
# h5_group_v1 = data['mit_w85k2.k1-0000116/frames/t0/v0']
# h5_group_v2 = data['mit_w85k2.k1-0000116/frames/t0/v1']
# h5_group_v1 = data['providence_station.providence_station-0000036/frames/t0/v0']
# h5_group_v2 = data['providence_station.providence_station-0000036/frames/t0/v1']
###### baseline 0.8_1.6m ######
# h5_group_v1 = data['hotel_beijing.beijing_hotel_2-0000022/frames/t0/v0']
# h5_group_v2 = data['hotel_beijing.beijing_hotel_2-0000022/frames/t0/v1']
# h5_group_v1 = data['hotel_beijing.beijing_hotel_2-0000069/frames/t0/v0']
# h5_group_v2 = data['hotel_beijing.beijing_hotel_2-0000069/frames/t0/v1']
# h5_group_v1 = data['hotel_beijing.beijing_hotel_2-0000099/frames/t0/v0']
# h5_group_v2 = data['hotel_beijing.beijing_hotel_2-0000099/frames/t0/v1']
# h5_group_v1 = data['hotel_beijing.beijing_hotel_2-0000127/frames/t0/v0']
# h5_group_v2 = data['hotel_beijing.beijing_hotel_2-0000127/frames/t0/v1']
###### baseline 1.6m_INFm ######
# h5_group_v1 = data['mit_32_g7_lounge.g7_lounge_1-0000088/frames/t0/v0']
# h5_group_v2 = data['mit_32_g7_lounge.g7_lounge_1-0000088/frames/t0/v1']
# h5_group_v1 = data['mit_32_g660.g660_1-0000008/frames/t0/v0']
# h5_group_v2 = data['mit_32_g660.g660_1-0000008/frames/t0/v1']

# DeMoN compatible size
w = 256
h = 192
normalized_intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5],np.float32)
target_K = np.eye(3)
target_K[0,0] = w*normalized_intrinsics[0]
target_K[1,1] = h*normalized_intrinsics[1]
target_K[0,2] = w*normalized_intrinsics[2]
target_K[1,2] = h*normalized_intrinsics[3]

K1,R1,t1 = read_camera_params(h5_group_v1['camera'])
view1 = read_view(h5_group_v1)
view1 = adjust_intrinsics(view1, target_K, w, h,)
img1 = view1.image

K2,R2,t2 = read_camera_params(h5_group_v2['camera'])
view2 = read_view(h5_group_v2)
view2 = adjust_intrinsics(view2, target_K, w, h,)
img2 = view2.image

# # a colormap and a normalization instance
# cmap = plt.cm.jet
# # plt.imshow(data['mit_w85_basement.wg_big_lounge_1-0000055/frames/t0/v0']["depth"].value, cmap='Greys')
# # plt.imshow(data['mit_w85_basement.wg_big_lounge_1-0000055/frames/t0/v0']["image"].value, cmap='Greys')
#
# print("camera params = ", data['mit_w85_basement.wg_big_lounge_1-0000055/frames/t0/v0']["camera"].value)
# print("image.size = ", curView.image.size)
# print("depth.shape = ", curView.depth.shape)
# print("depth = ", curView.depth)

########### Retrieve Colmap Depth Data
tmp_dict={}
tmpName1 = SUN3D_datasetname.split('.')
image_name1_inColmap = tmpName1[0]+'~'+tmpName1[1]+'_baseline_2_v0.JPG'
tmpName2 = SUN3D_datasetname.split('.')
image_name2_inColmap = tmpName2[0]+'~'+tmpName2[1]+'_baseline_2_v1.JPG'
print(image_name1_inColmap, "; ", image_name2_inColmap)
for image_id, image in imagesColmap.items():
    # print(image.name, "; ", image_name1, "; ", image_name2)
    if image.name == image_name1_inColmap:
        tmp_dict[image_id] = image
    if image.name == image_name2_inColmap:
        tmp_dict[image_id] = image

print("tmp_dict = ", tmp_dict)
# if len(tmp_dict)<2:
#     continue
tmp_views = colmap.create_views(camerasColmap, tmp_dict, os.path.join(recondir,'images'), os.path.join(recondir,'stereo','depth_maps'))
# print("tmp_views = ", tmp_views)
tmp_views[0] = adjust_intrinsics(tmp_views[0], target_K, w, h,)
tmp_views[1] = adjust_intrinsics(tmp_views[1], target_K, w, h,)


# depth1Colmap = tmp_views[0].depth
# depth2Colmap = tmp_views[1].depth
view1Colmap = tmp_views[0]
view2Colmap = tmp_views[1]
scaleColmap = np.linalg.norm(-np.dot(view2Colmap.R.T, view2Colmap.t)+np.dot(view1Colmap.R.T, view1Colmap.t))
print("scaleColmap = ", scaleColmap)

#### If use colmap result global poses or not?
R1 = view1Colmap.R
t1 = view1Colmap.t
R2 = view2Colmap.R
t2 = view2Colmap.t
# ##################################################################


# # read data
# # img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
# # img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))
# # img1 = Image.open(os.path.join(examples_dir,'0000131-000004390596.jpg'))
# # img2 = Image.open(os.path.join(examples_dir,'0000151-000005060916.jpg'))
# img1 = Image.open(os.path.join(examples_dir,'0000231-000007742196.jpg'))
# img2 = Image.open(os.path.join(examples_dir,'0000251-000008412516.jpg'))

# print("img1.size = ", img1.size)



input_data = prepare_input_data(img1,img2,data_format)

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


####### image1---image2
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
rotation = result['predict_rotation']
print('rotation = ', rotation)
translation = result['predict_translation']
print('translation = ', translation)
predict_scale = result['predict_scale']
print('predict_scale = ', predict_scale)
result = refine_net.eval(input_data['image1'],result['predict_depth2'])


# plt.figure()
# plt.subplot(211) # equivalent to: plt.subplot(2, 2, 1)
# plt.imshow(view1.depth, cmap='Greys')
# plt.subplot(212)
# plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
# plt.show()

####### image2---image1
input_data21 = prepare_input_data(img2,img1,data_format)
# run the network
result21 = bootstrap_net.eval(input_data21['image_pair'], input_data21['image2_2'])
for i in range(3):
    result21 = iterative_net.eval(
        input_data21['image_pair'],
        input_data21['image2_2'],
        result21['predict_depth2'],
        result21['predict_normal2'],
        result21['predict_rotation'],
        result21['predict_translation']
    )
rotation21 = result21['predict_rotation']
print('rotation21 = ', rotation21)
translation21 = result21['predict_translation']
print('translation21 = ', translation21)
predict_scale21 = result21['predict_scale']
print('predict_scale21 = ', predict_scale21)
result21 = refine_net.eval(input_data21['image1'],result21['predict_depth2'])


# plt.figure()
# plt.subplot(121) # equivalent to: plt.subplot(2, 2, 1)
# plt.imshow(view2.depth, cmap='Greys')
# plt.subplot(122)
# plt.imshow(result21['predict_depth0'].squeeze(), cmap='Greys')
# plt.show()


plt.figure()
plt.subplot(241) # equivalent to: plt.subplot(2, 2, 1)
plt.imshow(view1.image)
plt.title('RGB Image 1')
plt.subplot(242) # equivalent to: plt.subplot(2, 2, 1)
plt.imshow(1/view1.depth, cmap='Greys')
plt.title('SUN3D Ground Truth Depth Image 1')
plt.subplot(243)
plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
plt.title('SUN3D DeMoN Depth Image 1')
plt.subplot(244) # equivalent to: plt.subplot(2, 2, 1)
plt.imshow(view1Colmap.depth, cmap='Greys')
plt.title('SUN3D Colmap Depth Image 1')
plt.subplot(245) # equivalent to: plt.subplot(2, 2, 1)
plt.imshow(view2.image)
plt.title('RGB Image 2')
plt.subplot(246) # equivalent to: plt.subplot(2, 2, 1)
plt.imshow(1/view2.depth, cmap='Greys')
plt.title('SUN3D Ground Truth Depth Image 2')
plt.subplot(247)
plt.imshow(result21['predict_depth0'].squeeze(), cmap='Greys')
plt.title('SUN3D DeMoN Depth Image 2')
plt.subplot(248) # equivalent to: plt.subplot(2, 2, 1)
plt.imshow(view2Colmap.depth, cmap='Greys')
plt.title('SUN3D Colmap Depth Image 2')
plt.show()

print("t1 = ", t1, "; t2 = ", t2)

from depthmotionnet.vis import *
import vtk
tmpPC1 = visualize_prediction(
# inverse_depth=result['predict_depth0'],
inverse_depth=1/view1.depth,
# inverse_depth=1/view1Colmap.depth,
image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
R1=R1,
t1=t1,
rotation=(rotation),
translation=translation[0],
# scale=predict_scale)
# scale=scaleColmap)
scale=1)

print("translation.shape = ", translation.shape)
print("translation[0].shape = ", translation[0].shape)
tmpPC2 = visualize_prediction(
# inverse_depth=result21['predict_depth0'],
inverse_depth=1/view2.depth,
# inverse_depth=1/view2Colmap.depth,
image=input_data21['image_pair'][0,0:3] if data_format=='channels_first' else input_data21['image_pair'].transpose([0,3,1,2])[0,0:3],
# R1=angleaxis_to_rotation_matrix(rotation[0]),
# t1=translation[0],
R1=R2,
t1=t2,
rotation=(rotation),
translation=translation[0],
# scale=predict_scale21)
# scale=scaleColmap)
scale=1)

tmpPC = {}
tmpPC['points'] = np.concatenate((tmpPC1['points'],tmpPC2['points']),axis=0)
# if 'colors' in tmpPC:
tmpPC['colors'] = np.concatenate((tmpPC1['colors'],tmpPC2['colors']),axis=0)

# tmpPC = tmpPC2

print("tmpPC1['points'].shape = ", tmpPC1['points'].shape)
print("tmpPC2['points'].shape = ", tmpPC2['points'].shape)
print("tmpPC['points'].shape = ", tmpPC['points'].shape)
print("tmpPC1['colors'].shape = ", tmpPC1['colors'].shape)
print("tmpPC2['colors'].shape = ", tmpPC2['colors'].shape)
print("tmpPC['colors'].shape = ", tmpPC['colors'].shape)
# export all point clouds in the same global coordinate to a local .ply file (for external visualization)
# # output_prefix = './'
# pointcloud_polydata = create_pointcloud_polydata(
# points=tmpPC['points'],
# # colors=tmpPC['colors'] if 'colors' in tmpPC else None,
# colors=tmpPC['colors'],
# )

renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)
pointcloud1_actor = create_pointcloud_actor(
points=tmpPC1['points'],
colors=tmpPC1['colors'] if 'colors' in tmpPC1 else None,
)
renderer.AddActor(pointcloud1_actor)
pointcloud2_actor = create_pointcloud_actor(
points=tmpPC2['points'],
colors=tmpPC2['colors'] if 'colors' in tmpPC2 else None,
)
renderer.AddActor(pointcloud2_actor)

# cam1_actor = create_camera_actor(np.eye(3), [0,0,0])
cam1_actor = create_camera_actor(R1, t1)
renderer.AddActor(cam1_actor)
# cam2_actor = create_camera_actor(angleaxis_to_rotation_matrix(rotation[0]), translation[0])
cam2_actor = create_camera_actor(R2, t2)
renderer.AddActor(cam2_actor)

###############################################################
appendFilterModel = vtk.vtkAppendPolyData()
cam1_polydata = create_camera_polydata(R1, t1, True)
cam2_polydata = create_camera_polydata(R2, t2, True)
pointcloud1_polydata = create_pointcloud_polydata(
points=tmpPC1['points'],
colors=tmpPC1['colors'] if 'colors' in tmpPC1 else None,
)
pointcloud2_polydata = create_pointcloud_polydata(
points=tmpPC2['points'],
colors=tmpPC2['colors'] if 'colors' in tmpPC2 else None,
)
appendFilterModel.AddInputData(pointcloud1_polydata)
appendFilterModel.AddInputData(pointcloud2_polydata)
appendFilterModel.AddInputData(cam1_polydata)
appendFilterModel.AddInputData(cam2_polydata)
appendFilterModel.Update()

plywriter = vtk.vtkPLYWriter()
plywriter.SetFileName(('DeMoN_pair_pointcloud_'+inputSUN3D_trainingdata.split('/')[-1][:-3]+'_'+SUN3D_datasetname+'_'+str(predict_scale)+'_'+str(predict_scale21)+'.ply'))
# plywriter.SetFileName(('SUN3D_GT_pair_pointcloud_'+inputSUN3D_trainingdata.split('/')[-1][:-3]+'_'+SUN3D_datasetname+'_'+str(predict_scale)+'_'+str(predict_scale21)+'.ply'))
# plywriter.SetInputData(pointcloud_polydata)
plywriter.SetInputData(appendFilterModel.GetOutput())
# plywriter.SetFileTypeToASCII()
plywriter.SetArrayName('Colors')
plywriter.Write()
###############################################################
axes = vtk.vtkAxesActor()
axes.GetXAxisCaptionActor2D().SetHeight(0.05)
axes.GetYAxisCaptionActor2D().SetHeight(0.05)
axes.GetZAxisCaptionActor2D().SetHeight(0.05)
axes.SetCylinderRadius(0.03)
axes.SetShaftTypeToCylinder()
renderer.AddActor(axes)

renwin = vtk.vtkRenderWindow()
renwin.SetWindowName("Point Cloud Viewer")
renwin.SetSize(800,600)
renwin.AddRenderer(renderer)


# An interactor
interactor = vtk.vtkRenderWindowInteractor()
interstyle = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(interstyle)
interactor.SetRenderWindow(renwin)

# Start
interactor.Initialize()
interactor.Start()


# # try to visualize the point cloud
# try:
#     from depthmotionnet.vis import *
#     import vtk
#     tmpPC1 = visualize_prediction(
#     # visualize_prediction(
#         inverse_depth=result['predict_depth0'],
#         image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
#         rotation=(rotation),
#         translation=translation[0],
#         scale=predict_scale)
#         # scale=1)
#
#     print("translation.shape = ", translation.shape)
#     print("translation[0].shape = ", translation[0].shape)
#     tmpPC2 = visualize_prediction(
#     # visualize_prediction(
#         inverse_depth=result21['predict_depth0'],
#         image=input_data21['image_pair'][0,0:3] if data_format=='channels_first' else input_data21['image_pair'].transpose([0,3,1,2])[0,0:3],
#         R1=angleaxis_to_rotation_matrix(rotation[0]),
#         t1=translation[0],
#         rotation=(rotation),
#         translation=translation[0],
#         scale=predict_scale21)
#         # scale=1)
#
#     tmpPC = {}
#     tmpPC['points'] = np.concatenate((tmpPC1['points'],tmpPC2['points']),axis=0)
#     # if 'colors' in tmpPC:
#     tmpPC['colors'] = np.concatenate((tmpPC1['colors'],tmpPC2['colors']),axis=0)
#
#     # tmpPC = tmpPC2
#
#     print("tmpPC1['points'].shape = ", tmpPC1['points'].shape)
#     print("tmpPC2['points'].shape = ", tmpPC2['points'].shape)
#     print("tmpPC['points'].shape = ", tmpPC['points'].shape)
#     # export all point clouds in the same global coordinate to a local .ply file (for external visualization)
#     # output_prefix = './'
#     pointcloud_polydata = create_pointcloud_polydata(
#         points=tmpPC['points'],
#         # colors=tmpPC['colors'] if 'colors' in tmpPC else None,
#         colors=tmpPC['colors'],
#         )
#
#     appendFilterModel = vtk.vtkAppendPolyData()
#     cam1_polydata = create_camera_polydata(np.eye(3), [0,0,0], True)
#     cam2_polydata = create_camera_polydata(angleaxis_to_rotation_matrix(rotation[0]), translation[0], True)
#     appendFilterModel.AddInputData(pointcloud_polydata)
#     appendFilterModel.AddInputData(cam1_polydata)
#     appendFilterModel.AddInputData(cam2_polydata)
#     appendFilterModel.Update()
#
#     plywriter = vtk.vtkPLYWriter()
#     plywriter.SetFileName('DeMoN_Sculpture_Example_pointcloud.ply')
#     # plywriter.SetInputData(pointcloud_polydata)
#     plywriter.SetInputData(appendFilterModel.GetOutput())
#     # plywriter.SetFileTypeToASCII()
#     plywriter.SetArrayName('colors')
#     plywriter.Write()
#
#     renderer = vtk.vtkRenderer()
#     renderer.SetBackground(0, 0, 0)
#     pointcloud_actor = create_pointcloud_actor(
#        points=tmpPC1['points'],
#        colors=tmpPC1['colors'] if 'colors' in tmpPC1 else None,
#        )
#     renderer.AddActor(pointcloud_actor)
#     pointcloud_actor = create_pointcloud_actor(
#        points=tmpPC2['points'],
#        colors=tmpPC2['colors'] if 'colors' in tmpPC2 else None,
#        )
#     renderer.AddActor(pointcloud_actor)
#
#     cam1_actor = create_camera_actor(np.eye(3), [0,0,0])
#     renderer.AddActor(cam1_actor)
#     cam2_actor = create_camera_actor(angleaxis_to_rotation_matrix(rotation[0]), translation[0])
#     renderer.AddActor(cam2_actor)
#
#
#     axes = vtk.vtkAxesActor()
#     axes.GetXAxisCaptionActor2D().SetHeight(0.05)
#     axes.GetYAxisCaptionActor2D().SetHeight(0.05)
#     axes.GetZAxisCaptionActor2D().SetHeight(0.05)
#     axes.SetCylinderRadius(0.03)
#     axes.SetShaftTypeToCylinder()
#     renderer.AddActor(axes)
#
#     renwin = vtk.vtkRenderWindow()
#     renwin.SetWindowName("Point Cloud Viewer")
#     renwin.SetSize(800,600)
#     renwin.AddRenderer(renderer)
#
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
#
# except ImportError as err:
#     print("Cannot visualize as pointcloud.", err)
