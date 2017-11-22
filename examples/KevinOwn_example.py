import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

from depthmotionnet.networks_original import *

def angleaxis_to_rotation_matrix(aa):
    """Converts the 3 element angle axis representation to a 3x3 rotation matrix

    aa: numpy.ndarray with 1 dimension and 3 elements
    Returns a 3x3 numpy.ndarray
    """
    if len(aa.shape) == 2:
        # aa = np.reshape(aa, (aa.shape[1]))
        aa = np.squeeze(aa)
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

# read data
# img1 = Image.open(os.path.join(examples_dir,'P1180170_processed.JPG'))
# img2 = Image.open(os.path.join(examples_dir,'P1180171_processed.JPG'))
# img1 = Image.open(os.path.join(examples_dir,'P1180170.JPG'))
# img2 = Image.open(os.path.join(examples_dir,'P1180171.JPG'))
img1 = Image.open(os.path.join(examples_dir,'P1180151.JPG'))
img2 = Image.open(os.path.join(examples_dir,'P1180153.JPG'))
# img1 = Image.open(os.path.join(examples_dir,'P1180153.JPG'))
# img2 = Image.open(os.path.join(examples_dir,'P1180151.JPG'))
# img1 = Image.open(os.path.join(examples_dir,'PIC004.JPG'))
# img2 = Image.open(os.path.join(examples_dir,'PIC005.JPG'))

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
print(len(rotation.shape))
print(len(rotation[:].shape))
print('rotation matrix = ', angleaxis_to_rotation_matrix(rotation))
translation = result['predict_translation']
print('translation = ', translation)
print(type(result))
print(result.keys())

result = refine_net.eval(input_data['image1'],result['predict_depth2'])

print(type(result))
# for k, v in result.items():
#     print(k, v)
print(result.keys())
plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
plt.show()
# plt.imshow(result['predict_depth_upsampled'].squeeze(), cmap='Greys')
# plt.show()
# plt.imshow(result['predict_depth2'].squeeze(), cmap='Greys')
# plt.show()

# try to visualize the point cloud
try:
    from depthmotionnet.vis import *
    visualize_prediction(
        inverse_depth=result['predict_depth0'],
        image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
        rotation=rotation,
        translation=translation)
except ImportError as err:
    print("Cannot visualize as pointcloud.", err)
