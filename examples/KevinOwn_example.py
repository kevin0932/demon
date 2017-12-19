import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys

"""
    flowfilter.plot
    ---------------
    Module containing functions to plot flow fields.
    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import pkg_resources

#import numpy as np
import scipy.ndimage.interpolation as interp
import scipy.misc as misc


#__all__ = ['flowToColor', 'colorWheel']

# load color wheel image
#colorWheel = misc.imread(pkg_resources.resource_filename('flowfilter.rsc', 'colorWheel.png'), flatten=False)
colorWheel = misc.imread('colorWheel.png', flatten=False)

# RGB components of colorwheel
_colorWheel_R = np.copy(colorWheel[...,0])
_colorWheel_G = np.copy(colorWheel[...,1])
_colorWheel_B = np.copy(colorWheel[...,2])

def flowToColor(flow, maxflow=1.0):
    """Returns the color wheel encoded version of the flow field.
    Parameters
    ----------
    flow : ndarray
        Optical flow field.
    maxflow : float, optional
        Maximum flow magnitude. Defaults to 1.0.
    Returns
    -------
    flowColor : ndarray
        RGB color encoding of input optical flow.
    Raises
    ------
    ValueError : if maxflow <= 0.0
    """

    if maxflow <= 0.0: raise ValueError('maxflow should be greater than zero')

    # height and width of color wheel texture
    h, w = colorWheel.shape[0:2]

    # scale optical flow to lie in range [0, 1]
    flow_scaled = (flow + maxflow) / float(2*maxflow)

    # re-scale to lie in range [0, w) and [0, h)
    flow_scaled[:,:,0] *= (w-1)
    flow_scaled[:,:,1] *= (h-1)

    # reshape to create a list of pixel coordinates
    flow_scaled = np.reshape(flow_scaled, (flow.shape[0]*flow.shape[1], 2)).T

    # swap x, y components of flow to match row, column
    flow_swapped = np.zeros_like(flow_scaled)
    flow_swapped[0,:] = flow_scaled[1,:]
    flow_swapped[1,:] = flow_scaled[0,:]

    # mapped RGB color components
    color_R = np.zeros((flow.shape[0]*flow.shape[1]), dtype=np.uint8)
    color_G = np.zeros_like(color_R)
    color_B = np.zeros_like(color_R)

    # interpolate flow coordinates into RGB textures
    interp.map_coordinates(_colorWheel_R, flow_swapped, color_R, order=0, mode='nearest', cval=0)
    interp.map_coordinates(_colorWheel_G, flow_swapped, color_G, order=0, mode='nearest', cval=0)
    interp.map_coordinates(_colorWheel_B, flow_swapped, color_B, order=0, mode='nearest', cval=0)

    # creates output image
    flowColor = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flowColor[:,:,0] = color_R.reshape(flow.shape[0:2])
    flowColor[:,:,1] = color_G.reshape(flow.shape[0:2])
    flowColor[:,:,2] = color_B.reshape(flow.shape[0:2])

    return flowColor




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
    # also convert original image to compatible format
    # transform range from [0,255] to [-0.5,0.5]
    orig_img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    orig_img2_arr = np.array(img2).astype(np.float32)/255 -0.5

    if data_format == 'channels_first':
        orig_img1_arr = orig_img1_arr.transpose([2,0,1])
        orig_img2_arr = orig_img2_arr.transpose([2,0,1])
        orig_image_pair = np.concatenate((orig_img1_arr,orig_img2_arr), axis=0)
    else:
        orig_image_pair = np.concatenate((orig_img1_arr,orig_img2_arr),axis=-1)

    orig_result = {
        'image_pair': orig_image_pair[np.newaxis,:],
        'image1': orig_img1_arr[np.newaxis,:], # first image
        'image2': orig_img2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }

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

    return result, orig_result


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

input_data, orig_data = prepare_input_data(img1,img2,data_format)

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
print("result['predict_flow2'].squeeze().shape = ", result['predict_flow2'].squeeze().shape)
print("result['predict_flow5'].squeeze().shape = ", result['predict_flow5'].squeeze().shape)
opticalflowData = result['predict_flow2'].squeeze()
print(opticalflowData)
print(np.max(opticalflowData))
#opticalflowImg = Image.fromarray(opticalflowData.astype(np.uint8))
opticalflowImg = flowToColor(opticalflowData, maxflow=1.0)
plt.imshow(opticalflowImg) #predict_flow5, predict_flow2
plt.show()

result = refine_net.eval(input_data['image1'],result['predict_depth2'])

print(type(result))
# for k, v in result.items():
#     print(k, v)
print(result.keys())
plt.imshow(result['predict_depth0'].squeeze(), cmap='Greys')
plt.show()
plt.imshow((1/result['predict_depth0']).squeeze(), cmap='Greys')
plt.show()
# plt.imshow(result['predict_depth_upsampled'].squeeze(), cmap='Greys')
# plt.show()
# plt.imshow(result['predict_depth2'].squeeze(), cmap='Greys')
# plt.show()

# try to visualize the point cloud
print("result['predict_depth0'] = ", result['predict_depth0'])
print("np.max(result['predict_depth0']) = ", np.max(result['predict_depth0']))
print("1/np.max(result['predict_depth0']) = ", 1/np.max(result['predict_depth0']))
print("np.min(result['predict_depth0']) = ", np.min(result['predict_depth0']))
print("1/np.min(result['predict_depth0']) = ", 1/np.min(result['predict_depth0']))
image_scale = np.array(img1).shape[0]/192
print("image_scale = ", image_scale)



try:
    from depthmotionnet.vis import *
    visualize_prediction(
        inverse_depth=result['predict_depth0'],
        # inverse_depth=(1/result['predict_depth0']),
        # intrinsics = np.array([2457.60/3072, 2457.60/2304, 0.5, 0.5]),#################################
        image=input_data['image_pair'][0,0:3] if data_format=='channels_first' else input_data['image_pair'].transpose([0,3,1,2])[0,0:3],
        rotation=rotation,
        translation=translation)
except ImportError as err:
    print("Cannot visualize as pointcloud.", err)

# if __name__ == "__main__":
#     main()
