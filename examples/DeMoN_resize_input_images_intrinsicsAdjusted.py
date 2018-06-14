import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import argparse
import subprocess
import collections
import sqlite3
import h5py
import six
import cv2

print(sys.path)
from depthmotionnet.vis import *
from depthmotionnet.networks_original import *
from collections import namedtuple
from depthmotionnet.dataset_tools.helpers import *
# from depthmotionnet.dataset_tools.view_tools import *
# from depthmotionnet.helpers import angleaxis_to_rotation_matrix
###### think about how to add lmbspecialops and depthmotionnet into anaconda env path!
# sys.path.append(r'/home/kevin/anaconda_tensorflow_demon_ws/demon/lmbspecialops/python/')
# sys.path.append(r'/home/kevin/anaconda_tensorflow_demon_ws/demon/python/depthmotionnet/')

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

### adapted from Ben's code
def adjust_intrinsics_crop_image(image, K_old, K_new, width_new, height_new):
    from PIL import Image
    from skimage.transform import resize
    #from .helpers import safe_crop_image, safe_crop_array2d

    #original parameters
    fx = K_old[0,0]    # 2457.60
    fy = K_old[1,1]    # 2457.60
    cx = K_old[0,2]    # 1536
    cy = K_old[1,2]    # 1152
    width = image.width    # 3072
    height = image.height  # 2304
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
    img_resize = image.resize((width_resize, height_resize), Image.BILINEAR if scale_x > 1 else Image.LANCZOS)
    # img_resize.show()

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
        img_new = safe_crop_image(img_resize,(x0,y0,x1,y1),(127,127,127))

    else:
        img_new = img_resize.crop((x0,y0,x1,y1))
        print("cropping is within the new image size")
        # img_new.show()

    # print("adjust_intrinsics function return view successfully!")
    return img_new


def warp_flow(img, flow):
    flow = np.transpose(flow, [1,2,0])
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def convert_flow_to_plot_img(flow):
    flow = np.transpose(flow, [1,2,0])
    # h, w = flow.shape[:2]
    # # flow = flow*256
    # # flow = flow*128+128
    # # flow = flow*128+128
    # flow[:,:,0] = flow[:,:,0]*w
    # flow[:,:,1] = flow[:,:,1]*h
    # zeros = np.zeros([h,w])
    # # res = np.concatenate((flow,zeros), axis=2)
    # res = np.dstack((flow,zeros))
    # # res = flow
    # # res[:,:,2] = 0
    # print(res.shape)
    # return res
    h, w = flow.shape[:2]
    # fx, fy = flow[:,:,0]*w, flow[:,:,1]*h
    fx, fy = flow[:,:,0]*255, flow[:,:,1]*255
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def convert_flow_to_save_img(flow):
    flow = np.transpose(flow, [1,2,0])
    h, w = flow.shape[:2]
    # flow = flow*256
    flow[:,:,0] = flow[:,:,0]*w
    flow[:,:,1] = flow[:,:,1]*h
    print(np.max(flow), np.min(flow))
    # flow = flow*128+128
    zeros = np.zeros([h,w])
    # res = np.concatenate((flow,zeros), axis=2)
    res = np.dstack((flow,zeros))
    # res = flow
    # res[:,:,2] = 0
    print(res.shape)
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images_dir_path", required=True)
    parser.add_argument("--input_cameras_textfile_path", required=True)
    parser.add_argument("--output_h5_dir_path", required=True)
    args = parser.parse_args()
    return args


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

        R = np.empty((3,3), dtype=np.float32)
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
Camera = namedtuple('Camera',['model','width','height','params'])

def read_colmap_cameras_txt(filename):
    result = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                pass
            else:
                items = line.split(' ')
                params = [ float(x) for x in items[4:] ]
                camera = Camera(
                    model = items[1],
                    width = int(items[2]),
                    height = int(items[3]),
                    params = params,
                )
                result[int(items[0])] = camera
    return result




def main():
    args = parse_args()

    image_exts = [ '.jpg', '.JPG', '.jpeg', '.png' ]
    input_dir = args.input_images_dir_path
    output_dir = args.output_h5_dir_path

    if args.input_cameras_textfile_path == 'manual':
        # K_old = np.eye(3)
        # K_old[0,0] = 2737.64
        # K_old[1,1] = 2737.64
        # K_old[0,2] = 1536
        # K_old[1,2] = 1152
        # print("K_old = ", K_old)
        K_old = np.eye(3)
        K_old[0,0] = 2704.457143
        K_old[1,1] = 2704.457143
        K_old[0,2] = 1632
        K_old[1,2] = 1224
        print("K_old = ", K_old)
    else:
        colmapCameras = read_colmap_cameras_txt(args.input_cameras_textfile_path)

        # my own datasets
        print("colmapCameras[1] = ", colmapCameras[1])
        K_old = np.eye(3)
        K_old[0,0] = colmapCameras[1].params[0]
        K_old[1,1] = colmapCameras[1].params[1]
        K_old[0,2] = colmapCameras[1].params[2]
        K_old[1,2] = colmapCameras[1].params[3]
        print("K_old = ", K_old)

        # ## ETH3D Datsets
        # print("colmapCameras[0] = ", colmapCameras[0])
        # K_old = np.eye(3)
        # K_old[0,0] = colmapCameras[0].params[0]
        # K_old[1,1] = colmapCameras[0].params[1]
        # K_old[0,2] = colmapCameras[0].params[2]
        # K_old[1,2] = colmapCameras[0].params[3]
        # print("K_old = ", K_old)

    # my own datasets
    K_new = np.eye(3)
    K_new[0,0] = 2737.64 # 0.89115971*3072 = 2737.64263
    K_new[1,1] = 2737.64 # 1.18821287*2304 = 2737.64245
    K_new[0,2] = 1536
    K_new[1,2] = 1152
    height_new = 2304
    width_new = 3072

    # # my own datasets for ab test
    # K_new = np.eye(3)
    # K_new[0,0] = 2737.64 # 0.89115971*3072 = 2737.64263
    # K_new[1,1] = 2737.64 # 1.18821287*2304 = 2737.64245
    # K_new[0,2] = 320
    # K_new[1,2] = 240
    # height_new = 480
    # width_new = 640

    # ### ETH3D Datasets
    # K_new = np.eye(3)
    # K_new[0,0] = 4790.87 # 0.89115971*5376 = 4790.8746
    # K_new[1,1] = 4790.87 # 1.18821287*4032 = 4790.87429
    # K_new[0,2] = 2688
    # K_new[1,2] = 2016
    # height_new = 4032
    # width_new = 5376

    # Create output directory, if not present
    try:
        os.stat(output_dir)
    except:
        os.mkdir(output_dir)
    try:
        os.stat(os.path.join(output_dir, "resized_images_%s_%s" % (height_new, width_new)))
    except:
        os.mkdir(os.path.join(output_dir, "resized_images_%s_%s" % (height_new, width_new)))

    iteration = 0
    # Iterate over working directory
    for file1 in os.listdir(input_dir):
        file1_path = os.path.join(input_dir, file1)
        file1_name, file1_ext = os.path.splitext(file1_path)

        if file1_ext not in image_exts:
            print("Skipping " + file1 + " (not an image file)")
            continue
        print("Processing image = " + file1)

        img1 = Image.open(file1_path)

        # # input_data = prepare_input_data(img1,img2,data_format)
        # if img1.size[0] != 3072 or img1.size[1] != 2304:
        #     resized_img1 = img1.resize((3072,2304))
        resized_img1 = adjust_intrinsics_crop_image(img1, K_old, K_new, width_new, height_new)

        resized_img1.save(os.path.join(output_dir, "resized_images_%s_%s" % (height_new, width_new), file1))
        # plt.imsave(os.path.join(output_dir, "resized_images_%s_%s" % (height_new, width_new), os.path.splitext(file1)[0]), resized_img1, cmap=cmap)

        # # # a colormap and a normalization instance
        # cmap = plt.cm.jet
        # plt.imsave(os.path.join(output_dir, "graydepthmap", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), result['predict_depth0'].squeeze(), cmap='Greys')
        # plt.imsave(os.path.join(output_dir, "vizdepthmap", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), result['predict_depth0'].squeeze(), cmap=cmap)
        #
        # ofplot = convert_flow_to_plot_img(flow2)
        # plt.imsave(os.path.join(output_dir, "optical_flow_48_64", os.path.splitext(file1)[0] + "---" + os.path.splitext(file2)[0]), ofplot, cmap=cmap)
        # # ofplot = convert_flow_to_plot_img(flow2)
        # # cv2.imshow('ofplot', ofplot)
        # # cv2.waitKey()
        # # cv2.destroyAllWindows()
        # # return
if __name__ == "__main__":
    main()
