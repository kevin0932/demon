# import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
# import h5py
examples_dir = os.path.dirname(__file__)

depth_img1 = Image.open(os.path.join(examples_dir,'0000231-000007685751.png'))
# depth_img2 = Image.open(os.path.join(examples_dir,'0000251-000008353136.png'))
depth_img1.show()

depth_img1 = np.array(depth_img1)
print(depth_img1.shape)

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
                n = n/np.linalg.norm(n)
            normal_map[y,x,:] = n

    # print(normal_map)
    # np.savetxt('test.txt', normal_map[:,:,0])
    print(normal_map.shape)
    # plt.imshow((normal_map-np.min(normal_map))/(np.max(normal_map)-np.min(normal_map)))
    plt.imshow(normal_map/2+0.5)
    print(np.min(normal_map), " ", np.max(normal_map))
    plt.show()

    return normal_map

compute_normals_from_depth(depth_img1)
