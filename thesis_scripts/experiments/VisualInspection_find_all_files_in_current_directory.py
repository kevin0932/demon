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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_depth_dir_path", required=True)
    parser.add_argument("--input_optical_flow_dir_path", required=True)
    parser.add_argument("--output_good_pairs_path", required=True)
    parser.add_argument("--ext", default='JPG')
    # parser.add_argument("--image_scale", type=float, default=12)
    args = parser.parse_args()

    # input_dir = os.path.join(os.getcwd(), 'img')
    # output_dir = os.path.join(os.getcwd(), 'out')
    return args

def main():

    args = parse_args()

    image_exts = [ '.jpg', '.JPG', '.jpeg', '.png', '' ]
    input_depth_dir = args.input_depth_dir_path
    input_optical_flow_dir = args.input_optical_flow_dir_path
    # output_dir = args.output_h5_dir_path

    good_pairs_depth = []

    iteration = 0
    # Iterate over working directory
    for file1 in os.listdir(input_depth_dir):
        file1_path = os.path.join(input_depth_dir, file1)
        file1_name, file1_ext = os.path.splitext(file1_path)
        # # # Check if file is an image file
        # if file1_ext not in image_exts:
        #     print("Skipping " + file1 + " (not an image file)")
        #     continue

        good_pairs_depth.append(file1)
        # img1 = Image.open(file1_path)
        iteration += 1
    print("total image number in filtered depth dir = ", iteration)

    good_pairs_flow = []

    iteration = 0
    # Iterate over working directory
    for file1 in os.listdir(input_optical_flow_dir):
        file1_path = os.path.join(input_depth_dir, file1)
        file1_name, file1_ext = os.path.splitext(file1_path)
        # # # Check if file is an image file
        # if file1_ext not in image_exts:
        #     print("Skipping " + file1 + " (not an image file)")
        #     continue

        good_pairs_flow.append(file1)
        # img1 = Image.open(file1_path)
        iteration += 1
    print("total image number in filtered flow dir = ", iteration)

    good_pairs_filtered_by_depth_and_flow = list(set(good_pairs_depth) & set(good_pairs_flow))
    print(good_pairs_filtered_by_depth_and_flow)
    print(len(good_pairs_filtered_by_depth_and_flow))

    with open(os.path.join(args.output_good_pairs_path, 'good_pairs_from_visual_inspection.txt'), "w") as fid:
        for i in range(len(good_pairs_filtered_by_depth_and_flow)):
            tmp = good_pairs_filtered_by_depth_and_flow[i].split('---')
            # if args.ext=='JPG':
            #     image_pair12 = tmp[0]+'.JPG---'+tmp[1]+'.JPG'
            # if args.ext=='png':
            # # image_pair12 = tmp[0]+'.png---'+tmp[1]+'.png'
            image_pair12 = tmp[0]+'.'+args.ext+'---'+tmp[1]+'.'+args.ext
            fid.write("%s\n" % (image_pair12))
    fid.close()

if __name__ == "__main__":
    main()
