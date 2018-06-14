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
    parser.add_argument("--input_image_dir_path", required=True)
    # parser.add_argument("--input_image_dir_path", required=True)
    # parser.add_argument("--input_optical_flow_dir_path", required=True)
    parser.add_argument("--output_image_dir_path", required=True)
    # parser.add_argument("--image_scale", type=float, default=12)
    args = parser.parse_args()

    # input_dir = os.path.join(os.getcwd(), 'img')
    # output_dir = os.path.join(os.getcwd(), 'out')
    return args

def main():

    args = parse_args()

    image_exts = [ '.jpg', '.JPG', '.jpeg', '.png', '' ]
    input_image_dir = args.input_image_dir_path
    output_dir = args.output_image_dir_path

    target_images = []

    iteration = 0
    # Iterate over working directory
    for file1 in os.listdir(input_image_dir):
        file1_path = os.path.join(input_image_dir, file1)
        file1_name, file1_ext = os.path.splitext(file1_path)
        # # Check if file is an image file
        if file1_ext not in image_exts:
            print("Skipping " + file1 + " (not an image file)")
            continue

        if file1_name[-4:]=='cam0':
            target_images.append(file1)
            output_image_path = os.path.join(output_dir, file1)
            img1 = Image.open(file1_path)
            img1.save(output_image_path)
            iteration += 1
    print("total target image number retrieved in current dir = ", iteration)



if __name__ == "__main__":
    main()
