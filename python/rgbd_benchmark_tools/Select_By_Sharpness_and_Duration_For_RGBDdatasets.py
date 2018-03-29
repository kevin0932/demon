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
import numpy as np
from PIL import Image
from scipy.ndimage.filters import laplace


def measure_sharpness(img):
    """Measures the sharpeness of an image using the variance of the laplacian

    img: PIL.Image

    Returns the variance of the laplacian. Higher values mean a sharper image
    """
    img_gray = np.array(img.convert('L'), dtype=np.float32)
    return np.var(laplace(img_gray))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_dir_path", required=True)
    parser.add_argument("--input_associated_pair_file_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    # parser.add_argument("--image_scale", type=float, default=12)
    args = parser.parse_args()

    # input_dir = os.path.join(os.getcwd(), 'img')
    # output_dir = os.path.join(os.getcwd(), 'out')
    return args

def read_associated_pair_file_dict(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def read_associated_pair_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    # list = [((l[0]),l[1:]) for l in list if len(l)>1]
    list = [(l[0:]) for l in list if len(l)>1]
    return (list)

def main():

    args = parse_args()
    image_exts = [ '.jpg', '.JPG', '.jpeg', '.png' ]
    input_dir = args.input_dataset_dir_path
    output_dir = args.output_dir_path

    associated_pair_list = read_associated_pair_file_list(args.input_associated_pair_file_path)
    print("len(associated_pair_list) = ", len(associated_pair_list))
    # # # for item in associated_pair_list.items():
    # # for ky in associated_pair_list.keys():
    # #     print(ky, " => ", associated_pair_list[ky])
    # for item in range(len(associated_pair_list)):
    #     print(item, " => ", associated_pair_list[item])
    # # for file1 in os.listdir(input_dir):

    windownSize = 10
    idSeq = np.arange(0,len(associated_pair_list),windownSize)
    # print(idSeq)
    numSaveImages = 0
    for cnt in range(len(idSeq)-1):
        # print(idSeq[cnt]," ", idSeq[cnt+1])
        SharpestImageId = -1
        BestSharpest = -1
        windownIdSeq = np.arange(idSeq[cnt],idSeq[cnt+1],1)
        # print(windownIdSeq)
        for i in windownIdSeq:
            img = Image.open(os.path.join(input_dir, associated_pair_list[i][1]))
            # img.show()
            # return
            # # img1 = Image.open(file1_path)
            tmpSharpness = measure_sharpness(img)
            # print(i, " ", associated_pair_list[i], " ", tmpSharpness)
            if tmpSharpness>=BestSharpest:
                BestSharpest = tmpSharpness
                SharpestImageId = i
        if SharpestImageId > -1:
            img4Save = Image.open(os.path.join(input_dir, associated_pair_list[SharpestImageId][1]))
            img4Save.save(os.path.join(output_dir, associated_pair_list[SharpestImageId][1]),"PNG")
            numSaveImages += 1

    print("Done! numSaveImages = ", numSaveImages)

if __name__ == "__main__":
    main()
