import os
import argparse
import subprocess
import collections
import h5py
import numpy as np
import math
#import os
import sys
import re
import six
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--demon_path", required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    data = h5py.File(args.demon_path)

    valid_pair_num = 0
    output_file_path = os.path.join(args.output_path, 'relative_poses_prediction.txt')

    with open(output_file_path, "w") as fid:
        for image_pair12 in data.keys():
            image_name1, image_name2 = image_pair12.split('---')

            print("converting ", image_pair12)
            valid_pair_num += 1

            rotation_matrix = data[image_pair12]["rotation"]
            print(rotation_matrix.shape)

            translation = data[image_pair12]["translation"]
            print(translation.shape)

            fid.write("%s %s %s " % (image_pair12, image_name1, image_name2))
            fid.write("%s %s %s %s %s %s %s %s %s " % (rotation_matrix[0,0], rotation_matrix[0,1], rotation_matrix[0,2], rotation_matrix[1,0], rotation_matrix[1,1], rotation_matrix[1,2], rotation_matrix[2,0], rotation_matrix[2,1], rotation_matrix[2,2]))
            fid.write("%s %s %s\n" % (translation[0], translation[1], translation[2]))


    ### copy the saved quantization map to another file with valid_pair_num and delete the original one
    print("valid_pair_num = ", valid_pair_num)
    final_output_file_path = os.path.join(args.output_path, 'relative_poses_prediction'+'_validPairNum_'+str(int(valid_pair_num))+'.txt')
    shutil.copy(output_file_path, final_output_file_path)
    os.remove(output_file_path)

if __name__ == "__main__":
    main()
