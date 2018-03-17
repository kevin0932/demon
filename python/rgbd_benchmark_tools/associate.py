#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements:
# sudo apt-get install python-argparse

"""
The Kinect provides the color and depth images in an un-synchronized way. This means that the set of time stamps from the color images do not intersect with those of the depth images. Therefore, we need some way of associating color images to depth images.

For this purpose, you can use the ''associate.py'' script. It reads the time stamps from the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""

import argparse
import sys
import os
import numpy


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches

if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them
    ''')
    parser.add_argument('first_file', help='first text file (e.g. rgb list, format: timestamp data)')
    parser.add_argument('second_file', help='second text file (e.g. ground truth list, format: timestamp data)')
    parser.add_argument('third_file', help='third text file (e.g. depth list, format: timestamp data)')
    parser.add_argument('output_path', help='output file path (format: associated pairs from first and second files: timestamp data timestamp data)')
    parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)
    third_list = read_file_list(args.third_file)

    # matches = associate(first_list, second_list,float(args.offset),float(args.max_difference))
    # matches_filtered = map(list, zip(*matches))
    # matches_filtered_list = [(float(l[0]),l[1:]) for l in matches if len(l)>1]
    # matches_filtered_dict = dict(matches_filtered_list)
    # matches_2ndPair = associate(matches_filtered_dict, third_list,float(args.offset),float(args.max_difference))
    # matches_filtered = map(list, zip(*matches_2ndPair))
    # matches_filtered_list = [(float(l[0]),l[1:]) for l in matches if len(l)>1]
    # matches_filtered_dict = dict(matches_filtered_list)
    # matches_1stPair = associate(matches_filtered_dict, second_list,float(args.offset),float(args.max_difference))

    matches = associate(first_list, second_list,float(args.offset),float(args.max_difference))
    # matches_filtered_list = [(float(l[0]),l[1:]) for l in matches if len(l)>1]
    # matches_filtered_dict = dict(matches_filtered_list)
    valid_keys_by12 = [((l[0])) for l in matches if len(l)>1]
    filtered_first_list = dict(filter(lambda i:i[0] in valid_keys_by12, first_list.iteritems()))
    matches_2ndPair = associate(filtered_first_list, third_list,float(args.offset),float(args.max_difference))
    valid_keys_by123 = [((l[0])) for l in matches_2ndPair if len(l)>1]
    filtered_first_list = dict(filter(lambda i:i[0] in valid_keys_by123, first_list.iteritems()))
    # matches_filtered = map(list, zip(*matches_2ndPair))
    # matches_filtered_list = [(float(l[0]),l[1:]) for l in matches if len(l)>1]
    # matches_filtered_dict = dict(matches_filtered_list)
    matches_1stPair = associate(filtered_first_list, second_list,float(args.offset),float(args.max_difference))
    print(len(matches), ", ", len(valid_keys_by12), ", ", len(valid_keys_by123), ", ", len(filtered_first_list), ", ", len(first_list))
    print(type(matches), ", ", type(valid_keys_by12), ", ", type(valid_keys_by123), ", ", type(filtered_first_list), ", ", type(first_list))


    # print(len(matches), ", ", len(matches_filtered), ", ", len(matches_filtered_dict), ", ", len(matches_2ndPair), ", ", len(matches_1stPair))
    # print(type(first_list), ", ", type(matches), ", ", type(matches_filtered), ", ", type(matches_2ndPair), ", ", type(matches_1stPair))
    # print(first_list[0], ", ", matches[0])

    output_textfile_associated_rgb_and_ground_truth_path = args.output_path
    output_textfile_associated_rgb_and_ground_truth = open(output_textfile_associated_rgb_and_ground_truth_path,'w')

    print("matches_1stPair[0] = ", matches_1stPair[0])
    print("matches_2ndPair[0] = ", matches_2ndPair[0])
    tmp = zip(matches_1stPair, matches_2ndPair)
    print(tmp)
    if args.first_only:
        for a,b in matches:
            print("%f %s"%(a," ".join(first_list[a])))
    else:
        # for a,b in matches:
            # # print("%f %s %f %s"%(a," ".join(first_list[a]),b-float(args.offset)," ".join(second_list[b])))
            # output_textfile_associated_rgb_and_ground_truth.write("%f %s %f %s\n"%(a," ".join(first_list[a]),b-float(args.offset)," ".join(second_list[b])))
        for associated_data in zip(matches_1stPair, matches_2ndPair):
            print(associated_data)
            ab = associated_data[0]
            ac = associated_data[1]
            # print(ab, " ", ac)
            if ab[0]==ac[0]:
                print(ab[0])
                a = ab[0]
                b = ab[1]
                c = ac[1]
                print(a, " ", b, " ", c)
                output_textfile_associated_rgb_and_ground_truth.write("%f %s %f %s %f %s\n"%(a," ".join(first_list[a]),b-float(args.offset)," ".join(second_list[b]), c-float(args.offset)," ".join(third_list[c])))

    output_textfile_associated_rgb_and_ground_truth.close()
