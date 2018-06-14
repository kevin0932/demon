#!/bin/bash

# save relative pose predictions to text file
for dataset_name
do
  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/Scripts_for_low_resolution_opticalflow_depth/SouthBuilding/bidir_matches_for_subpixel_correspondences/Convert_hdf5_to_relative_prediction_text_file.py --output_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/ --demon_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5
  #echo "python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/Scripts_for_low_resolution_opticalflow_depth/SouthBuilding/bidir_matches_for_subpixel_correspondences/Convert_hdf5_to_relative_prediction_text_file.py --output_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/ --demon_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5"
done
