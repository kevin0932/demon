#!/bin/bash

for dataset_name
do
  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/VisualInspection_find_all_files_in_current_directory.py --input_depth_dir_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/vizdepthmap --input_optical_flow_dir_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/optical_flow_48_64/ --output_good_pairs_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/
done



