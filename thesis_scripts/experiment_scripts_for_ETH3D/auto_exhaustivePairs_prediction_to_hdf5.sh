#!/bin/bash

for dataset_name
do
  #python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/DeMoN_prediction_to_h5.py --input_images_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/DenseSIFT/resized_images_4032_5376/ --output_h5_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/demon_prediction_exhaustive_pairs
  echo "python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/DeMoN_prediction_to_h5.py --input_images_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/DenseSIFT/resized_images_4032_5376/ --output_h5_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/demon_prediction_exhaustive_pairs"
done
