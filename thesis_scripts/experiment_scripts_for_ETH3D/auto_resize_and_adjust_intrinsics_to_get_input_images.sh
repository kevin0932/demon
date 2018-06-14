#!/bin/bash

for dataset_name
do
  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/DeMoN_resize_input_images_intrinsicsAdjusted.py --input_images_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/images/dslr_images_undistorted/ --input_cameras_textfile_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/dslr_calibration_undistorted/cameras.txt  --output_h5_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/DenseSIFT
  #echo "python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/DeMoN_resize_input_images_intrinsicsAdjusted.py --input_images_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/images/dslr_images_undistorted/ --input_cameras_textfile_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/dslr_calibration_undistorted/cameras.txt  --output_h5_dir_path /home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name/DenseSIFT"
done
