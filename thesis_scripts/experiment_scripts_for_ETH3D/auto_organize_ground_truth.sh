#!/bin/bash

for dataset_name
do
  ### set paths for automatic experiments
  dataset_dir=/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name

  calibration_dir=$dataset_dir/dslr_calibration_undistorted
  copy_dir=$dataset_dir/DenseSIFT/ground_truth_sparse/0
  mkdir $copy_dir
  alignment_dir=$dataset_dir/DenseSIFT/ground_truth_sparse/0_aligned
  mkdir $alignment_dir
  geo_registration_positions_file_outdir=$calibration_dir

  original_ETH3D_ground_truth_images_txt_file=images.txt
  original_ETH3D_ground_truth_cameras_txt_file=cameras.txt
  original_ETH3D_ground_truth_points3D_txt_file=points3D.txt
  nameChanged_ETH3D_ground_truth_images_txt_file=images_onlyImageNames.txt

  geo_registration_positions_filename=geo_registration_positions.txt


  copy_nameChanged_ETH3D_ground_truth_images_txt_file=images.txt
  copy_nameChanged_ETH3D_ground_truth_cameras_txt_file=cameras.txt
  copy_nameChanged_ETH3D_ground_truth_points3D_txt_file=points3D.txt

  cp $calibration_dir/$original_ETH3D_ground_truth_images_txt_file $calibration_dir/$nameChanged_ETH3D_ground_truth_images_txt_file
  sed -i "s|dslr_images_undistorted/DSC*|DSC|" $calibration_dir/$nameChanged_ETH3D_ground_truth_images_txt_file

  python /home/kevin/devel_lib/SfM/colmap/scripts/python/import_model_and_save_the_geo_registration_file_ETH3D.py $calibration_dir .txt $geo_registration_positions_file_outdir

  cp $calibration_dir/$nameChanged_ETH3D_ground_truth_images_txt_file $copy_dir/$copy_nameChanged_ETH3D_ground_truth_images_txt_file
  cp $calibration_dir/$original_ETH3D_ground_truth_cameras_txt_file $copy_dir/$copy_nameChanged_ETH3D_ground_truth_cameras_txt_file
  cp $calibration_dir/$original_ETH3D_ground_truth_points3D_txt_file $copy_dir/$copy_nameChanged_ETH3D_ground_truth_points3D_txt_file
  cp $calibration_dir/$geo_registration_positions_filename $copy_dir/$geo_registration_positions_filename

done
