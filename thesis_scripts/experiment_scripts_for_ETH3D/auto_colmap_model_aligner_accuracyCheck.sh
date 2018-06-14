#!/bin/bash

for dataset_name
do
  #### AdapOF
  opt_image_scale_factor=84
  opt_OF_scale_factor=1
  opt_uncertainty_radius=63

  opt_max_squared_pixel_err=16 
  opt_min_survivor_ratio=4

  tmpChar=_
  format_factor_scaler=100

  dataset_dir=/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name

  matched_database_suffix=AdapOF

  image_pair_retrieval_params=OFscale_${opt_OF_scale_factor}_err_${opt_max_squared_pixel_err}000_survivorRatio_$(($format_factor_scaler*$opt_min_survivor_ratio))

  colmap_export_path=$dataset_dir/DenseSIFT/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius
  aligned_colmap_export_path=$colmap_export_path/0_aligned
  mkdir $aligned_colmap_export_path

  /home/kevin/devel_lib/SfM/colmap/release/src/exe/model_aligner --input_path $colmap_export_path/0 --output_path $aligned_colmap_export_path --ref_images_path $dataset_dir/dslr_calibration_undistorted/geo_registration_positions.txt --robust_alignment_max_error 3
  read -p "Press enter to continue"

  #### FixedOF
  opt_image_scale_factor=84
  opt_OF_scale_factor=1
  opt_uncertainty_radius=84

  matched_database_suffix=FixedOF

  image_pair_retrieval_params=OFscale_${opt_OF_scale_factor}_err_${opt_max_squared_pixel_err}000_survivorRatio_$(($format_factor_scaler*$opt_min_survivor_ratio))

  colmap_export_path=$dataset_dir/DenseSIFT/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius
  aligned_colmap_export_path=$colmap_export_path/0_aligned
  mkdir $aligned_colmap_export_path

  /home/kevin/devel_lib/SfM/colmap/release/src/exe/model_aligner --input_path $colmap_export_path/0 --output_path $aligned_colmap_export_path --ref_images_path $dataset_dir/dslr_calibration_undistorted/geo_registration_positions.txt --robust_alignment_max_error 3
  read -p "Press enter to continue"
 
  #### SRef
  opt_image_scale_factor=0
  opt_OF_scale_factor=1
  opt_uncertainty_radius=0

  matched_database_suffix=SRef

  image_pair_retrieval_params=OFscale_${opt_OF_scale_factor}_err_${opt_max_squared_pixel_err}000_survivorRatio_$(($format_factor_scaler*$opt_min_survivor_ratio))

  colmap_export_path=$dataset_dir/DenseSIFT/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius
  aligned_colmap_export_path=$colmap_export_path/0_aligned
  mkdir $aligned_colmap_export_path

  /home/kevin/devel_lib/SfM/colmap/release/src/exe/model_aligner --input_path $colmap_export_path/0 --output_path $aligned_colmap_export_path --ref_images_path $dataset_dir/dslr_calibration_undistorted/geo_registration_positions.txt --robust_alignment_max_error 3
  read -p "Press enter to continue"

done
