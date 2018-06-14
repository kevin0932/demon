#!/bin/bash

for dataset_name
do

  opt_new_optical_flow_guided_matching=0
  opt_optical_flow_guided_matching=1
  opt_ManualCrossCheck=1
  opt_only_image_pairs_as_ref=0
  opt_image_scale_factor=48
  opt_OF_scale_factor=1
  opt_uncertainty_radius=32

  opt_max_squared_pixel_err=1 
  opt_min_survivor_ratio=5

  opt_max_rotation_consistency=360
  opt_max_translation_consistency=360

  tmpChar=_
  format_factor_scaler=100

  dataset_dir=/home/kevin/ThesisDATA/$dataset_name

  database_name=database_test
  matched_database_suffix=AdapOF
  database_ext=.db

  raw_database_file=$dataset_dir/DenseSIFT/$database_name$database_ext
  image_file_dir=$dataset_dir/DenseSIFT/resized_images_2304_3072
  exhaustive_prediction_dir=$dataset_dir/demon_prediction_exhaustive_pairs

  image_pair_retrieval_params=OFscale_${opt_OF_scale_factor}_err_${opt_max_squared_pixel_err}000_survivorRatio_$(($format_factor_scaler*$opt_min_survivor_ratio))

  colmap_export_path=$dataset_dir/DenseSIFT/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius
  mkdir $colmap_export_path

  matched_database_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$database_ext

  #match_list_file=$(find $exhaustive_prediction_dir -name "BothSideSurvivor_OrderEnforced_filter_360_360_full_quantization_map_$image_pair_retrieval_params*.txt" -print0 | while read -d $'\0' )
  #optical_flow_file=$(find $exhaustive_prediction_dir -name "BothSideSurvivor_OrderEnforced_OpticalFlow_360_360_full_quantization_map_$image_pair_retrieval_params*.txt" -print0 | while read -d $'\0')


  find $exhaustive_prediction_dir -name "BothSideSurvivor_OrderEnforced_filter_${opt_max_rotation_consistency}_${opt_max_translation_consistency}_full_quantization_map_$image_pair_retrieval_params*.txt" -print0 | while read -d $'\0' match_list_file
  do
    echo $match_list_file

    find $exhaustive_prediction_dir -name "BothSideSurvivor_OrderEnforced_OpticalFlow_${opt_max_rotation_consistency}_${opt_max_translation_consistency}_full_quantization_map_$image_pair_retrieval_params*.txt" -print0 | while read -d $'\0' optical_flow_file
    do
      echo $optical_flow_file

      ###### Feature Extractor, O.F.Guided Matching or Standard Matching
      /home/kevin/devel_lib/SfM/colmap/release/src/exe/feature_extractor --database_path $raw_database_file --image_path $image_file_dir
      cp $raw_database_file $matched_database_file

      /home/kevin/devel_lib/SfM/colmap/release/src/exe/optical_flow_guided_feature_matcher --database_path $matched_database_file --optical_flow_path $optical_flow_file --match_list_path $match_list_file --image_scale_factor $opt_image_scale_factor --OF_scale_factor $opt_OF_scale_factor --uncertainty_radius $opt_uncertainty_radius --new_optical_flow_guided_matching $opt_new_optical_flow_guided_matching --optical_flow_guided_matching $opt_optical_flow_guided_matching --ManualCrossCheck $opt_ManualCrossCheck --only_image_pairs_as_ref $opt_only_image_pairs_as_ref

      /home/kevin/devel_lib/SfM/colmap/release/src/exe/mapper --database_path $matched_database_file --image_path $image_file_dir --export_path $colmap_export_path

    done

  done

done
