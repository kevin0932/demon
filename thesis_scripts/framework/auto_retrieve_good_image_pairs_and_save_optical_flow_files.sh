#!/bin/bash

# thresholds could be tuned!
for dataset_name
do
  opt_new_optical_flow_guided_matching=0
  opt_optical_flow_guided_matching=0
  opt_ManualCrossCheck=0
  opt_only_image_pairs_as_ref=1
  opt_image_scale_factor=0
  opt_OF_scale_factor=1
  opt_uncertainty_radius=0

  opt_max_squared_pixel_err=1 
  opt_min_survivor_ratio=5

  opt_max_rotation_consistency=20
  opt_max_translation_consistency=50

  tmpChar=_
  format_factor_scaler=100

  dataset_dir=/home/kevin/ThesisDATA/$dataset_name

  database_name=database
  matched_database_suffix=SRef
  database_ext=.db

  raw_database_file=$dataset_dir/DenseSIFT/$database_name$database_ext
  image_file_dir=$dataset_dir/DenseSIFT/resized_images_2304_3072
  exhaustive_prediction_dir=$dataset_dir/demon_prediction_exhaustive_pairs

  colmap_export_path=$dataset_dir/DenseSIFT/sparse
  mkdir $colmap_export_path

  matched_database_file=$dataset_dir/DenseSIFT/database_matched.db

  ###### Standard Colmap pipeline for reference and retrievel
  /home/kevin/devel_lib/SfM/colmap/release/src/exe/feature_extractor --database_path $raw_database_file --image_path $image_file_dir
  cp $raw_database_file $matched_database_file

  /home/kevin/devel_lib/SfM/colmap/release/src/exe/optical_flow_guided_feature_matcher --database_path $matched_database_file --optical_flow_path $optical_flow_file --match_list_path $match_list_file --image_scale_factor $opt_image_scale_factor --OF_scale_factor $opt_OF_scale_factor --uncertainty_radius $opt_uncertainty_radius --new_optical_flow_guided_matching $opt_new_optical_flow_guided_matching --optical_flow_guided_matching $opt_optical_flow_guided_matching --ManualCrossCheck $opt_ManualCrossCheck --only_image_pairs_as_ref $opt_only_image_pairs_as_ref

  /home/kevin/devel_lib/SfM/colmap/release/src/exe/mapper --database_path $matched_database_file --image_path $image_file_dir --export_path $colmap_export_path
  

  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/Scripts_for_low_resolution_opticalflow_depth/SouthBuilding/bidir_matches_for_subpixel_correspondences/Optical_Flow_Prediction_Matching_Guiding_full_quantization_maps_FilteredByBiDirCrossCheckSurvivorRatioandRelativePoseConsistency_WithLowResoFlow_ColmapIdOrderEnforced.py --database_path $raw_database_file --output_path $exhaustive_prediction_dir --image_path $image_file_dir --demon_path $exhaustive_prediction_dir/kevin_southbuilding_demon.h5 --input_good_pairs_path $exhaustive_prediction_dir/good_pairs_from_visual_inspection.txt --max_pixel_err 1 --OF_scale_factor 1 --survivor_ratio 0.6 --rotation_consistency_error_deg 360.0 --translation_consistency_error_deg 360.0

  #python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/Scripts_for_low_resolution_opticalflow_depth/SouthBuilding/bidir_matches_for_subpixel_correspondences/Optical_Flow_Prediction_Matching_Guiding_full_quantization_maps_FilteredByBiDirCrossCheckSurvivorRatioandRelativePoseConsistency_WithLowResoFlow_ColmapIdOrderEnforced.py --database_path /home/kevin/ThesisDATA/$dataset_name/DenseSIFT/database.db --output_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs --image_path /home/kevin/ThesisDATA/$dataset_name/DenseSIFT/resized_images_2304_3072/ --demon_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5 --input_good_pairs_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/good_pairs_from_visual_inspection.txt --max_pixel_err 1 --OF_scale_factor 1 --survivor_ratio 0.6 --rotation_consistency_error_deg 360.0 --translation_consistency_error_deg 360.0

done
