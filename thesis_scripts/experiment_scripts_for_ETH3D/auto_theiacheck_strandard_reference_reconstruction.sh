#!/bin/bash

for dataset_name
do
  ### set parameters
  opt_camera_init_focal_length=4790.87

  opt_new_optical_flow_guided_matching=0
  opt_optical_flow_guided_matching=0
  opt_ManualCrossCheck=0
  opt_only_image_pairs_as_ref=1
  opt_image_scale_factor=0
  opt_OF_scale_factor=1
  opt_uncertainty_radius=0

  matched_database_suffix=SRef

  opt_max_squared_pixel_err=1
  opt_min_survivor_ratio=4

  tmpChar=_
  format_factor_scaler=100

  ### set paths for automatic experiments
  dataset_dir=/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/$dataset_name

  database_name=database_test

  database_ext=.db
  cereal_ext=.cereal
  txt_ext=.txt

  raw_database_file=$dataset_dir/DenseSIFT/$database_name$database_ext
  image_file_dir=$dataset_dir/DenseSIFT/resized_images_4032_5376
  exhaustive_prediction_dir=$dataset_dir/demon_prediction_exhaustive_pairs


  image_pair_retrieval_params=OFscale_${opt_OF_scale_factor}_err_${opt_max_squared_pixel_err}000_survivorRatio_$(($format_factor_scaler*$opt_min_survivor_ratio))

  colmap_export_path=$dataset_dir/DenseSIFT/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius

  matched_database_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$database_ext

  theia_reconstruction_dir=$dataset_dir/DenseSIFT/TheiaTrial

  base_theia_flag_file=/home/kevin/ThesisDATA/ETH3D/multi_view_training_dslr_undistorted/base_theia_flag_file.txt

  theia_output_reconstructions_dir=$theia_reconstruction_dir/output_reconstructions/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius

  theia_match_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$cereal_ext

  theia_flag_file=$theia_reconstruction_dir/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$txt_ext


  ### further substitute the decomposed relative poses with DeMoN prediction relative poses
  ColmapResult_theiamatch_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius${tmpChar}ColmapRt${cereal_ext}


  ### theia reconstruction and statistics
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/build_reconstruction --flagfile=$theia_flag_file
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/compute_reconstruction_statistics --reconstruction=$theia_output_reconstructions_dir/-0 --logtostderr
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/view_reconstruction --reconstruction=$theia_output_reconstructions_dir/-0
  read -p "Press enter to continue"
  ### theia reconstruction and statistics for ColmapRt
  theia_output_reconstructions_ColmapRt_dir=$theia_reconstruction_dir/output_reconstructions/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_ColmapRt

  theia_flag_ColmapRt_file=$theia_reconstruction_dir/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_ColmapRt$txt_ext

  theia_match_ColmapRt_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_ColmapRt$cereal_ext
  
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/build_reconstruction --flagfile=$theia_flag_ColmapRt_file
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/compute_reconstruction_statistics --reconstruction=$theia_output_reconstructions_ColmapRt_dir/-0 --logtostderr
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/view_reconstruction --reconstruction=$theia_output_reconstructions_ColmapRt_dir/-0
  read -p "Press enter to continue"

  ### theia reconstruction and statistics for DeMoNPredictionRt
  theia_output_reconstructions_DeMoNPredictionRt_dir=$theia_reconstruction_dir/output_reconstructions/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_DeMoNPredictionRt

  theia_flag_DeMoNPredictionRt_file=$theia_reconstruction_dir/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_DeMoNPredictionRt$txt_ext

  theia_match_DeMoNPredictionRt_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_DeMoNPredictionRt$cereal_ext
  
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/build_reconstruction --flagfile=$theia_flag_DeMoNPredictionRt_file
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/compute_reconstruction_statistics --reconstruction=$theia_output_reconstructions_DeMoNPredictionRt_dir/-0 --logtostderr
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/view_reconstruction --reconstruction=$theia_output_reconstructions_DeMoNPredictionRt_dir/-0
  read -p "Press enter to continue"
done
