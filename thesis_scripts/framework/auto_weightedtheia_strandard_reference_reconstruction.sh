#!/bin/bash

for dataset_name
do
  ### set parameters
  opt_camera_init_focal_length=2737.64

  opt_new_optical_flow_guided_matching=0
  opt_optical_flow_guided_matching=0
  opt_ManualCrossCheck=0
  opt_only_image_pairs_as_ref=1
  opt_image_scale_factor=0
  opt_OF_scale_factor=1
  opt_uncertainty_radius=0

  matched_database_suffix=SRef

  opt_max_squared_pixel_err=16
  opt_min_survivor_ratio=4

  tmpChar=_
  format_factor_scaler=100

  ### set paths for automatic experiments
  dataset_dir=/home/kevin/ThesisDATA/$dataset_name

  database_name=database_test

  database_ext=.db
  cereal_ext=.cereal
  txt_ext=.txt

  raw_database_file=$dataset_dir/DenseSIFT/$database_name$database_ext
  image_file_dir=$dataset_dir/DenseSIFT/resized_images_2304_3072
  exhaustive_prediction_dir=$dataset_dir/demon_prediction_exhaustive_pairs


  image_pair_retrieval_params=OFscale_${opt_OF_scale_factor}_err_${opt_max_squared_pixel_err}000_survivorRatio_$(($format_factor_scaler*$opt_min_survivor_ratio))

  colmap_export_path=$dataset_dir/DenseSIFT/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius

  matched_database_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$database_ext

  theia_reconstruction_dir=$dataset_dir/DenseSIFT/TheiaTrial
  mkdir $theia_reconstruction_dir

  base_theia_flag_file=/home/kevin/ThesisDATA/base_theia_flag_file.txt
  mkdir $theia_reconstruction_dir/featureWS
  mkdir $theia_reconstruction_dir/output_reconstructions

  theia_output_reconstructions_dir=$theia_reconstruction_dir/output_reconstructions/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius
  mkdir $theia_output_reconstructions_dir

  theia_match_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$cereal_ext

  theia_flag_file=$theia_reconstruction_dir/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius$txt_ext

  cp $base_theia_flag_file $theia_flag_file
  sed -i "s|--matches_file=|--matches_file=$theia_match_file|" $theia_flag_file
  sed -i "s|--output_reconstruction=|--output_reconstruction=$theia_output_reconstructions_dir/|" $theia_flag_file
  sed -i "s|--matching_working_directory=|--matching_working_directory=$theia_reconstruction_dir/featureWS/|" $theia_flag_file

  ### convert colmap matched database file to theia match file
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/read_standard_colmap_DB_content_withOptimizedCameras --camera_focal_length $opt_camera_init_focal_length --colmap_camera_txt $colmap_export_path/0/cameras.txt --database_filepath $matched_database_file

  ### further substitute the decomposed relative poses with the colmap results/ground truth
  ### /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/read_colmap_relativeRt_to_theia_output_matchfile --matchfile=$theia_match_file --colmap_global_poses_images_textfile=$dataset_dir/dslr_calibration_undistorted/images_onlyImageNames.txt
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/read_colmap_relativeRt_to_theia_output_matchfile --matchfile=$theia_match_file --colmap_global_poses_images_textfile=$colmap_export_path/0/images.txt

  ### further substitute the decomposed relative poses with DeMoN prediction relative poses
  ColmapResult_theiamatch_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar$opt_uncertainty_radius${tmpChar}ColmapRt${cereal_ext}

  find $exhaustive_prediction_dir -name "relative_poses_prediction_validPairNum*.txt" -print0 | while read -d $'\0' demon_relative_poses_txtfile
  do
    #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/read_DeMoN_prediction_hdf5_to_theia_output_matchfile --matchfile=$theia_match_file --demon_prediction_relative_poses_textfile=$demon_relative_poses_txtfile

    survivor_score_textfile=$exhaustive_prediction_dir/survivor_scores.txt
    /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/read_DeMoN_prediction_hdf5_to_theia_output_matchfile_WeightedMotionAveraging --matchfile=$theia_match_file --demon_prediction_relative_poses_textfile=$demon_relative_poses_txtfile --survivor_score_textfile=$survivor_score_textfile

    ### compare the DeMoN prediction relative poses with Colmap Result/Ground Truth
    #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/read_DeMoN_prediction_hdf5_to_theia_output_matchfile --matchfile=$ColmapResult_theiamatch_file --demon_prediction_relative_poses_textfile=$demon_relative_poses_txtfile
  done

  ### theia reconstruction and statistics
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/build_reconstruction --flagfile=$theia_flag_file
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/compute_reconstruction_statistics --reconstruction=$theia_output_reconstructions_dir/-0 --logtostderr
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/view_reconstruction --reconstruction=$theia_output_reconstructions_dir/-0

  ### theia reconstruction and statistics for ColmapRt
  theia_output_reconstructions_ColmapRt_dir=$theia_reconstruction_dir/output_reconstructions/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_ColmapRt
  mkdir $theia_output_reconstructions_ColmapRt_dir

  theia_match_ColmapRt_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_ColmapRt$cereal_ext
  cp $base_theia_flag_file $theia_flag_ColmapRt_file
  sed -i "s|--matches_file=|--matches_file=$theia_match_ColmapRt_file|" $theia_flag_ColmapRt_file
  sed -i "s|--output_reconstruction=|--output_reconstruction=$theia_output_reconstructions_ColmapRt_dir/|" $theia_flag_ColmapRt_file
  sed -i "s|--matching_working_directory=|--matching_working_directory=$theia_reconstruction_dir/featureWS/|" $theia_flag_ColmapRt_file

  theia_flag_ColmapRt_file=$theia_reconstruction_dir/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_ColmapRt$txt_ext
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/build_reconstruction --flagfile=$theia_flag_ColmapRt_file
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/compute_reconstruction_statistics --reconstruction=$theia_output_reconstructions_ColmapRt_dir/-0 --logtostderr
  #/home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/view_reconstruction --reconstruction=$theia_output_reconstructions_ColmapRt_dir/-0

  
  ### theia reconstruction and statistics for DeMoNPredictionRt
  theia_output_reconstructions_DeMoNPredictionRt_dir=$theia_reconstruction_dir/output_reconstructions/$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_DeMoNPredictionRt
  mkdir $theia_output_reconstructions_DeMoNPredictionRt_dir
  theia_flag_DeMoNPredictionRt_file=$theia_reconstruction_dir/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_DeMoNPredictionRt$txt_ext

  theia_match_DeMoNPredictionRt_file=$dataset_dir/DenseSIFT/$database_name$tmpChar$matched_database_suffix$tmpChar$image_pair_retrieval_params$tmpChar$opt_image_scale_factor$tmpChar$opt_OF_scale_factor$tmpChar${opt_uncertainty_radius}_DeMoNPredictionRt$cereal_ext
  cp $base_theia_flag_file $theia_flag_DeMoNPredictionRt_file
  sed -i "s|--matches_file=|--matches_file=$theia_match_DeMoNPredictionRt_file|" $theia_flag_DeMoNPredictionRt_file
  sed -i "s|--output_reconstruction=|--output_reconstruction=$theia_output_reconstructions_DeMoNPredictionRt_dir/|" $theia_flag_DeMoNPredictionRt_file
  sed -i "s|--matching_working_directory=|--matching_working_directory=$theia_reconstruction_dir/featureWS/|" $theia_flag_DeMoNPredictionRt_file

  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/build_reconstruction --flagfile=$theia_flag_DeMoNPredictionRt_file
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/compute_reconstruction_statistics --reconstruction=$theia_output_reconstructions_DeMoNPredictionRt_dir/-0 --logtostderr
  /home/kevin/devel_lib/SfM/TheiaSfM/theia-release/bin/view_reconstruction --reconstruction=$theia_output_reconstructions_DeMoNPredictionRt_dir/-0

  
done
