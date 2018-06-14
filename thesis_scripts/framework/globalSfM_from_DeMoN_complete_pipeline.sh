#!/bin/bash

# To run the pipeline or experiments, please make sure that all the scripts are run under the same environment that satisfies all the dependencies!
# Our experiments are done in the following environments:
#	Ubuntu 16.04
#	python 3 in anaconda with vtk, demon installed
#	Dependencies of DeMoN, Theia and Colmap should be satisfied!

# feature extraction and matching are run in Colmap. They can be done in either GUI or Command-Line interface. Currently-called executables are located in the build folder (/home/kevin/devel_lib/SfM/colmap/release/src/exe/feature_extractor) and feel free to change the path accordingly or just call the one in the system if you install it into your system folder.

# $dataset_name is the folder name that stores the input datasets under the specific directory
# you may change the directory path accordingly

# please organize your input dataset like "Sample_Dataset_Folder" (in the same directory)
#	--Sample_Dataset_Folder
#		--demon_prediction_exhaustive_pairs	->	DeMoN predictions will be stored here
#		--dense_results_by_COLMAP		->	dense reconstruction by COLMAP if you need it
#		--DenseSIFT				->	our results will be stored accordingly in this folder
#		--original_images			->	original image files in either .JPG or .png
#		--sparse				->	COLMAP sparse results from original images (it can be used as accurate reference if COLMAP makes reasonable reconstructions)

for dataset_name
do
  # preprocess the input images
  auto_resize_and_adjust_intrinsics_to_get_input_images.sh $dataset_name

  # DeMoN inference exhaustively on all possible image pairs and save results into .hdf5 file
  auto_exhaustivePairs_prediction_to_hdf5.sh $dataset_name

  # save relative pose predictions to text file
  auto_convert_hdf5_predictions_to_text.sh $dataset_name

  # (mandatory but useless in the final pipeline) manually select the image pairs that will be considered:just for the compatibility with old scripts 
  auto_visualinspection.sh $dataset_name

  # retrieve "good" image pairs that DeMoN will mostly give reliable predictions [thresholds should be tuned!]
  auto_retrieve_good_image_pairs_and_save_optical_flow_files.sh $dataset_name

  # {feature extraction & matching in colmap} tune parameters
  auto_colmap_standard_reference_reconstruction.sh $dataset_name

  # {feature extraction & fixed-radius guided matching in colmap} tune parameters
  auto_colmap_fixedR_optical_flow_guided_reconstruction.sh $dataset_name

  # {feature extraction & adaptive-radius (by flow_conf_score) guided matching in colmap} tune parameters
  auto_colmap_adapR_optical_flow_guided_reconstruction.sh $dataset_name

  # {convert Colmap database into Theia matchfile and run global SfM in theia: standard matching with standard/accurate/DeMoN relative poses} tune parameters or change setup in the flagfile under the same directory (base_theia_flag_file.txt)
  auto_theia_strandard_reference_reconstruction.sh $dataset_name

  # {convert Colmap database into Theia matchfile and run global SfM in theia: fixed-radius guided matching with standard/accurate/DeMoN relative poses} tune parameters or change setup in the flagfile under the same directory (base_theia_flag_file.txt)
  auto_theia_fixedR_optical_flow_guided_reconstruction.sh $dataset_name

  # {convert Colmap database into Theia matchfile and run global SfM in theia: fixed-radius guided matching with standard/accurate/DeMoN relative poses} tune parameters or change setup in the flagfile under the same directory (base_theia_flag_file.txt)
  auto_theia_adapR_optical_flow_guided_reconstruction.sh $dataset_name

  # (trial with weighted motion averaging) you need to change the setup (rotation and position solvers) in base_theia_flag_file.txt
  #auto_weightedtheia_strandard_reference_reconstruction.sh $dataset_name

  # check the models produced by Colmap incremental SfM by model aligner
  #auto_colmap_model_aligner_accuracyCheck.sh $dataset_name

  # check the input relative pose quality (compared to either accurate results from COLMAP or ground truth poses if available)
  #auto_poseErrCheck_theia_strandard_reference_reconstruction.sh $dataset_name

  # check the reconstruction results of standard matching pipelines
  #auto_theiacheck_strandard_reference_reconstruction.sh $dataset_name

  # check the reconstruction results of adaptive-radius guided matching pipelines
  #auto_theiacheck_adapR_optical_flow_guided_reconstruction.sh $dataset_name

done



