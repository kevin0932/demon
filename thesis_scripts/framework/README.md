To run the pipeline or experiments, please make sure that all the scripts are run under the same environment that satisfies all the dependencies!

Our experiments are done in the following environments:
* Ubuntu 16.04
* python 3 in anaconda with vtk, demon installed
* Dependencies of DeMoN, Theia and Colmap should be satisfied!

Feature extraction and matching are run in Colmap. They can be done in either GUI or Command-Line interface. Currently-called executables are located in the build folder (/home/kevin/devel_lib/SfM/colmap/release/src/exe/feature_extractor) and feel free to change the path accordingly or just call the one in the system if you install it into your system folder.

$dataset_name is the folder name that stores the input datasets under the specific directory.

Please set your directory path accordingly.

please organize your input dataset like "Sample_Dataset_Folder" (in the same directory)
* Sample_Dataset_Folder
*   --demon_prediction_exhaustive_pairs	->	DeMoN predictions will be stored here
*   --dense_results_by_COLMAP		->	dense reconstruction by COLMAP if you need it (should be run manually if you want to use it as reference)
*   --DenseSIFT				->	our results will be stored accordingly in this folder
*   --original_images			->	original image files in either .JPG or .png
*   --sparse				->	COLMAP sparse results from original images (it can be used as accurate reference if COLMAP makes reasonable reconstructions)

Please make the .sh scripts and this script executable (chmod +x scriptfile) if they are not executable.

Generally, please consider about and check the following parameters existing in the following scripts if the pipeline is broken or should be modified:
* feature_matcher setups: standard, fixed_radius, adaptive_radius
* thresholds: consistency_filtering_ratio (e.g. 0.6), flow_consistency_error (squared_error, e.g. 1), pre-defined searching radius (e.g. 4), relative rotation/translation consistency errors in degrees (e.g. 360)
* executable paths, directory paths
* initial intrinsics for intrinsic adjustment
* paths or linking of dependencies

Please modify theia global SfM setup in the file "base_theia_flag_file.txt"
* modify calibration parameters as the file "calibrationfile.txt" if you want to manually provide intrinsics for Theia pipelines
