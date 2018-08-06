# Pre-processing
For all the experiments, DeMoN predictions should be made and saved locally so that predictions can be retrieved later whenever you want according to different settings. Thus, again, we need to run scripts to resize and adjust the intrinsics of input images. The results will be saved in .h5 files.

* DeMoN_resize_input_images_intrinsicsAdjusted.py
* DeMoN_prediction_to_h5_filter_pairs.py


# Fuse depth map predictions by DeMoN: add the point cloud one by one
* depth_experiment_v3_PointCloudEvolution.py
* the script is run with South-Building datasets and the output is just the point cloud .PLY file and the VTK visualisation
* as described in the thesis, the depth maps can be scaled by either the ground truth translation lengths or the ground truth depth maps (the COLMAP results are used here as accurate depth maps)
* depth_experiment_v1_southbuilding.py - a short-cut script to fuse all the views without step visualisation

# Fuse depth map predictions by DeMoN and filter the point cloud by checking multi-view geometric/photometric consistency
* depth_experiment_v6_PointCloudNoiseRemoval.py
* the script is run with South-Building datasets and the output is just the point cloud .PLY file and the VTK visualisation
* as described in the thesis, the depth maps can be scaled by either the ground truth translation lengths or the ground truth depth maps (the COLMAP results are used here as accurate depth maps)

# Filter_image_pairs_in_HDF5_file_by_some_conditions.py
A test script to filter image pairs according to different conditions (view angle differences, view overlap ratios, camera positions, "left-right" consistency scores)

# colmap_util.py
Utility functions to interact with COLMAP 2.0, from Benjamin. it should be placed in the same directory as the experiment scripts for portability.
