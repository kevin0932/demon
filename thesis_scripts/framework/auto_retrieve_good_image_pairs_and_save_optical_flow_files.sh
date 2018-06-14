#!/bin/bash

# thresholds could be tuned!
for dataset_name
do
  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/Scripts_for_low_resolution_opticalflow_depth/SouthBuilding/bidir_matches_for_subpixel_correspondences/Optical_Flow_Prediction_Matching_Guiding_full_quantization_maps_FilteredByBiDirCrossCheckSurvivorRatioandRelativePoseConsistency_WithLowResoFlow_ColmapIdOrderEnforced.py --database_path /home/kevin/ThesisDATA/$dataset_name/DenseSIFT/database.db --output_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs --image_path /home/kevin/ThesisDATA/$dataset_name/DenseSIFT/resized_images_2304_3072/ --demon_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/kevin_southbuilding_demon.h5 --input_good_pairs_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs/good_pairs_from_visual_inspection.txt --max_pixel_err 1 --OF_scale_factor 1 --survivor_ratio 0.6 --rotation_consistency_error_deg 360.0 --translation_consistency_error_deg 360.0

done
