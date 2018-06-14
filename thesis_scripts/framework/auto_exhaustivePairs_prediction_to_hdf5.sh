#!/bin/bash

for dataset_name
do
  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/DeMoN_prediction_to_h5.py --input_images_dir_path /home/kevin/ThesisDATA/$dataset_name/DenseSIFT/resized_images_2304_3072/ --output_h5_dir_path /home/kevin/ThesisDATA/$dataset_name/demon_prediction_exhaustive_pairs

done
