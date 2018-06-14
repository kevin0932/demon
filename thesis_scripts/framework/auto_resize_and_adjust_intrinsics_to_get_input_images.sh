#!/bin/bash

for dataset_name
do
  python /home/kevin/anaconda_tensorflow_demon_ws/demon/examples/DeMoN_resize_input_images_intrinsicsAdjusted.py --input_images_dir_path /home/kevin/ThesisDATA/$dataset_name/original_images --input_cameras_textfile_path manual  --output_h5_dir_path /home/kevin/ThesisDATA/$dataset_name/DenseSIFT
done
