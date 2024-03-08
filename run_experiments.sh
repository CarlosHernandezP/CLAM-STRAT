#!/bin/bash

# Define the common part of the command
common_command="python main.py --drop_out --early_stopping --lr 1e-4 --k 10 --weighted_sample --task tcga_cvpr --reg 0.001 --data_root_dir tcga_level_1_cvpr"

# Array of different parts of the commands
declare -a model_types=("transformer --use_fga" "transformer" "max" "mean")

# Loop through each combination of data_root_dir and model_type
for model_type in "${model_types[@]}"; do
	# Execute the command
	$common_command --model_type $model_type

	# Wait for the command to finish before starting the next one
	wait
done

