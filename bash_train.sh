#!/bin/bash

SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

for hidden_value in 100
do
	singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 gru_arithmetic.py \
		--num_epochs 500 --hidden_size $hidden_value --training_set datasets/arithmetic/fixed_L4_1e3/training.txt --testing_set datasets/arithmetic/fixed_L4_1e3/testing.txt --model_prefix arithmetic_L4_1e3_fixed\
		|| exit 1
done

# 14292209

# sbatch --time 1000 --gres=gpu:tesla-k80:1 bash_train.sh
