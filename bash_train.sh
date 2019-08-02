#!/bin/bash

SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

for hidden_value in 75 50 25
do
	singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 gru_arithmetic.py \
		--num_epochs 500 --hidden_size $hidden_value --use_cuda True\
		|| exit 1
done

# 14255230

# 14257747

# sbatch --time 1000 --gres=gpu:tesla-k80:1 bash_train.sh 
