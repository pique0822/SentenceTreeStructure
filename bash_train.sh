#!/bin/bash

SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

for hidden_value in 100 75 50 25
do
	singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 gru_arithmetic.py \
		--num_epochs 100 --hidden_size $hidden_value\
		|| exit 1
done
