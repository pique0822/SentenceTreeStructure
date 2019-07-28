#!/bin/bash/

singularity run -B /om2 -B /home/drmiguel/SentenceTreeStructure /om2/user/drmiguel/singularity_images/deepo-cpu.simg train_gru.sh
