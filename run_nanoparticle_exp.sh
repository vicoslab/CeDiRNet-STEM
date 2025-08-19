#!/bin/bash

export NANOPARTICLES_KI_DATASET_DIR="$(dirname $BASH_SOURCE)/dataset"
export OUTPUT_DIR="$(dirname $BASH_SOURCE)/results"

mkdir -p $OUTPUT_DIR

export PYTHONPATH=$PYTHONPATH:$(dirname $BASH_SOURCE)/src

########################################
# GET LOCALIZATION NETWORK

centernet_filename="$OUTPUT_DIR/localization_checkpoint.pth"
wget -O $centernet_filename https://box.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth

########################################
# TRAINING

DATASET="nanoparticles" python -m train --config "pretrained_center_model_path=$centernet_filename" display=True skip_if_exists=True resume=True

########################################
# EVALUATING

EVAL_EPOCHS=("") # ("" _010 _020 _030 _040 _050 _060 _070 _080 _090)

# FOR DISPLAY
#DISPLAY_ARGS="display=True eval.score_combination_and_thr.0.center=[0.40] visualizer.opts.plot_only=[image]"
# FOR FULL EVAL
DISPLAY_ARGS="display=False "

for epoch_eval in "${EVAL_EPOCHS[@]}"; do
    DATASET="nanoparticles" python -m test --config eval_epoch=$epoch_eval skip_if_exists=False $DISPLAY_ARGS
done
