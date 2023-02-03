#!/bin/bash

#$ID_PATH = baseline-resnet18dilated-c1_deepsup-ngpus3-batchSize6-imgMaxSize1000-paddingConst8-segmDownsampleRate8-LR_encoder0.02-LR_decoder0.02-epoch20

# Inference
python -u eval_multipro.py \
  --gpus 0,1 \
  --id video-PSP101 \
  --suffix _epoch_40.pth \
  --low_encoder low_resnet101dilated \
  --high_encoder high_resnet101dilated \
  --arch_decoder nonkeyc1 \
  --fc_dim 2048 \
  --visualize
