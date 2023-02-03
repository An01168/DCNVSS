#!/bin/bash



python -u train.py \
  --gpus 0,1,2,3 \
  --low_encoder low_resnet101dilated \
  --high_encoder high_resnet101dilated \
  --arch_decoder nonkeyc1 \
  --fc_dim 2048 \
  --num_epoch 40 \
  --epoch_iters 6000 \
  --batch_size_per_gpu 2 \
  --lr_match 0.1 \
  --lr_nonkeydecoder 0.1 \
  --start_epoch 1


