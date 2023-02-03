#!/bin/bash



python -u train_KDM.py \
  --gpus 0,1 \
  --low_encoder low_resnet101dilated \
  --high_encoder high_resnet101dilated \
  --arch_decoder nonkeyc1 \
  --fc_dim 2048 \
  --num_epoch 20 \
  --epoch_iters 6000 \
  --batch_size_per_gpu 2 \
  --lr_keyselect 0.01 \
  --start_epoch 1


