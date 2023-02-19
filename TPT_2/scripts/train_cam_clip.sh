#!/bin/bash

data_root='/data/tangbowen/datasets'
datasets=CUB
# arch=RN10
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./train_cam_classification.py ${data_root} --data_sets ${datasets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init}