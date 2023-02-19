#!/bin/bash

data_root='/data/tangbowen/datasets'
testsets=CUB
# arch=RN10
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./tpt_cam_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init}