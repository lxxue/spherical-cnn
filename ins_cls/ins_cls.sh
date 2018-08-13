#!/bin/bash
num_cls=(626 106 513 171 570 335 64 197 885 167 79 138 200 109 199 145 161 155 145 124 149 282 463 199 88 229 236 103 113 128 677 123 90 390 162 343 266 473 87 103)
for i in `seq 0 39`
do
    echo ${num_cls[$i]}
    TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
                           python3 ../scripts/train.py \
                                   @../params/model-64.txt \
                                   @../params/ins_cls-64.txt \
                                   @../params/training.txt \
                                   --n_classes ${num_cls[$i]} \
                                   --dset_dir ~/data/ins_cls/$i \
                                   --logdir ./log_$i \
                                   --run_id $i
done
