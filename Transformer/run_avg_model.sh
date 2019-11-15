#!/bin/bash

model=/data2/mmyin/teach-stu-summ-v0.1/checkpoints/teach-stu-no-nll-v3
out=$model/avg_model_last_20.pt
num_epoch=20
upper_bound=50

python /data2/mmyin/tf-baseline/scripts/average_checkpoints.py --inputs $model \
                    --output $out\
                    --num-epoch-checkpoints $num_epoch\
                    --checkpoint-upper-bound $upper_bound
