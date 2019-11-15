#!/bin/bash

model=/data2/mmyin/tf-baseline/checkpoints/baseline2/32k-bpe-joint
out=aaaa.pt
num_epoch=10
upper_bound=40

python /data2/mmyin/tf-baseline/scripts/average_checkpoints.py --inputs $model  --output $out --num-epoch-checkpoints $num_epoch --checkpoint-upper-bound $upper_bound
