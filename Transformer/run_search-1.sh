#!/bin/bash

ID=5
Model_Path=./checkpoints/rush-32k-joint-pre
DATA_PATH=/data/mmyin/data-raw/summ/rush-32k-joint
SRC_DATA=/data/mmyin/sum_data/rush_data/train/32k-joint

for i in `find $Model_Path/checkpoint_acc*e*.pt`; do
    echo $i
    CUDA_VISIBLE_DEVICES=$ID python interactive.py\
        -s src -t tgt \
        --buffer-size 1024 \
        -data $DATA_PATH --path $i \
        --src_path $SRC_DATA/test.bpe.src \
        --remove-bpe \
        --max-len-b 50 \
        --beam 12 > tmp.123
    grep ^H tmp.123 | cut -f3- > rush.tgt.123
    sed -i "s/<unk>/UNK/g" rush.tgt.123
    files2rouge rush.tgt.123 $SRC_DATA/test.tgt
done

