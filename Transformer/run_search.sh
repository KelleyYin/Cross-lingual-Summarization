#!/bin/bash

ID=6
Model_Path=/data2/mmyin/relay-attn-v8/checkpoints/guide-last-layer-relay-attn-5-words-sqrt-0.3
DATA_PATH=/data2/mmyin/data-raw/summ/baseline2/relay-attn-v2
SRC_DATA=/data2/mmyin/sum_data/rush_data/mt_ldc/32k-bpe-joint
ref=/data2/mmyin/sum_data/rush_data/train/valid.2000.tgt
output=$Model_Path/rush.valid.summ

for i in `find $Model_Path/checkpoint_acc*e*.pt`; do
    echo $i
    CUDA_VISIBLE_DEVICES=$ID python interactive.py\
        -s src -t tgt \
        --buffer-size 2048 \
        -data $DATA_PATH --path $i \
        --src_path $SRC_DATA/rush.valid.bpe.ch \
        --max-len-b 50 \
        --remove-bpe \
        --beam 12 > ${output}_

    grep ^H ${output}_ | cut -f3- > $output
    sed -i "s/<unk>/UNK/g" $output
    #python /data2/mmyin/tools/word2numb.py --input $output --output ${output}.num --word-dict /data2/mmyin/sum_data/rush_data/lcsts/vocab.char --char

    files2rouge $output $ref
done
