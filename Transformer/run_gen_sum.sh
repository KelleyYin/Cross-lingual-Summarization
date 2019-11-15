#!/bin/bash
ID=7
model_file=./checkpoints/tsinghua/lcsts-32k-joint
model=`find $model_file/checkpoint_*e64.pt`
echo $model
DATA_RAW=/data2/mmyin/data-raw/summ/tsinghua/lcsts-32k-joint
#src_path=/data2/mmyin/sum_data/rush_data/mt_ldc/32k-bpe-joint/duc2004.bpe.ch
src_path=/data2/mmyin/sum_data/rush_data/lcsts/joint-bpe/lcsts-32k-bpe/test.bpe.src
pred=$model_file/lcsts.summ

CUDA_VISIBLE_DEVICES=$ID python interactive.py \
        -s src -t tgt \
        --buffer-size 1024 \
        -data $DATA_RAW --path $model \
        --max-len-b 50 \
        --min-len 1 \
        --remove-bpe \
        --src_path $src_path \
        --nbest 1 \
        --beam 12 > gen.en_

grep ^H gen.en_ | cut -f3- > $pred
sed -i "s/<unk>/UNK/g" $pred

python /data2/mmyin/tools/word2numb.py --input $pred --output $pred.num --word-dict /data2/mmyin/sum_data/rush_data/lcsts/vocab.char --char
files2rouge $pred.num /data2/mmyin/sum_data/rush_data/lcsts/joint-bpe/lcsts-32k-bpe/test.#.tgt.num
echo "Translation Have Done !"

