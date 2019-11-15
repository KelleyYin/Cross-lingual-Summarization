#!/bin/bash

ID=0
Model_Path=/data2/mmyin/teach-stu-summ-v0.1/checkpoints/tsinghua/teacher-student-v1
DATA_PATH=/data2/mmyin/data-raw/summ/tsinghua/ours-method-v1
SRC_DATA=/data2/mmyin/sum_data/rush_data/lcsts/joint-bpe/ours-method-v1
#Ref_data=/data/mmyin/sum_data/rush_data/train/test.tgt
Ref_data=/data2/mmyin/sum_data/rush_data/lcsts/joint-bpe/tsinghua_duc
tools=/data2/mmyin/tools
output=$Model_Path/lcsts.summ

for i in `find $Model_Path/checkpoint_acc*e*.pt`; do
    echo $i
    CUDA_VISIBLE_DEVICES=$ID python interactive.py\
        -s src -t tgt \
        --buffer-size 1024 \
        -data $DATA_PATH --path $i \
        --src_path $SRC_DATA/test.bpe.src \
        --max-len-b 50 \
        --remove-bpe \
        --beam 12 > ${output}.tmp

    grep ^H ${output}.tmp | cut -f3- > ${output}
    sed -i "s/<unk>/UNK/g" ${output}

    python $tools/convert_#_to_num.py --input ${output} --src-file ${Ref_data}/duc2004_en.txt >${output}.tmp
    python $tools/word2numb.py --input ${output}.tmp --word-dict ${Ref_data}/vocab.char --output ${output}.num --char
    files2rouge ${output}.num $Ref_data/test.ref.char.num
done
