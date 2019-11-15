#!/bin/bash
DATA_RAW=/data3/mmyin/data-raw/wmt16_en-de
#word_emb=/data2/mmyin/sum_data/rush_data/independent-bpe-file/32k-bpe-inde.vec

CUDA_VISIBLE_DEVICES=4,5,6 python train.py -data $DATA_RAW \
        -s en -t de \
        --lr 0.0005 --min-lr 1e-09 \
        --weight-decay 0 --clip-norm 0.0 \
        --dropout 0.3 \
        --max-tokens 4500 \
        --arch transformer \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
        --warmup-updates 4000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --save-dir checkpoints/wmt16_en-de\
        --share-all-embeddings \
