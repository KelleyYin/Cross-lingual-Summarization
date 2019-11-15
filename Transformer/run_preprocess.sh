#!/bin/bash

DATA_PATH=/data2/mmyin/wmt16_en_de
DATA_RAW=/data2/mmyin/data-raw/wmt16_en-de

python preprocess.py -s en -t de \
        --trainpref $DATA_PATH/train.tok.clean.bpe.32000\
        --validpref $DATA_PATH/newstest2013.tok.bpe.32000\
        --testpref $DATA_PATH/newstest2014.tok.bpe.32000\
        --destdir $DATA_RAW\
        --output-format binary\
        --joined-dictionary\
