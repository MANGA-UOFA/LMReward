#!/bin/bash

MODEL_DIR=/path/to/checkpoints
SPLIT=valid_or_test # specify valid or test
SRC=src # the file suffix for source
TGT=tgt # the file suffix for target
DATASET=/path/to/dataset # should contain $SPLIT.$SRC and $SPLIT.$TGT

for ckpt in `ls $MODEL_DIR`
do
python decoding.py --max-length-a 3 --max-sentences 64 --do-sample --top-k 1 -i $DATASET/$SPLIT.$SRC -mn $MODEL_DIR/$ckpt -o /var/tmp/$SPLIT.gen
python metrics.py --lowercase --gen /var/tmp/$SPLIT.gen --ref $DATASET/$SPLIT.$TGT --src $DATASET/$SPLIT.$SRC
done