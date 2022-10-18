#!/bin/bash
export TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

SRC=src # file suffix of the source file
TGT=tgt # file suffix of the target file
DATA=/path/to/data/ # should contain train.$SRC and train.$TGT
PRETRAINED_MODEL=/path/to/model # the transformers model dir
CONFIG=/path/to/model/config.json # the transformers model config
TOKENIZER=/path/to/tokenizer # the transformers model or tokenizer dir
SAVE=/path/to/save/dir # your saving dir

mkdir -p $SAVE
cp $0 $SAVE/

# The hyper-parameters for 4-GPU training
# For other #GPUs, use --max-tokens and --iter-per-update to adjust batch size

python train.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s $SRC $TGT \
  --max-tokens 4096 \
  --num-training-steps 100000 \
  -lr 7e-4 \
  --num-warmup-steps 4000 \
  --iter-per-update 2 \
  --save-dir $SAVE \
  --update-per-save 1000 \
  -mn $PRETRAINED_MODEL \
  --fp32 \
  --label-smoothing 0.1 \
  | tee -a $SAVE/train.log
