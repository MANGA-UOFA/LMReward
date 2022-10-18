#!/bin/bash
export TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

SRC=src # file suffix of the source file
DATA=/path/to/data/ # should contain train.$SRC and train.$TGT
CONFIG=/path/to/model/config.json # the transformers model config
TOKENIZER=/path/to/tokenizer # the transformers model or tokenizer dir
REWARD_MODEL=/path/to/reward_model # the transformers reward model dir
INIT_MODEL=/path/to/init_model # the transformers model dir
SAVE=/path/to/save/dir # your saving dir

mkdir -p $SAVE
cp $0 $SAVE/

# The hyper-parameters for 4-GPU training
# For other #GPUs, use --max-tokens and --iter-per-update to adjust batch size

python reinforce.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s $SRC $SRC \
  --max-tokens 2048 \
  --num-training-steps 100000 \
  -lr 1e-5 \
  --scheduler constant \
  --num-warmup-steps 4000 \
  --iter-per-update 4 \
  --save-dir $SAVE \
  --update-per-save 1000 \
  -mn $INIT_MODEL \
  --reward-model $REWARD_MODEL \
  --fp32 \
  --max-norm 1 \
  --entropy 0.1 \
  --topk 2 \
  --reward-clip 1 \
  --baseline \
  --softmax \
  --denom 100 \
  --update-per-sync 5000 \
  --length-discard \
  | tee -a $SAVE/train.log
