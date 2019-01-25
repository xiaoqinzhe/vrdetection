#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$2
INFERENCE_ITER=2
EXP_DIR=$4
GPU_ID=$3

INFERENCE_ITER=2

CFG_FILE=experiments/cfgs/sparse_graph.yml
PRETRAINED=data/pretrained/coco_vgg16_faster_rcnn_final.npy

# dataset
DATASET=$1
ITERS=$5

# log
OUTPUT=checkpoints/$DATASET/$EXP_DIR
TF_LOG=checkpoints/$DATASET/$EXP_DIR/tf_logs
rm -rf ${OUTPUT}/logs/
rm -rf ${TF_LOG}
rm -rf ${TF_LOG}val
mkdir -p ${OUTPUT}/logs
LOG="$OUTPUT/logs/`date +'%Y-%m-%d_%H-%M-%S'`"

export CUDA_VISIBLE_DEVICES=$GPU_ID

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --gpu 0 \
  --weights ${PRETRAINED} \
  --dataset ${DATASET} \
  --iters ${ITERS} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --output ${OUTPUT} \
  --tf_log ${TF_LOG}
