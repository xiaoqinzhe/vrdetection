set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=$1
NUM_IM=$3
NET=$2
INFERENCE_ITER=$4
WEIGHT_FN=$4
TEST_MODE=$5
GPU_ID=$3

DATASET=$1
NUM_IM=$6
INFERENCE_ITER=2
WEIGHT_FN=checkpoints/$DATASET/${WEIGHT_FN}

CFG_FILE=experiments/cfgs/sparse_graph.yml




# LOG="$OUTPUT/logs/`date +'%Y-%m-%d_%H-%M-%S'`"

export CUDA_VISIBLE_DEVICES=$GPU_ID

time ./tools/test_net.py --gpu ${GPU_ID} \
  --weights ${WEIGHT_FN} \
  --cfg ${CFG_FILE} \
  --dataset ${DATASET} \
  --network ${NET} \
  --test_size ${NUM_IM} \
  --test_mode ${TEST_MODE} 
