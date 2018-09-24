set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=$1
NUM_IM=$3
NET=$3
INFERENCE_ITER=$4
WEIGHT_FN=$5
TEST_MODE=$6
GPU_ID=$1

DATASET=vg
NUM_IM=-1
NET=dual_graph_vrd_final
INFERENCE_ITER=2
WEIGHT_FN=checkpoints/chk/
TEST_MODE=all

CFG_FILE=experiments/cfgs/sparse_graph.yml


case $DATASET in
    vg)
        ROIDB=VG-SGG
        RPNDB=proposals
        IMDB=imdb_1024
        ;;
    mini-vg)
        ROIDB=mini_VG-SGG
        RPNDB=mini_proposals
        IMDB=mini_imdb_1024
        ;;
    *)
        echo "Wrong dataset"
        exit
        ;;
esac

LOG="$OUTPUT/logs/`date +'%Y-%m-%d_%H-%M-%S'`"

export CUDA_VISIBLE_DEVICES=$GPU_ID

time ./tools/test_net.py --gpu ${GPU_ID} \
  --weights ${WEIGHT_FN} \
  --imdb ${IMDB}.h5 \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB}.h5 \
  --cfg ${CFG_FILE} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --test_size ${NUM_IM} \
  --test_mode ${TEST_MODE} 
