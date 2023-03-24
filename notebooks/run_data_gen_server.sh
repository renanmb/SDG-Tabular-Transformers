#!/bin/bash
# This example will start serving the model.
source ./model_config.sh
SEED="${1:-42}"
PORT="${2:-5000}"
echo $SEED

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

# CHECKPOINT=$LOADPATH
VOCAB_FILE=credit_card_coder.pickle

python -m torch.distributed.launch $DISTRIBUTED_ARGS tools/run_text_generation_server.py \
       --tensor-model-parallel-size $TENSOR_MP_SIZE  \
       --pipeline-model-parallel-size $PIPELINE_MP_SIZE  \
       --num-layers $NUM_LAYERS  \
       --hidden-size $HIDDEN_SIZE  \
       --load $CHECKPOINT_PATH  \
       --num-attention-heads $NUM_HEADS  \
       --max-position-embeddings $MAX_POS_EMD  \
       --tokenizer-type TabularTokenizer \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length $SEQ_LEN  \
       --out-seq-length $SEQ_LEN  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --top_p 1.0  \
       --seed $SEED \
       --port $PORT
