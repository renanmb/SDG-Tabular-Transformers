#! /bin/bash

# Runs the model
# change out TOY_MODEL variables for larger model 

source ./model_config.sh

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers $TOY_MODEL_NUM_LAYERS \
       --hidden-size $TOY_MODEL_HIDDEN_SIZE \
       --num-attention-heads $TOY_MODEL_NUM_HEADS \
       --micro-batch-size $TOY_MODEL_MICRO_BATCH_SIZE \
       --global-batch-size $TOY_MODEL_GLOBAL_BATCH_SIZE \
       --seq-length $TOY_MODEL_SEQ_LEN \
       --max-position-embeddings $TOY_MODEL_MAX_POS_EMD \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --tensorboard-dir $TB_PATH \
       --save $TOY_MODEL_CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --tensor-model-parallel-size $TENSOR_MP_SIZE \
       --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 5000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --load $TOY_MODEL_LOADPATH \
       --vocab-file $VOCAB_FILE \
       --fp16
