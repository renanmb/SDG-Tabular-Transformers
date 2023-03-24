############################################################################
##
## Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
############################################################################

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# INPUTS
INPUT_FILE=credit_card_pd.jl
VOCAB_FILE=credit_card_coder.pickle
TOKENIZER=TabularTokenizer
PREPROCESS_WORKERS=$(nproc)

OUTPUT_PATH=checkpoints
PROJECT_NAME=creditcard
CHECKPOINT_PATH=${OUTPUT_PATH}/gpt_${PROJECT_NAME}
LOADPATH=${OUTPUT_PATH}/gpt_${PROJECT_NAME}
TB_PATH=${OUTPUT_PATH}/checkpoints/tb
DATA_PATH=${PROJECT_NAME}_text_document

TENSOR_MP_SIZE=1
PIPELINE_MP_SIZE=1

#model architecture for larger model
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_HEADS=16
SEQ_LEN=6144
MAX_POS_EMD=6144

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$GPUS_PER_NODE))

# inputs and path parameters for toy model used in workshop
TOY_MODEL_PROJECT_NAME=toy_model
TOY_MODEL_CHECKPOINT_PATH=${OUTPUT_PATH}/gpt_${TOY_MODEL_PROJECT_NAME}
TOY_MODEL_LOADPATH=${OUTPUT_PATH}/gpt_${TOY_MODEL_PROJECT_NAME}
TOY_MODEL_TB_PATH=${OUTPUT_PATH}/checkpoints/tb/${TOY_MODEL_PROJECT_NAME}

TOY_MODEL_NUM_LAYERS=6
TOY_MODEL_HIDDEN_SIZE=1024
TOY_MODEL_NUM_HEADS=16
TOY_MODEL_SEQ_LEN=768
TOY_MODEL_MAX_POS_EMD=768

TOY_MODEL_MICRO_BATCH_SIZE=4
TOY_MODEL_GLOBAL_BATCH_SIZE=$(($TOY_MODEL_MICRO_BATCH_SIZE*$GPUS_PER_NODE))

# hidden size % attention heads == 0