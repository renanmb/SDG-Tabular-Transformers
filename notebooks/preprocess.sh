#!/bin/bash
source ./model_config.sh

python tools/preprocess_data.py --input=$INPUT_FILE \
                                --output-prefix=$PROJECT_NAME \
                                --vocab=$VOCAB_FILE \
                                --dataset-impl=mmap \
                                --tokenizer-type=$TOKENIZER \
                                --workers=$PREPROCESS_WORKERS
