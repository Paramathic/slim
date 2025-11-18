export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"

export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"

# HF_TOKEN="--hf_token HUGGINGFACE_ACCESS_TOKEN"
# HF_TOKEN=""

export WANDB_MODE="offline"

export TRITON_CACHE_DIR="/tmp"


NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


for MODEL_NAME in qwen2.5 # llama3.2 #opt #llama2 #llama3.1
do
    if [ $MODEL_NAME == 'llama2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-2-
        MODEL_POSTFIX=-hf
        MODEL_SIZE_LIST="7b"
    elif [ $MODEL_NAME == 'opt' ]
    then   
        MODEL_PREFIX=facebook/opt-
        MODEL_POSTFIX=""
        MODEL_SIZE_LIST="125m" #30b"
    elif [ $MODEL_NAME == 'llama3.2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.2-
        MODEL_SIZE_LIST="1B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'llama3.1' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.1-
        MODEL_SIZE_LIST="8B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'llama3' ]
    then
        MODEL_PREFIX=meta-llama/Meta-Llama-3-
        MODEL_SIZE_LIST="8B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'qwen2.5' ]
    then
        MODEL_PREFIX="Qwen/Qwen2.5-"
        MODEL_SIZE_LIST="0.5B"
        MODEL_POSTFIX=""
    else
        echo "Unknown model name: $MODEL_NAME"
        exit 1
    fi

    for MODEL_SIZE in $MODEL_SIZE_LIST
    do
        for STRUCTURE in 2:4
        do
            for METHOD in wanda #sparsegpt #maskllm sparsegpt joint_pq
            do
                for LORA_RANK in 0 #0.1
                do
                    for SLIM_LORA in '--slim_lora' #''
                    do
                        for NUM_CALIBRATION_SAMPLES in 128
                        do
                            for QUANTIZE_WEIGHT in '' #'--quantize_weight' # ''
                            do
                                for TILED_WEIGHT_QUANTIZATION in '--tiled_weight_quantization'
                                do
                                    for LEARNING_RATE in 1e-5 #5e-5 1e-4 5e-4
                                    do
                                        LOCAL_FILES_ONLY='--local_files_only'
                                        SPARSITY_RATIO=0.5
                                        SHIFT_ZERO_METRICS='--shift_zero_metrics'
                                        EVAL_DATASET='wikitext2'
                                        BITWIDTH=4
                                        INPUT_GROUP_SIZE=128
                                        # SLIM_QUANT='--slim_quant'
                                        EVAL_BATCH_SIZE=1
                                        SEPARATE_LORA='--separate_lora'
                                        TEST_LMHARNESS='--test_lmharness'
                                        LM_HARNESS_TASKS="--lm_harness_tasks mmlu piqa arc_easy arc_challenge winogrande openbookqa race hellaswag"
                                        # FINE_TUNE='--fine_tune'
                                        EVALUATE_PERPLEXITY='--evaluate_perplexity'
                                        OPTIMIZER="adamw_torch"
        #                                PRUNE_LORA="--prune_lora"
                                        QUANTIZE_LORA="--quantize_lora"
                                        LORA_TILE_SIZE=128
                                        WEIGHT_TILE_SIZE=128
                                        JOINT_PQ_MIXING_FACTOR=2.1
                                        CALIBRATION_DATASET="c4"
                                        # QUANTIZE_INPUT="--quantize_input"
                                        INPUT_BITWIDTH=8
                                        INPUT_GROUP_SIZE=-1
                                        PAD_LORA='--pad_lora'
    #                                    SCALE_IMPORTANT_WEIGHTS='--scale_important_weights'
                                        MASKLLM_CHECKPOINT="--maskllm_checkpoint tiled_models/llama_3.2_1b_maskllm.pt"
                                        WANDB="--use_wandb"
                                        SAVE_CHECKPOINT_PATH="--save_checkpoint_path checkpoints/${MODEL_NAME}_${MODEL_SIZE}_${METHOD}_${STRUCTURE}_lr${LORA_RANK}_sparsity${SPARSITY_RATIO}"
                                        if [ $FINE_TUNE == '--fine_tune' ]
                                        then
                                            SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH}_ft_lr${LEARNING_RATE}"
                                        fi
                                        COLUMN_WISE_GROUPING="--column_wise_grouping"
                                        PARALLELISM="model_parallel"
                                        FINETUNE_TOKEN_COUNT=560000
                                        WEIGHT_DECAY=1e-2
                                        FINE_TUNING_GLOBAL_BATCH_SIZE=256
                                        FINE_TUNING_SEQLEN=4096

                                        if [ "$PARALLELISM" = "data_parallel" ]; then
                                            echo "Using data parallelism with $NUM_GPUS GPUs"
                                            STARTER_CMD="torchrun --nproc_per_node=$NUM_GPUS --rdzv_endpoint=localhost:29500"
                                        elif [ "$PARALLELISM" = "model_parallel" ]; then
                                            echo "Using model parallelism"
                                            STARTER_CMD="accelerate launch --num_processes=1 --mixed_precision=bf16"
                                        else
                                            echo "Unknown parallelism type: $PARALLELISM. Defaulting to data parallelism."
                                            STARTER_CMD="torchrun --nproc_per_node=$NUM_GPUS --rdzv_endpoint=localhost:29500"
                                        fi

                                        $STARTER_CMD main.py \
                                            --model ${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX} \
                                            --prune_method $METHOD \
                                            --sparsity_ratio $SPARSITY_RATIO \
                                            --sparsity_type $STRUCTURE \
                                            --lora_rank $LORA_RANK \
                                            $SLIM_LORA \
                                            --eval_dataset $EVAL_DATASET \
                                            $SHIFT_ZERO_METRICS \
                                            $QUANTIZE_WEIGHT \
                                            --bitwidth $BITWIDTH \
                                            $SLIM_QUANT \
                                            --eval_batch_size $EVAL_BATCH_SIZE \
                                            $SEPARATE_LORA \
                                            $TEST_LMHARNESS \
                                            $LM_HARNESS_TASKS \
                                            --output_csv_path results/tmp.csv \
                                            $FINE_TUNE \
                                            $EVALUATE_PERPLEXITY \
                                            $LOCAL_FILES_ONLY \
                                            $QUANTIZE_INPUT \
                                            --input_bitwidth $INPUT_BITWIDTH \
                                            --input_group_size $INPUT_GROUP_SIZE \
                                            --nsample $NUM_CALIBRATION_SAMPLES \
                                            --optimizer $OPTIMIZER \
                                            $TILED_INPUT_QUANTIZATION \
                                            $PRUNE_LORA \
                                            $QUANTIZE_LORA \
                                            --lora_tile_size $LORA_TILE_SIZE \
                                            $TILED_WEIGHT_QUANTIZATION \
                                            --weight_tile_size $WEIGHT_TILE_SIZE \
                                            $HF_TOKEN \
                                            --joint_pq_mixing_factor $JOINT_PQ_MIXING_FACTOR \
                                            --calibration_dataset $CALIBRATION_DATASET \
                                            $PAD_LORA \
                                            $SCALE_IMPORTANT_WEIGHTS \
                                            $MASKLLM_CHECKPOINT \
                                            $WANDB \
                                            $SAVE_CHECKPOINT_PATH \
                                            $COLUMN_WISE_GROUPING \
                                            --learning_rate $LEARNING_RATE \
                                            --finetune_token_count $FINETUNE_TOKEN_COUNT \
                                            --weight_decay $WEIGHT_DECAY \
                                            --fine_tuning_global_batch_size $FINE_TUNING_GLOBAL_BATCH_SIZE \
                                            --fine_tuning_seqlen $FINE_TUNING_SEQLEN
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
