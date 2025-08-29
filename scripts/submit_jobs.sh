#!/bin/bash

# --- Configuration ---
# Define the ranges for your hyperparameters
MODEL_NAMES=("llama2" "opt")
STRUCTURE=("2:4" "unstructured")
SPARSITY_RATIO=(0.5)
METHOD=(wanda)
LORA_RANK=(0.1)
SLIM_LORA=(true)
SEPARATE_LORA=true
QUANTIZE_LORA=(true false)
LORA_TILE_SIZE=128
PAD_LORA=true
CALIBRATION_DATASET="c4"
NUM_CALIBRATION_SAMPLES=128
QUANTIZE_WEIGHT=(true false)
BITWIDTH=4
WEIGHT_TILE_SIZE=128
SLIM_QUANT=(true false)
LOCAL_FILES_ONLY=true
EVAL_DATASET="wikitext2"
EVALUATE_PERPLEXITY=true
TEST_LMHARNESS=true
FINE_TUNE=(false)
OPTIMIZER="adafactor"
SCALE_IMPORTANT_WEIGHTS=false
MASKLLM_CHECKPOINT="MASKLLM_CHECKPOINT_PLACEHOLDER"
QUANTIZE_INPUT=false
INPUT_BITWIDTH=8
INPUT_GROUP_SIZE=-1
JOINT_PQ_MIXING_FACTOR=2.1
WANDB=true
HF_TOKEN="HF_TOKEN_PLACEHOLDER"
OUTPUT_CSV_FILE="results/results.csv"


NGPUS_PER_NODE=1
NTASKS_PER_NODE=$((12 * NGPUS_PER_NODE))
MEM=$((64 * NGPUS_PER_NODE))
GPU_TYPE=""
TIME="4:00:00"


for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    if [ $MODEL_NAME == 'llama2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-2-
        MODEL_POSTFIX=-hf
        MODEL_SIZE_LIST='7b 13b'
    elif [ $MODEL_NAME == 'opt' ]
    then   
        MODEL_PREFIX=facebook/opt-
        MODEL_POSTFIX=''
        MODEL_SIZE_LIST='125m 350m 1.3b 2.7b 6.7b 13b'
    elif [ $MODEL_NAME == 'llama3.2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.2-
        MODEL_SIZE_LIST='1B 3B'
        MODEL_POSTFIX=''
    elif [ $MODEL_NAME == 'llama3.1' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.1-
        MODEL_SIZE_LIST='8B'
        MODEL_POSTFIX=''
    elif [ $MODEL_NAME == 'gemma2' ]
    then
        MODEL_PREFIX=google/gemma-2-
        MODEL_SIZE_LIST='2b'
        MODEL_POSTFIX=''
    elif [ $MODEL_NAME == 'gemma3' ]
    then
        MODEL_PREFIX=google/gemma-3-
        MODEL_SIZE_LIST='1b 4b'
        MODEL_POSTFIX='-pt'
    fi

    # --- Loop through hyperparameter combinations ---
    for MODEL_SIZE in $MODEL_SIZE_LIST
    do
        for STRUCTURE in "${STRUCTURE[@]}"
        do
            for METHOD in "${METHOD[@]}"
            do
                for LORA_RANK in "${LORA_RANK[@]}"
                do
                    for SLIM_LORA in "${SLIM_LORA[@]}"
                    do
                        for QUANTIZE_LORA in "${QUANTIZE_LORA[@]}"
                        do
                            for QUANTIZE_WEIGHT in "${QUANTIZE_WEIGHT[@]}"
                            do
                                for SLIM_QUANT in "${SLIM_QUANT[@]}"
                                do
                                    if [ "${SLIM_QUANT}" == "true" ]; then
                                        TILED_WEIGHT_QUANTIZATION=false
                                    else
                                        TILED_WEIGHT_QUANTIZATION=true
                                    fi
                                    for FINE_TUNE in "${FINE_TUNE[@]}"
                                    do
                                        JOB_NAME=${MODEL_NAME}_${MODEL_SIZE}_${METHOD}_${STRUCTURE}_lr${LORA_RANK}_sparsity${SPARSITY_RATIO}_slimlora${SLIM_LORA}_quantlora${QUANTIZE_LORA}_quantweight${QUANTIZE_WEIGHT}_slimquant${SLIM_QUANT}_finetune${FINE_TUNE}
                                        SAVE_CHECKPOINT_PATH="checkpoints/${MODEL_NAME}_${MODEL_SIZE}_${METHOD}_${STRUCTURE}_lr${LORA_RANK}_sparsity${SPARSITY_RATIO}_slimlora${SLIM_LORA}_quantlora${QUANTIZE_LORA}_quantweight${QUANTIZE_WEIGHT}_slimquant${SLIM_QUANT}_finetune${FINE_TUNE}"
                                        # Construct the arguments for the script
                                        

                                        sbatch --account=def-mmehride \
                                            --job-name="${GPU_TYPE}${JOB_NAME}" \
                                            --gpus-per-node=${NGPUS_PER_NODE} \
                                            --ntasks-per-node=${NTASKS_PER_NODE} \
                                            --mem=${MEM}G \
                                            --time=${TIME} \
                                            scripts/job_template.sh \
                                            "${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX}" \
                                            "${STRUCTURE}" \
                                            "${SPARSITY_RATIO}" \
                                            "${METHOD}" \
                                            "${LORA_RANK}" \
                                            "${SLIM_LORA}" \
                                            "${SEPARATE_LORA}" \
                                            "${QUANTIZE_LORA}" \
                                            "${LORA_TILE_SIZE}" \
                                            "${PAD_LORA}" \
                                            "${CALIBRATION_DATASET}" \
                                            "${NUM_CALIBRATION_SAMPLES}" \
                                            "${QUANTIZE_WEIGHT}" \
                                            "${BITWIDTH}" \
                                            "${TILED_WEIGHT_QUANTIZATION}" \
                                            "${WEIGHT_TILE_SIZE}" \
                                            "${SLIM_QUANT}" \
                                            "${LOCAL_FILES_ONLY}" \
                                            "${EVAL_DATASET}" \
                                            "${EVALUATE_PERPLEXITY}" \
                                            "${TEST_LMHARNESS}" \
                                            "${FINE_TUNE}" \
                                            "${OPTIMIZER}" \
                                            "${SCALE_IMPORTANT_WEIGHTS}" \
                                            "${MASKLLM_CHECKPOINT}" \
                                            "${QUANTIZE_INPUT}" \
                                            "${INPUT_BITWIDTH}" \
                                            "${INPUT_GROUP_SIZE}" \
                                            "${JOINT_PQ_MIXING_FACTOR}" \
                                            "${WANDB}" \
                                            "${HF_TOKEN}" \
                                            "${SAVE_CHECKPOINT_PATH}" \
                                            "${OUTPUT_CSV_FILE}"

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

    