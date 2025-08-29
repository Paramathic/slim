export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"


export WANDB_API_KEY="WANDB_API_KEY_PLACEHOLDER"
export WANDB_MODE="offline"


MODEL_NAME="${1:-'meta-llama/Llama-3.2-1B'}"
STRUCTURE="${2:-'2:4'}"
SPARSITY_RATIO="${3:-0.5}"
METHOD="${4:-wanda}"
LORA_RANK="${5:-0.0}"
SLIM_LORA="${6:-'false'}"
SEPARATE_LORA="${7:-'true'}"
QUANTIZE_LORA="${8:-'false'}"
LORA_TILE_SIZE="${9:-128}"
PAD_LORA="${10:-'true'}"
CALIBRATION_DATASET="${11:-c4}"
NUM_CALIBRATION_SAMPLES="${12:-128}"
QUANTIZE_WEIGHT="${13:-'false'}"
BITWIDTH="${14:-4}"
TILED_WEIGHT_QUANTIZATION="${15:-'false'}"
WEIGHT_TILE_SIZE="${16:-128}"
SLIM_QUANT="${17:-'false'}"
LOCAL_FILES_ONLY="${18:-'false'}"
EVAL_DATASET="${19:-wikitext2}"
EVALUATE_PERPLEXITY="${20:-'true'}"
TEST_LMHARNESS="${21:-'false'}"
FINE_TUNE="${22:-'false'}"
OPTIMIZER="${23:-adafactor}"
SCALE_IMPORTANT_WEIGHTS="${24:-'false'}"
MASKLLM_CHECKPOINT="${25:-""}"
QUANTIZE_INPUT="${26:-'false'}"
INPUT_BITWIDTH="${27:-8}"
INPUT_GROUP_SIZE="${28:-128}"
JOINT_PQ_MIXING_FACTOR="${29:-2.1}"
WANDB="${30:-'true'}"
HF_TOKEN="${31:-""}"
SAVE_CHECKPOINT_PATH="${32:-""}"
OUTPUT_CSV_FILE="${33:-'results/results.csv'}"


if [ "$SLIM_LORA" = "true" ]; then
    SLIM_LORA='--slim_lora'
else
    SLIM_LORA=""
fi

if [ "$SEPARATE_LORA" = "true" ]; then
    SEPARATE_LORA='--separate_lora'
else
    SEPARATE_LORA=""
fi

if [ "$QUANTIZE_LORA" = "true" ]; then
    QUANTIZE_LORA='--quantize_lora'
else
    QUANTIZE_LORA=""
fi

if [ "$QUANTIZE_WEIGHT" = "true" ]; then
    QUANTIZE_WEIGHT='--quantize_weight'
else
    QUANTIZE_WEIGHT=""
fi

if [ "$TILED_WEIGHT_QUANTIZATION" = "true" ]; then
    TILED_WEIGHT_QUANTIZATION='--tiled_weight_quantization'
else
    TILED_WEIGHT_QUANTIZATION=""
fi

if [ "$SLIM_QUANT" = "true" ]; then
    SLIM_QUANT='--slim_quant'
else
    SLIM_QUANT=""
fi

if [ "$EVALUATE_PERPLEXITY" = "true" ]; then
    EVALUATE_PERPLEXITY='--evaluate_perplexity'
else
    EVALUATE_PERPLEXITY=""
fi

if [ "$TEST_LMHARNESS" = "true" ]; then
    TEST_LMHARNESS='--test_lmharness'
else
    TEST_LMHARNESS=""
fi

if [ "$FINE_TUNE" = "true" ]; then
    FINE_TUNE='--fine_tune'
else
    FINE_TUNE=""
fi

if [ "$PAD_LORA" = "true" ]; then
    PAD_LORA='--pad_lora'
else
    PAD_LORA=""
fi

if [ "$LOCAL_FILES_ONLY" = "true" ]; then
    LOCAL_FILES_ONLY='--local_files_only'
else
    LOCAL_FILES_ONLY=""
fi

if [ "$SCALE_IMPORTANT_WEIGHTS" = "true" ]; then
    SCALE_IMPORTANT_WEIGHTS='--scale_important_weights'
else
    SCALE_IMPORTANT_WEIGHTS=""
fi

if [ "$QUANTIZE_INPUT" = "true" ]; then
    QUANTIZE_INPUT='--quantize_input'
else
    QUANTIZE_INPUT=""
fi

if [ "$WANDB" = "true" ]; then
    WANDB='--use_wandb'
else
    WANDB=""
fi

if [ -n "$HF_TOKEN" ]; then
    HF_TOKEN_ARGS="--hf_token $HF_TOKEN"
    export HF_TOKEN=${HF_TOKEN:-""}
else
    HF_TOKEN_ARGS=""
fi

if [ -n "$SAVE_CHECKPOINT_PATH" ]; then
    SAVE_CHECKPOINT_PATH="--save_checkpoint_path $SAVE_CHECKPOINT_PATH"
fi

if [ -n "$MASKLLM_CHECKPOINT" ]; then
    MASKLLM_CHECKPOINT="--maskllm_checkpoint $MASKLLM_CHECKPOINT"
else
    MASKLLM_CHECKPOINT=""
fi


SHIFT_ZERO_METRICS='--shift_zero_metrics'
EVAL_BATCH_SIZE=1


python main.py \
    --model $MODEL_NAME \
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
    --output_csv_path $OUTPUT_CSV_FILE \
    $FINE_TUNE \
    $EVALUATE_PERPLEXITY \
    $LOCAL_FILES_ONLY \
    $QUANTIZE_INPUT \
    --input_bitwidth $INPUT_BITWIDTH \
    --input_group_size $INPUT_GROUP_SIZE \
    --nsample $NUM_CALIBRATION_SAMPLES \
    --optimizer $OPTIMIZER \
    $QUANTIZE_LORA \
    --lora_tile_size $LORA_TILE_SIZE \
    $TILED_WEIGHT_QUANTIZATION \
    --weight_tile_size $WEIGHT_TILE_SIZE \
    $HF_TOKEN_ARGS \
    --joint_pq_mixing_factor $JOINT_PQ_MIXING_FACTOR \
    --calibration_dataset $CALIBRATION_DATASET \
    $PAD_LORA \
    $SCALE_IMPORTANT_WEIGHTS \
    $MASKLLM_CHECKPOINT \
    $WANDB \
    $SAVE_CHECKPOINT_PATH 