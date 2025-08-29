#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=rrg-mmehride
#SBATCH --job-name=slim # Base name, will be overridden by submit_jobs.sh

echo "SLURM Job $SLURM_JOB_ID started at $(date)"

ARG_MODEL_NAME="${1:-'meta-llama/Llama-3.2-1B'}"
ARG_STRUCTURE="${2:-'2:4'}"
ARG_SPARSITY_RATIO="${3:-0.5}"
ARG_METHOD="${4:-wanda}"
ARG_LORA_RANK="${5:-0.0}"
ARG_SLIM_LORA="${6:-'false'}"
ARG_SEPARATE_LORA="${7:-'true'}"
ARG_QUANTIZE_LORA="${8:-'false'}"
ARG_LORA_TILE_SIZE="${9:-128}"
ARG_PAD_LORA="${10:-'true'}"
ARG_CALIBRATION_DATASET="${11:-c4}"
ARG_NUM_CALIBRATION_SAMPLES="${12:-128}"
ARG_QUANTIZE_WEIGHT="${13:-'false'}"
ARG_BITWIDTH="${14:-4}"
ARG_TILED_WEIGHT_QUANTIZATION="${15:-'false'}"
ARG_WEIGHT_TILE_SIZE="${16:-128}"
ARG_SLIM_QUANT="${17:-'false'}"
ARG_LOCAL_FILES_ONLY="${18:-'false'}"
ARG_EVAL_DATASET="${19:-wikitext2}"
ARG_EVALUATE_PERPLEXITY="${20:-'true'}"
ARG_TEST_LMHARNESS="${21:-'false'}"
ARG_FINE_TUNE="${22:-'false'}"
ARG_OPTIMIZER="${23:-adafactor}"
ARG_SCALE_IMPORTANT_WEIGHTS="${24:-'false'}"
ARG_MASKLLM_CHECKPOINT="${25:-""}"
ARG_QUANTIZE_INPUT="${26:-'false'}"
ARG_INPUT_BITWIDTH="${27:-8}"
ARG_INPUT_GROUP_SIZE="${28:-128}"
ARG_JOINT_PQ_MIXING_FACTOR="${29:-2.1}"
ARG_WANDB="${30:-'true'}"
ARG_HF_TOKEN="${31:-""}"
ARG_SAVE_CHECKPOINT_PATH="${32:-'checkpoints/${ARG_MODEL_NAME}_${ARG_METHOD}_${ARG_STRUCTURE}_lr${ARG_LORA_RANK}_sparsity${ARG_SPARSITY_RATIO}'}"
ARG_OUTPUT_CSV_FILE="${33:-'results/results.csv'}"


SCRIPT_TO_RUN=scripts/run_slim_args.sh

module load apptainer 
export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="$SLURM_TMPDIR/data" 
export MASTER_PORT=29501
export OMP_NUM_THREADS=12

mkdir -p "$HF_HOME"

USERNAME=$(whoami)


# --- Data and Container Preparation ---
if [ "$ARG_COPY_DATA" = true ]; then
    echo "Copying data to SLURM_TMPDIR..."
    DATA_DIR_SRC="/home/${USERNAME}/projects/def-mmehride/${USERNAME}/data"
    DATA_DIR_TMP="$SLURM_TMPDIR/data"
    cp -r "$DATA_DIR_SRC" "$SLURM_TMPDIR/"
    echo "Data copied to $DATA_DIR_TMP"
else
    echo "Skipping data copy as per user request."
    DATA_DIR_TMP="/home/${USERNAME}/projects/def-mmehride/${USERNAME}/data" # Use the original data directory
fi

echo "Preparing container..."
rm -rf $SLURM_TMPDIR/torch-one-shot.sif;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif;
tar -xf /home/${USERNAME}/projects/def-mmehride/${USERNAME}/torch-one-shot.tar -C $SLURM_TMPDIR;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs;
cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs/ca-bundle.crt;

# --- Execute inside Singularity ---
echo "Executing run_experiment.sh inside Singularity..."
singularity exec \
    --bind $PWD:/home/${USERNAME} \
    --bind $SLURM_TMPDIR:/tmp \
    --nv \
    ${SLURM_TMPDIR}/torch-one-shot.sif \
    mkdir -p /home/${USERNAME}/data
    
singularity exec \
    --bind $PWD:/home/${USERNAME} \
    --bind $SLURM_TMPDIR:/tmp \
    --bind $DATA_DIR_TMP:/home/${USERNAME}/data \
    --nv ${SLURM_TMPDIR}/torch-one-shot.sif \
    bash "${SCRIPT_TO_RUN}" \
    "${ARG_MODEL_NAME}" \
    "${ARG_STRUCTURE}" \
    "${ARG_SPARSITY_RATIO}" \
    "${ARG_METHOD}" \
    "${ARG_LORA_RANK}" \
    "${ARG_SLIM_LORA}" \
    "${ARG_SEPARATE_LORA}" \
    "${ARG_QUANTIZE_LORA}" \
    "${ARG_LORA_TILE_SIZE}" \
    "${ARG_PAD_LORA}" \
    "${ARG_CALIBRATION_DATASET}" \
    "${ARG_NUM_CALIBRATION_SAMPLES}" \
    "${ARG_QUANTIZE_WEIGHT}" \
    "${ARG_BITWIDTH}" \
    "${ARG_TILED_WEIGHT_QUANTIZATION}" \
    "${ARG_WEIGHT_TILE_SIZE}" \
    "${ARG_SLIM_QUANT}" \
    "${ARG_LOCAL_FILES_ONLY}" \
    "${ARG_EVAL_DATASET}" \
    "${ARG_EVALUATE_PERPLEXITY}" \
    "${ARG_TEST_LMHARNESS}" \
    "${ARG_FINE_TUNE}" \
    "${ARG_OPTIMIZER}" \
    "${ARG_SCALE_IMPORTANT_WEIGHTS}" \
    "${ARG_MASKLLM_CHECKPOINT}" \
    "${ARG_QUANTIZE_INPUT}" \
    "${ARG_INPUT_BITWIDTH}" \
    "${ARG_INPUT_GROUP_SIZE}" \
    "${ARG_JOINT_PQ_MIXING_FACTOR}" \
    "${ARG_WANDB}" \
    "${ARG_HF_TOKEN}" \
    "${ARG_SAVE_CHECKPOINT_PATH}" \
    "${ARG_OUTPUT_CSV_FILE}"

echo $ARG_WANDB

echo "Singularity execution finished successfully."
echo "SLURM Job $SLURM_JOB_ID finished at $(date)"