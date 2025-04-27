#!/usr/bin/env bash

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
# Define the BASE model here
BASE_MODEL=google/flan-t5-xl
# Short name for directory structure
BASE_MODEL_SHORT_NAME=flan-t5-xl
# Define the source of silver labels (the LLM used to generate them)
LLM_NAME=flan_t5_xl
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
# Define LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q v" # T5 target modules (query and value layers)
# Other parameters
LEARNING_RATE=1e-4 # Often good starting point for LoRA
BATCH_SIZE=64      # Can potentially increase this with LoRA compared to full fine-tuning
MAX_SEQ_LENGTH=384
GPU="0,1" # Set your target GPU

# Add error checking: exit if any command fails
set -e

# --- Define Data Paths ---
# Make sure these paths are correct for your setup
TRAIN_DATA_FILE="./data/${DATASET_NAME}/${LLM_NAME}/binary_silver/train.json"
VALID_DATA_FILE="./data/${DATASET_NAME}/${LLM_NAME}/silver/valid.json"
PREDICT_DATA_FILE="./data/${DATASET_NAME}/predict.json"

# Check if data files exist
if [ ! -f "$TRAIN_DATA_FILE" ]; then
    echo "Error: Training data file not found at $TRAIN_DATA_FILE"
    exit 1
fi
if [ ! -f "$VALID_DATA_FILE" ]; then
    echo "Error: Validation data file not found at $VALID_DATA_FILE"
    exit 1
fi
if [ ! -f "$PREDICT_DATA_FILE" ]; then
    echo "Error: Prediction data file not found at $PREDICT_DATA_FILE"
    exit 1
fi


# We'll loop only once as per the original script intent, but keep the structure
for EPOCH_COUNT in 20 30 40 # Represents the number of epochs to train for *in this run*
do
    # Train
    # Output directory specific to LoRA and base model
    # Using EPOCH_COUNT in the path to represent the number of epochs trained for this specific adapter
    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${BASE_MODEL_SHORT_NAME}/${LLM_NAME}/r${LORA_R}_alpha${LORA_ALPHA}/epochs_${EPOCH_COUNT}/${DATE}
    echo "Creating Training Output Directory: ${TRAIN_OUTPUT_DIR}"
    mkdir -p ${TRAIN_OUTPUT_DIR}

    echo "======================================================"
    echo " Starting Training for ${EPOCH_COUNT} Epoch(s)..."
    echo " Outputting adapter to: ${TRAIN_OUTPUT_DIR}"
    echo "======================================================"
    # Pass BASE_MODEL for training
    CUDA_VISIBLE_DEVICES=${GPU} python run_flan_t5_xl_lora.py \
        --model_name_or_path ${BASE_MODEL} \
        --do_train \
        --train_file ${TRAIN_DATA_FILE} \
        --question_column question \
        --answer_column answer \
        --learning_rate ${LEARNING_RATE} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 2 \
        --output_dir ${TRAIN_OUTPUT_DIR} \
        --overwrite_cache \
        --train_column 'train' \
        --num_train_epochs ${EPOCH_COUNT} \
        --seed 42 \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target_modules ${TARGET_MODULES} \
        --checkpointing_steps "epoch" \
        --report_to "none" # Change to wandb/tensorboard if needed
        # --with_tracking # Uncomment if using tracking
        # --push_to_hub \ # Uncomment to push final adapter
        # --hub_model_id "your-hub-username/your-adapter-repo-name" # Set if pushing

    # Validation
    echo "======================================================"
    echo " Starting Validation..."
    echo " Using Adapter: ${TRAIN_OUTPUT_DIR}"
    echo " Using Base Model: ${BASE_MODEL}"
    echo "======================================================"
    VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid/
    echo "Creating Validation Output Directory: ${VALID_OUTPUT_DIR}"
    mkdir -p ${VALID_OUTPUT_DIR}

    # Pass TRAIN_OUTPUT_DIR (adapter) and BASE_MODEL for validation
    CUDA_VISIBLE_DEVICES=${GPU} python run_flan_t5_xl_lora.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --base_model_name_or_path ${BASE_MODEL} \
        --do_eval \
        --validation_file ${VALID_DATA_FILE} \
        --question_column question \
        --answer_column answer \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --output_dir ${VALID_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target_modules ${TARGET_MODULES} \
        --report_to "none"

    # Prediction on Test Questions
    echo "======================================================"
    echo " Starting Prediction on Test Set..."
    echo " Using Adapter: ${TRAIN_OUTPUT_DIR}"
    echo " Using Base Model: ${BASE_MODEL}"
    echo "======================================================"
    PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict/
    echo "Creating Prediction Output Directory: ${PREDICT_OUTPUT_DIR}"
    mkdir -p ${PREDICT_OUTPUT_DIR}

    # Pass TRAIN_OUTPUT_DIR (adapter) and BASE_MODEL for prediction
    # Note: predict.json seems to use 'validation' as split key in args, ensure file structure matches
    CUDA_VISIBLE_DEVICES=${GPU} python run_flan_t5_xl_lora.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --base_model_name_or_path ${BASE_MODEL} \
        --do_eval \
        --validation_file ${PREDICT_DATA_FILE} \
        --question_column question \
        --answer_column answer \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target_modules ${TARGET_MODULES} \
        --report_to "none"

    echo "======================================================"
    echo " Finished Run for ${EPOCH_COUNT} Epoch(s)."
    echo "======================================================"
done

echo "All runs finished."