#!/bin/bash

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=prajjwal1/bert-tiny
LLM_NAME=flan_t5_xl   # You can keep this or rename if you like
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
GPU=0

for EPOCH in 20
do
    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/tiny_bert/${LLM_NAME}/epoch/${EPOCH}/${DATE}/train
    mkdir -p ${TRAIN_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier_encoder.py \
        --model_name_or_path ${MODEL} \
        --train_file ./data/${DATASET_NAME}/${LLM_NAME}/binary_silver/train.json \
        --output_dir ${TRAIN_OUTPUT_DIR} \
        --do_train \
        --num_train_epochs ${EPOCH} \
        --batch_size 32 \
        --learning_rate 2e-5

    VALID_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/tiny_bert/${LLM_NAME}/epoch/${EPOCH}/${DATE}/valid
    mkdir -p ${VALID_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier_encoder.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/valid.json \
        --output_dir ${VALID_OUTPUT_DIR} \
        --do_eval \
        --batch_size 100

    PREDICT_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/tiny_bert/${LLM_NAME}/epoch/${EPOCH}/${DATE}/predict
    mkdir -p ${PREDICT_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier_encoder.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --prediction_file ./data/${DATASET_NAME}/predict.json \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --do_predict \
        --batch_size 100
done
