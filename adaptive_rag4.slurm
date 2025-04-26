#!/bin/bash

#SBATCH --job-name=adaptiverag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=out_%j.log
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wahibkapdi@tamu.edu

# Optional environment variables
# export HF_HOME=/scratch/user/wahibkapdi/huggingface
# export TRANSFORMERS_CACHE=/scratch/user/wahibkapdi/fall2024/free_generation/huggingface

echo "[INFO] Starting elastic server on port 9300..."
srun --exact -n1 --mem-per-cpu 3770M bash ./run_elastic_search.sh &

echo "[INFO] Starting retriever server on port 8000..."
srun --exact -n1 --mem-per-cpu 3770M bash ./run_retriever_server.sh &

echo "[INFO] Starting LLM server on port 8010..."
srun --exact -n1 --mem-per-cpu 3770M bash ./run_llm_server.sh &

# Wait a few seconds to ensure servers boot up
echo "[INFO] Waiting for servers to initialize..."
sleep 60

echo "[INFO] Starting training job..."
srun --exact -n1 --mem-per-cpu 3770M bash ./run_training.sh
