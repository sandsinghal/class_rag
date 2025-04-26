#!/bin/bash

#SBATCH --job-name=adaptiverag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=out_%j.log
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wahibkapdi@tamu.edu

# Start all servers concurrently in the background
echo "[INFO] Launching Elastic server on port 9300..."
bash ./run_elastic_search.sh > elastic.log 2>&1 &

echo "[INFO] Launching Retriever server on port 8000..."
bash ./run_retriever_server.sh > retriever.log 2>&1 &

echo "[INFO] Launching LLM server (flan-t5-xl) on port 8010..."
bash ./run_llm_server.sh > llm.log 2>&1 &

# Give servers time to initialize
echo "[INFO] Waiting for all servers to become available..."
sleep 45

# Run training job in foreground
echo "[INFO] Starting training..."
bash ./run_training.sh > training.log 2>&1
