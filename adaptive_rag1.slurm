#!/bin/bash

#SBATCH --job-name=elasticsearch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=out_%j.log     # Save output to log file
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=wahibkapdi@tamu.edu    #Send all emails to email_address
#SBATCH --export=ALL,HF_HOME=/scratch/user/wahibkapdi/huggingface,TRANSFORMERS_CACHE=/scratch/user/wahibkapdi/fall2024/free_generation/huggingface


#First Executable Line
srun --exact -n1 --mem-per-cpu 3770M  bash ./run_elastic_search.sh
# srun --exact -n1 --mem-per-cpu 3770M  bash ./run_retriever_server.sh
# srun --exact -n1 --mem-per-cpu 3770M  bash ./run_llm_server.sh
# srun --exact -n1 --mem-per-cpu 3770M  bash ./run_training.sh