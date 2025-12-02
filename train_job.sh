#!/bin/bash
#SBATCH --account=def-masaduzz-ab
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=codegpt-py-java
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

module load python/3.10 cuda/11.8

cd $SCRATCH/llm-sniffer
source env/bin/activate

python scripts/train.py

echo "Job finished: $(date)"
