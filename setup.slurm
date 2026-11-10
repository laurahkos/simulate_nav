#!/bin/bash --login
#SBATCH -J llm_setup
#SBATCH -p multicore_small
#SBATCH -n 4
#SBATCH -t 03:00:00
#SBATCH -o slurm-%x-%j.out
# Submit with: export HF_TOKEN=hf_... && sbatch setup.slurm
set -e
cd "$SLURM_SUBMIT_DIR"
: "${HF_TOKEN:?export HF_TOKEN before sbatch}"
bash setup_env.sh
