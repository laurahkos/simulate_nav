#!/bin/bash --login
#SBATCH -J llm_nav
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --array=0-5
#SBATCH -o slurm-%x-%A_%a.out
# One task per model × department: 0-2 = Phi-3, Mistral, Llama on DEPTS[0]; 3-5 on DEPTS[1].
# Override at submission, e.g.:
#   PASSES=4 sbatch run.slurm
#   SHUFFLE=1 sbatch run.slurm             # control: randomised link order
#   PROMPT="what does sociology at Liverpool offer postgrads?" DEPTS=sociology sbatch --array=0-2 run.slurm
set -e

PROMPT="${PROMPT:-What is the postgraduate training offer at the University of Liverpool?}"
DEPTS=(${DEPTS:-sociology geography})
PASSES="${PASSES:-1}"
MAX_PAGES="${MAX_PAGES:-8}"
MODELS=(microsoft/Phi-3-mini-4k-instruct mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.1-8B-Instruct)
TAGS=(phi3 mistral7b llama8b)

MODEL="${MODELS[SLURM_ARRAY_TASK_ID % 3]}"
DEPT="${DEPTS[SLURM_ARRAY_TASK_ID / 3]}"
OUT_DIR="runs/${DEPT}_${TAGS[SLURM_ARRAY_TASK_ID % 3]}_p${PASSES}${SHUFFLE:+_shuffled}"

module purge
module load libs/cuda
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
source "$LLMNAV_ENV/bin/activate"
export HF_HUB_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'"

python simulate.py --agentic --prompt "$PROMPT" --dept "$DEPT" --answer-model "$MODEL" \
    --max-pages "$MAX_PAGES" --passes "$PASSES" --out-dir "$OUT_DIR" ${SHUFFLE:+--shuffle-links}
