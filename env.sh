# Shared paths, sourced by setup_env.sh and run.slurm.
export LLMNAV_ENV="${LLMNAV_ENV:-$HOME/envs/llmnav}"
if [ -d "$HOME/scratch" ]; then
    export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$LLMNAV_ENV/ms-playwright}"
