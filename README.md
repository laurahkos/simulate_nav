# simulate_nav
Given a prompt, simulate an LLM agent navigating university webpages, and log what it retrieves and how it answers.

`simulate.py` starts from a department page on liverpool.ac.uk and follows links until it has read a fixed number of pages. It selects evidence sentences from the pages it reads and asks an instruct model to answer the prompt from that evidence alone. Every link choice and every generated token is logged, so a trace shows where retrieval went wrong as well as what the final answer was. The pipeline was built for a study of how small open models find postgraduate research training content on the University of Liverpool site, starting from a generic question *"What is the postgraduate training offer at the University of Liverpool?"*, informed by the attitude that a user cannot know what they do not know. 

## How it works

1. **Fetch.** Pages are rendered with headless Chromium (Playwright) and converted to text. Cookie banners and site chrome are stripped. URLs are canonicalised, so fragments, `utm_*` parameters and trailing slashes don't create duplicate pages.
2. **Probe.** GPT-2 computes token surprisal over each page. Each sentence is scored as postgraduate keyword count × mean surprisal (capped). The top six sentences across all pages read become the evidence.
3. **Navigate.** There are two modes:
   - **Agentic (`--agentic`).** The instruct model gets the question, a summary of the findings so far, and up to 15 numbered links, with links into the starting department listed first. It replies in the form `CHOICE / REASON / STOP`, and decoding is greedy.
   - **Standard (the default).** GPT-2 ranks links by perplexity of the link text given the query.
In both modes, if a page has no unvisited links, the agent backtracks to the next unvisited link from an earlier page, in the order that page showed them. Pages whose text duplicates a page already read are skipped and don't count against the page budget.
4. **Answer.** The answer model writes a response using only the evidence (temperature 0.3, top-p 0.9). Each token is logged with its probability and three alternatives. Grounding is the share of answer content words (4+ letters) that appear verbatim in the evidence. Each answer sentence is also matched to its closest evidence item.
5. **Refine (optional, `--passes N`).** The script re-runs with the query narrowed by high-interest terms from the previous pass. It stops early once a pass finds nothing new.

**Shuffle control (`--shuffle-links`).** This randomises the order of the links shown to the agent, using a seeded shuffle per step. Comparing shuffled and unshuffled runs shows whether a choice depends on a link's text or its position. Backtracking uses the shuffled order.

## Models

| Role | Model | Reference |
|---|---|---|
| Surprisal probe and standard-mode navigator | [`gpt2`](https://huggingface.co/openai-community/gpt2) | Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* |
| Instruct navigator and answer model (default) | [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | Abdin et al. (2024) |
| Instruct navigator and answer model | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Jiang et al. (2023) |
| Instruct navigator and answer model | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Grattafiori et al. (2024) |

Llama 3.1 is gated: accept its licence on Hugging Face and use `export HF_TOKEN=...` before env setup.

## Setup
Requires Python ≥ 3.9.
```bash
bash setup_env.sh              # venv, torch, requirements, Chromium, model download
bash setup_env.sh --no-models  # skip the model download
```
Paths are set in `env.sh` and can be overridden:
| Variable | Default |
|---|---|
| `LLMNAV_ENV` | `~/envs/llmnav` |
| `HF_HOME` | `~/scratch/hf_cache` if `~/scratch` exists, else `~/.cache/huggingface` |
| `PLAYWRIGHT_BROWSERS_PATH` | inside the venv |
On Linux, torch is installed from the CUDA 12.6 wheel index; set `TORCH_INDEX` to change this. On macOS, the default wheel is used.

## Usage
```bash
source env.sh && source "$LLMNAV_ENV/bin/activate"
#agentic run with Phi-3 on the sociology department
python simulate.py --agentic --dept sociology \
    --prompt "What is the postgraduate training offer at the University of Liverpool?"
#same run with a different model and shuffled link order
python simulate.py --agentic --dept geography \
    --answer-model meta-llama/Llama-3.1-8B-Instruct --shuffle-links --out-dir runs/geo_llama_shuf
#inspect what the fetcher extracts from a page, then exit
python simulate.py --start-url https://www.liverpool.ac.uk/geography-and-planning/ --dump-fetch
```
| Flag | Default | |
|---|---|---|
| `--prompt` | generic postgraduate question | |
| `--dept` | `sociology` | `sociology` or `geography` |
| `--start-url` | | overrides `--dept` |
| `--max-pages` | 8 | pages actually read |
| `--agentic` | off | instruct model chooses links |
| `--nav-model` | `gpt2` | probe / standard-mode navigator |
| `--answer-model` | Phi-3-mini | also the navigator in agentic mode |
| `--shuffle-links` | off | randomise link order |
| `--passes` | 1 | query-refinement passes |
| `--no-answer` | off | navigate only |
| `--seed` | 0 | |
| `--out-dir` | `./run` | |

## Running on Slurm
The scripts assume using a cluster with a `gpuA40GB` partition; edit the `#SBATCH` lines for other clusters or extract code logic to run locally.

```bash
export HF_TOKEN=hf_...
sbatch setup.slurm                 # build the env and download models (CPU node)

sbatch run.slurm                   # array 0-5: 3 models × 2 departments
SHUFFLE=1 sbatch run.slurm         # same, with randomised link order
PASSES=4 sbatch run.slurm          # with query refinement
PROMPT="..." DEPTS=sociology sbatch --array=0-2 run.slurm
```

## Output
Each run writes to `--out-dir`:
- **`trace.txt`**: a readable log covering each page's text summary, the links offered, the model's choice and reason, findings with their scores, and the answer with grounding statistics.
- **`trace.json`**: the same information in structured form:
  - `meta`
  - `visited_urls` and `skipped`
  - `all_findings`
  - `pages[]`, with the links offered, the `chosen` flag, raw model output and surprisal profiles
  - `answer`, with `token_log`, `grounding_stats`, `sentence_sources` and `evidence_used`
With `--passes > 1`, each pass is written to `pass<N>/` and a summary to `passes.json`.

## Limitations
- **Small models, greedy decoding.** Results don't predict how production browsing agents behave.
- **One run per configuration.** Differences between conditions rest on single traces.
- **Grounding checks words, not claims.** An answer can recombine words from the evidence into a false statement and still score as grounded.
- **Stated reasons are generated text.** The `REASON` given for each link choice is not evidence of what caused the choice.
- **Script-rendered pages may come back empty.** The fetcher waits for the page to load, but pages that load their content later with JavaScript can still return no text.
- **Hard-coded seeds.** Seed URLs and keyword lists are specific to liverpool.ac.uk.

## References 

- Abdin, M., Aneja, J., Awadalla, H., et al. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv:2404.14219. https://arxiv.org/abs/2404.14219
- Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). *The Llama 3 Herd of Models*. arXiv:2407.21783. https://arxiv.org/abs/2407.21783
- Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023). *Mistral 7B*. arXiv:2310.06825. https://arxiv.org/abs/2310.06825
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI technical report.
- Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6
