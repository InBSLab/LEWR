# LEWR

A pipeline that turns a natural-language process description into a complete
BPMN workflow (tasks, dependencies, gateways) and recommends the most
relevant APIs for each task.

The core idea builds on the BPMN Assistant approach for single-shot BPMN
generation, and adds:

- **Task semantic enhancement** — rewrites user-facing task language into the
  technical/API-oriented phrasing needed for retrieval.
- **API recommendation** — vector retrieval over an API embedding index,
  followed by LLM re-ranking, to produce a Top-5 candidate list per task.
- **Structured, evaluation-ready output** — every run returns both the raw
  BPMN structure and a normalized `evaluation_format` with API bindings.

## Project layout

Everything lives in two top-level folders: `model_simple/` holds the
generation pipeline itself, and `evaluation/` holds the batch/statistical
evaluation harness. `REPRODUCIBILITY.md`, `requirements.txt`, and
`.gitignore` sit at the repo root.

### `model_simple/`

`complete_main.py` is the CLI entry point. It defines
`CompleteWorkflowSystem`, the top-level class you call (or invoke from the
command line) to run the whole pipeline end-to-end.

`complete_workflow_generator.py` does the orchestration. It defines
`CompleteWorkflowGenerator`, which drives BPMN generation and then hands
each task off to `APIRecommender` for API matching.

`enhanced_bpmn_generator.py` contains `EnhancedBPMNGenerator`, the
LLM-driven component that actually produces the BPMN structure — tasks,
dependencies, and gateways — from the process description.

`llm_confirmer.py` provides LLM-based confirmation/validation of API
candidates before they're finalized. It also wraps the OpenAI-SDK-based
client used for this step; if no API key is set, it falls back to a mock
client so the rest of the pipeline still runs for offline testing.

`task_enhancer.py` rewrites user-facing task language into the more
technical, API-oriented phrasing that retrieval needs — this is the "task
semantic enhancement" step mentioned above.

`simcse_encoder.py` is a SimCSE-style sentence embedding encoder, used to
embed tasks and API descriptions into the same vector space for retrieval.

`mine_domain_keywords.py` is an offline tool for mining domain keywords
from a corpus; it's used to build data assets the pipeline depends on,
not called at runtime.

`verify_gateway_fix.py` is a small standalone script for manually sanity
checking gateway generation — not part of the main pipeline, just a check
you can run by hand.

The active call chain through these files is: `complete_main.py` calls
into `complete_workflow_generator.py` (`CompleteWorkflowGenerator` +
`APIRecommender`), which in turn calls `enhanced_bpmn_generator.py`
(`EnhancedBPMNGenerator`). `llm_confirmer.py`, `task_enhancer.py`,
`simcse_encoder.py`, and `mine_domain_keywords.py` support that chain at
various stages, or are used to build the data assets it depends on.

### `evaluation/`

`run_evaluation.py` is a batch evaluation harness. Note: it was written
against a broader internal evaluation framework and imports a `scenarios`
module (`BPMN_TEST_SCENARIOS`, `UnifiedOutput`, `save_results`) that isn't
part of this repo. It's included as a reference for how the model was
evaluated — you'll need to supply your own `scenarios.py`, or swap in your
own harness, to run it as-is.

`run_repeated_evaluation.py` runs n independent generations per scenario
(a fresh system instance per run, so the LLM-confirmer cache can't leak
across runs) and writes out raw per-run metrics as JSON.

`statistical_analysis.py` takes those raw metrics and produces the actual
statistics: Welch's t-test with Holm-Bonferroni correction, Hedges' g
effect sizes, and 95% confidence intervals.

`sample_outputs/` holds representative generated outputs — add your own
as you generate them.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need to provide your own data assets (not included in this
repo, since they're generated from your own API catalog):

```
model_simple/data/api_name.json
model_simple/data/api_description.json
model_simple/embeddings/api_features_roberta.npy
```

## API key

This project calls the DeepSeek chat completion API for BPMN generation and
API re-ranking. Set your key as an environment variable before running
anything — no key is hardcoded anywhere in this codebase:

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

If you use `llm_confirmer.py`'s OpenAI-SDK-based client directly, the same
environment variable is read; without it, the module falls back to a mock
client so the rest of the pipeline still runs (useful for offline testing of
non-LLM logic).

## Usage

Run on a single process description:

```bash
python model_simple/complete_main.py "Your process description here" -o output.json
```

Run on the built-in sample process descriptions:

```bash
python model_simple/complete_main.py --batch
```

Disable API recommendation (BPMN structure only):

```bash
python model_simple/complete_main.py "..." --no-api
```

Flags:

- `request` (positional) — process description text. Falls back to a
  built-in sample if omitted.
- `-o, --output` — output JSON file name (default `complete_output.json`).
- `--no-debug` — disable verbose debug logging.
- `--batch` — run over the built-in sample process descriptions.
- `--no-api` — skip API recommendation, only generate the BPMN structure.

## Output format

`CompleteWorkflowSystem.process(...)` returns a dict containing (among other
things) a `complete_output` with `tasks`, `dependencies`, `gateways`, and per-task
`api_recommendation` (with confidence and Top-5 candidates), plus an
`evaluation_format` used by the evaluation harness.

## Reproducibility

All LLM calls in this pipeline (BPMN structure generation and API
re-ranking/confirmation) use `deepseek-chat` at **temperature = 0.3**,
unified across every call site, with up to 3 retries on failure. Prompt
templates and the output validation rules ship in this repo (see
`enhanced_bpmn_generator.py` and `llm_confirmer.py`) — nothing is
redacted for release.

For the full LLM configuration table, retry/backoff behavior, caching
caveats (important if you're doing repeated-run statistics), prompt/schema
locations, and the statistical evaluation protocol used in the paper
(per-run statistical unit, n=10 independent runs per scenario, Welch's
t-test with Holm-Bonferroni correction, Hedges' g effect sizes, 95% CIs),
see [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

Two scripts implement that protocol end-to-end: `run_repeated_evaluation.py`
runs n independent generations per scenario (a fresh system instance per
run, so the LLM-confirmer cache can't leak across runs) and writes raw
per-run metrics; then `statistical_analysis.py` turns those into the
t/df/p/Hedges' g/CI table, Holm-corrected within each scenario. See
REPRODUCIBILITY.md §7 for the exact commands.

## Notes for contributors

- `__pycache__/` and any local `data/` or `embeddings/` directories are
  git-ignored — please don't commit compiled bytecode or your private API
  catalog / embeddings.
- No credentials are stored in this repository. If you're rotating a key
  that was previously baked into a private build of this code, treat it as
  compromised and issue a new one from your provider's dashboard.
