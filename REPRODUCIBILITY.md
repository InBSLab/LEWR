# Reproducibility Notes

This document collects the details a reviewer or re-implementer needs to
reproduce the numbers reported in the paper: LLM configuration, prompts,
output schema/validation, retry/caching behavior, and the statistical
evaluation protocol.

## 1. LLM configuration

All LLM calls in this pipeline go through the DeepSeek chat completion API.

| Component | File | Model | Temperature | Max tokens | Timeout |
|---|---|---|---|---|---|
| BPMN structure generation | `model_simple/enhanced_bpmn_generator.py` | `deepseek-chat` | 0.3 | 4000 | 90s |
| API candidate re-ranking / confirmation | `model_simple/llm_confirmer.py` (`LLMConfirmerConfig`) | `deepseek-chat` | 0.3 | 2500 | — |
| Evaluation harness fallback client | `evaluation/run_evaluation.py` (`LLMClient`) | `deepseek-chat` | 0.3 | 2500 | 180s |

Temperature is unified to **0.3** across all three call sites as of this
release.

No decoding parameters (top_p, frequency/presence penalty) are set
explicitly, so they use the DeepSeek API defaults at call time. We do not
pin a dated API snapshot; `deepseek-chat` is the standard/default DeepSeek
chat model alias, and its underlying weights may be updated by the
provider over time. Runs reported in the paper were collected in the
window stated in Section 5 of the paper.

## 2. Retry policy

Retry behavior differs by call site:

- **BPMN structure generation** (`enhanced_bpmn_generator.py`) and the
  **evaluation harness fallback client** (`run_evaluation.py`,
  `LLMClient`) retry up to `max_retries = 3` times on: HTTP status != 200,
  request timeout, connection error, or (structure generator only) JSON
  parse/schema failure of the model output. Backoff is a fixed sleep
  (5–15s depending on error type) — not exponential.
- **API re-ranking / confirmation** (`llm_confirmer.py`,
  `DeepSeekClient.generate`) does **not** retry. Each call is attempted
  once; on any exception it falls back to a safe default instead
  (`confirm()` falls back to `True`/accept, `confirm_multi()` falls back
  to the top-1 candidate — see the `except` blocks around lines ~537 and
  ~751 of `llm_confirmer.py`). This is a single-attempt-with-safe-fallback
  policy, not a retry policy.

If all retries are exhausted (structure generator / harness client), the
call site returns an empty/failure result rather than raising, and the
caller degrades to the next fallback stage (see §4 in the pipeline
overview / README).

**Retry/fallback statistics are now tracked, not just printed to stdout:**
- `EnhancedBPMNGenerator.get_statistics()` returns, among other fields,
  `total_llm_attempts` (sum of attempts across all requests, including
  retries) and `requests_needing_retry` (requests that only succeeded on
  attempt > 1).
- `LLMConfirmer.stats` includes `llm_call_failures` (single-attempt calls
  that raised and used the safe fallback).
- `run_evaluation.py`'s `LLMClient.get_retry_stats()` returns
  `total_calls`, `total_attempts`, `calls_needing_retry`, and
  `calls_exhausted`.

`evaluation/run_repeated_evaluation.py` collects the BPMN generator's
stats per run and writes them to a `*_retry_stats.json` file alongside the
raw metrics (see §7).

## 3. Caching

`llm_confirmer.py` enables an in-process LRU-style cache
(`LLMConfirmerConfig.ENABLE_CACHE = True`, `CACHE_SIZE = 500`) keyed on
`(task_spec, candidate API)`. A cache hit skips the LLM call entirely and
returns the previously observed confirmation/ranking decision.

**This matters for repeated-run statistics**: if you re-run the same
scenario in the same process without clearing `self._cache` /
`self._cache_multi` (e.g. instantiate a fresh `LLMConfirmer` per run, or
call `ENABLE_CACHE = False`), later runs will reuse earlier decisions for
identical task/API pairs and will not reflect independent LLM samples.
`enhanced_bpmn_generator.py` and the evaluation harness's `LLMClient` do
**not** cache — each call is a fresh API request.

## 4. Prompts, schema, and validation

- **BPMN structure generation prompt**: built by
  `EnhancedBPMNGenerator._build_enhanced_prompt()` in
  `enhanced_bpmn_generator.py`. This is the exact system
  prompt used at inference time — nothing is redacted or paraphrased for
  release.
- **API re-ranking / confirmation prompts**: built in
  `llm_confirmer.py` (see the prompt-construction methods
  feeding `DeepSeekClient.chat`).
- **Output schema and validation rules**: `_validate_bpmn()` in
  `enhanced_bpmn_generator.py` defines the structural checks applied to
  every generated workflow (required fields, task/gateway/edge
  consistency) before a result is accepted; a failed validation triggers
  a retry (see §2).
- **Representative outputs**: see `sample_outputs/` (add your
  scenario transcripts there — raw JSON in, raw JSON out, one file per
  scenario/run) so a reader can inspect real generations rather than only
  the aggregated metrics.

## 5. Statistical evaluation protocol (Section 5.4–5.5.1 of the paper)

- **Statistical unit**: the individual end-to-end generation run — not
  the scenario and not individual tasks within a workflow. For each of
  the 4 BPMN scenarios and each method, we perform **n = 10 independent
  end-to-end runs**.
- **Metric aggregation**: structural metrics are computed as a single
  score per full generated workflow (one number per run), so no test
  treats tasks within a workflow as independent samples.
- **Test**: per-scenario **Welch's t-test** (unequal variances assumed),
  with **Holm-Bonferroni correction** applied across the comparisons
  within a scenario.
- **Reported statistics per comparison**: t-statistic, Welch-Satterthwaite
  degrees of freedom, p-value (Holm-corrected), **Hedges' g** effect size,
  and 95% confidence interval.
- **Headline results** (Section 5.5.1): LEWR significantly outperforms
  AutoFlow-MTFM and MAO in every scenario (Hedges' g = 1.6–47.2,
  p < 0.01, Holm-corrected). Against BPMN-MLP, the difference is
  significant only in the Recourse scenario (p = 0.043); the remaining
  three scenarios show no significant difference.

If you re-run the evaluation yourself, use a **fresh LLM-confirmer
instance per run** (or disable caching — see §3) so that the 10 runs per
scenario are independent samples, not partially cached repeats.

## 6. What's included vs. what you must supply

This repository includes the full generation pipeline, prompts, and
validation logic (§4), the statistical-analysis and repeated-run scripts
described in §7, and (as of this release):

- **`scenarios.py`** — the real `BPMN_TEST_SCENARIOS`
  (4 scenarios: Dispatch of Goods, Recourse, Credit Scoring,
  Self-Service Restaurant), each with its verbatim process description
  and a reference to its ground-truth `.bpmn` solution file under
  `ground_truth/`, plus the `UnifiedOutput` result container and
  `save_results` helper used by `run_evaluation.py`.
- **`metrics.py`** — real structural-quality and
  API-binding metrics: a Hungarian-matching-based structural F1 (node
  F1 + edge F1 + gateway-type agreement) against the ground-truth BPMN,
  and a top-1/top-k API hit rate against `ground_truth/api_gold.json`
  (human-annotated gold API bindings per task). This replaces the
  previous element-count-only placeholder in `score_run()`.
- **`ground_truth/`** — the 4 scenarios' exercise text, instructor
  solution `.bpmn` files, and human-annotated gold API bindings, plus
  `tools/extract_ground_truth.py`, the (auditable) script used to parse
  the `.bpmn` files into the flat structure the metrics operate on.
- **`merge_baseline_results.py`** — merges this repo's raw
  per-run results with baseline methods' raw per-run results into the
  single `scenario -> method -> metric -> [values]` JSON shape
  `statistical_analysis.py` expects, with explicit (never silent)
  scenario/method/metric renaming and a validation pass that flags
  missing methods or under-powered (`n < 2`) cells before you run the
  significance tests.

It still does **not** include:
- your private API catalog / embeddings  — see main README "Setup",
- raw per-run results for baseline methods (AutoFlow-MTFM, MAO,
  BPMN-MLP) themselves — those must be produced by their own
  repos/scripts (this repo only drives its own model); use
  `merge_baseline_results.py` to combine them with this repo's output
  once you have them,

## 7. Running the statistical evaluation end-to-end

```bash
# 1. Generate n=10 independent runs per scenario for this model
#    (fresh CompleteWorkflowSystem instance per run -> no cache leakage
#    across runs, see §3). Uses the real BPMN_TEST_SCENARIOS and
#    structural_f1 / api_top1_hit_rate / api_top_k_hit_rate metrics from
#    evaluation/common/{scenarios,metrics}.py against ground_truth/.
python evaluation/run_repeated_evaluation.py \
    --methods LEWR --n-runs 10 \
    --out evaluation/results/raw_runs_LEWR.json

# 2. Produce each baseline's (AutoFlow-MTFM / MAO / BPMN-MLP) raw per-run
#    results from their own repos/scripts, in the same
#    scenario -> method -> metric -> [values] shape (see
#    evaluation/merge_baseline_results.py's docstring for the exact
#    expected input shape), then merge them all into one file. The merge
#    step never silently guesses a scenario/method/metric name mapping —
#    use --rename-* explicitly if a baseline's naming differs from this
#    repo's, and review the validation warnings it prints (missing
#    methods per scenario, any cell with n < 2) before proceeding.
python evaluation/merge_baseline_results.py \
    --base evaluation/results/raw_runs_LEWR.json \
    --baseline evaluation/results/raw_runs_autoflow_mtfm.json \
    --baseline evaluation/results/raw_runs_mao.json \
    --baseline evaluation/results/raw_runs_bpmn_mlp.json \
    --out evaluation/results/raw_runs_merged.json

# 3. Run the Welch's t-test / Holm-Bonferroni / Hedges' g / 95% CI
#    analysis described in §5.
python evaluation/statistical_analysis.py evaluation/results/raw_runs_merged.json \
    --baseline LEWR --metric structural_f1 \
    --out evaluation/results/statistical_report.json
```

`statistical_analysis.py` has been cross-checked against `scipy.stats.
ttest_ind(equal_var=False)` for the t-statistic, Welch-Satterthwaite
degrees of freedom, and p-value (they match to floating-point precision);
Hedges' g and the Holm-Bonferroni step-down correction are implemented
directly in the script (no extra dependency beyond `scipy`, already added
to `requirements.txt`).
