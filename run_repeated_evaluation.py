import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "model_simple"))
sys.path.insert(0, str(SCRIPT_DIR / "common"))
sys.path.insert(0, str(PROJECT_ROOT))  # for `tools.extract_ground_truth`

from scenarios import BPMN_TEST_SCENARIOS
from metrics import structural_score, api_topk_hit_rate


def score_run(complete_output: dict, scenario) -> dict:

    from tools.extract_ground_truth import extract_bpmn_structure

    gt_path = PROJECT_ROOT / "ground_truth" / scenario.ground_truth_file
    ground_truth = extract_bpmn_structure(str(gt_path))

    generated = {
        "tasks": complete_output.get("tasks", []),
        "gateways": complete_output.get("gateways", []),
        "dependencies": complete_output.get("dependencies", []),
    }
    structural = structural_score(generated, ground_truth)

    api_bindings = complete_output.get("api_bindings", {})
    gold_tasks = scenario.load_gold_api_bindings()
    api_scores = api_topk_hit_rate(api_bindings, gold_tasks)

    return {
        "structural_f1": structural["structural_f1"],
        "node_f1": structural["node_f1"],
        "edge_f1": structural["edge_f1"],
        "gateway_type_accuracy": structural["gateway_type_accuracy"],
        "api_top1_hit_rate": api_scores["top1_hit_rate"],
        "api_top_k_hit_rate": api_scores["top_k_hit_rate"],
        # kept for backward-compat / sanity-checking, not used in the paper's tests
        "task_count": float(len(generated["tasks"])),
        "dependency_count": float(len(generated["dependencies"])),
        "gateway_count": float(len(generated["gateways"])),
    }


def run_once(scenario, api_key: str, use_api_recommendation: bool):

    from complete_main import CompleteWorkflowSystem, Config


    Config.ENABLE_API_RECOMMENDATION = use_api_recommendation

    system = CompleteWorkflowSystem(api_key=api_key, debug=False)
    result = system.process(scenario.requirement)

    complete_output = result.get("complete_output", result)
    metrics = score_run(complete_output, scenario)

    retry_stats = {}
    bpmn_gen = getattr(getattr(system, "generator", None), "bpmn_generator", None)
    if bpmn_gen is not None and hasattr(bpmn_gen, "get_statistics"):
        retry_stats["bpmn_generator"] = bpmn_gen.get_statistics()

    return metrics, retry_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["LEWR"],
                     help="Method name(s) to label these runs with in the output JSON "
                          "(this harness only drives the model in this repo; runs for "
                          "baseline methods like AutoFlow-MTFM/MAO/BPMN-MLP must come "
                          "from their own repos/scripts and be merged into the same "
                          "output file before running statistical_analysis.py).")
    ap.add_argument("--n-runs", type=int, default=10, help="Independent runs per scenario per method (default: 10)")
    ap.add_argument("--no-api", action="store_true", help="Skip API recommendation, BPMN structure only")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "results" / "raw_runs.json"))
    ap.add_argument("--api-key", default=None, help="Defaults to DEEPSEEK_API_KEY env var")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    results: dict = {}

    retry_log: dict = {}

    for scenario in BPMN_TEST_SCENARIOS:
        results[scenario.name] = {}
        retry_log[scenario.name] = {}

        for method in args.methods:
            per_metric_values: dict = {}
            per_run_retry_stats = []

            for run_idx in range(args.n_runs):
                print(f"[{scenario.name}] {method} run {run_idx + 1}/{args.n_runs}")
                try:
                    metrics, retry_stats = run_once(scenario, args.api_key, use_api_recommendation=not args.no_api)
                except Exception as e:
                    print(f"  ✗ run failed: {e}", file=sys.stderr)
                    traceback.print_exc()
                    continue

                for k, v in metrics.items():
                    per_metric_values.setdefault(k, []).append(v)
                per_run_retry_stats.append(retry_stats)
                time.sleep(1)

            results[scenario.name][method] = per_metric_values
            retry_log[scenario.name][method] = per_run_retry_stats

            for metric_name, values in per_metric_values.items():
                if len(values) >= 2:
                    print(f"  {metric_name}: mean={statistics.mean(values):.4f} "
                          f"sd={statistics.stdev(values):.4f} (n={len(values)})")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw per-run results written to {out_path}")

    retry_path = out_path.with_name(out_path.stem + "_retry_stats.json")
    with open(retry_path, "w", encoding="utf-8") as f:
        json.dump(retry_log, f, indent=2, ensure_ascii=False)
    print(f"Per-run retry statistics written to {retry_path}")
    print(f"\nNext: python statistical_analysis.py {out_path} --baseline {args.methods[0]} --metric <metric_name>")


if __name__ == "__main__":
    main()
