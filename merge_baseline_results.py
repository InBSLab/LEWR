import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List


def _parse_rename(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--rename must be in the form OLD=NEW, got: {p!r}")
        old, new = p.split("=", 1)
        out[old] = new
    return out


def _apply_renames(data: Dict, scenario_map: Dict[str, str],
                    method_map: Dict[str, str], metric_map: Dict[str, str]) -> Dict:
    renamed = {}
    for scenario, methods in data.items():
        s_name = scenario_map.get(scenario, scenario)
        renamed.setdefault(s_name, {})
        for method, metrics in methods.items():
            m_name = method_map.get(method, method)
            renamed[s_name].setdefault(m_name, {})
            for metric, values in metrics.items():
                met_name = metric_map.get(metric, metric)
                if met_name in renamed[s_name][m_name]:
                    print(f"[warn] duplicate metric '{met_name}' for scenario="
                          f"'{s_name}' method='{m_name}' after renaming — "
                          f"overwriting with the later file's values", file=sys.stderr)
                renamed[s_name][m_name][met_name] = values
    return renamed


def merge(base: Dict, additions: List[Dict]) -> Dict:

    merged = copy.deepcopy(base)
    for addition in additions:
        for scenario, methods in addition.items():
            merged.setdefault(scenario, {})
            for method, metrics in methods.items():
                if method in merged[scenario]:
                    raise ValueError(
                        f"Method '{method}' already present for scenario '{scenario}' "
                        f"in an earlier file — refusing to silently overwrite. "
                        f"If this method legitimately appears in multiple input "
                        f"files (e.g. different metric subsets), merge them "
                        f"upstream before calling this script, or rename one of "
                        f"the two with --rename-method.")
                merged[scenario][method] = metrics
    return merged


def validate(data: Dict, min_n: int = 2) -> List[str]:

    warnings = []
    scenario_names = list(data.keys())
    method_sets = {s: set(data[s].keys()) for s in scenario_names}

    all_methods = set()
    for s in method_sets.values():
        all_methods |= s
    for s, methods in method_sets.items():
        missing = all_methods - methods
        if missing:
            warnings.append(f"Scenario '{s}' is missing method(s) {sorted(missing)} "
                             f"present in other scenarios.")

    for scenario, methods in data.items():
        for method, metrics in methods.items():
            for metric, values in metrics.items():
                if len(values) < min_n:
                    warnings.append(f"scenario='{scenario}' method='{method}' "
                                     f"metric='{metric}': only n={len(values)} "
                                     f"values (need >= {min_n} for a t-test).")
    return warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="This repo's raw_runs.json (e.g. from run_repeated_evaluation.py)")
    ap.add_argument("--baseline", action="append", default=[], dest="baselines",
                     help="A baseline method's raw results JSON, same shape. Repeat for multiple baselines.")
    ap.add_argument("--rename-scenario", action="append", default=[],
                     help="OLD=NEW scenario name mapping, applied to ALL --baseline files. Repeatable.")
    ap.add_argument("--rename-method", action="append", default=[],
                     help="OLD=NEW method name mapping, applied to ALL --baseline files. Repeatable.")
    ap.add_argument("--rename-metric", action="append", default=[],
                     help="OLD=NEW metric name mapping, applied to ALL --baseline files. Repeatable.")
    ap.add_argument("--out", required=True, help="Output merged JSON path")
    ap.add_argument("--min-n", type=int, default=2, help="Minimum runs per cell to warn below (default 2)")
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = json.load(f)

    scenario_map = _parse_rename(args.rename_scenario)
    method_map = _parse_rename(args.rename_method)
    metric_map = _parse_rename(args.rename_metric)

    additions = []
    for path in args.baselines:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        additions.append(_apply_renames(raw, scenario_map, method_map, metric_map))

    merged = merge(base, additions)

    warnings = validate(merged, min_n=args.min_n)
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged {1 + len(additions)} file(s) -> {out_path}")
    print(f"Scenarios: {list(merged.keys())}")
    for s, methods in merged.items():
        print(f"  {s}: methods={list(methods.keys())}")

    if warnings:
        print(f"\n{len(warnings)} warning(s) — review before running statistical_analysis.py.", file=sys.stderr)


if __name__ == "__main__":
    main()
