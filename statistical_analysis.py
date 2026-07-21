import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

try:
    from scipy import stats as scipy_stats
except ImportError:
    print("This script requires scipy. Install with: pip install scipy", file=sys.stderr)
    raise


def welch_ttest(a: List[float], b: List[float]):

    a = list(a)
    b = list(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Need at least 2 runs per group for a t-test, got n1={n1}, n2={n2}")

    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)

    se2 = v1 / n1 + v2 / n2
    se = math.sqrt(se2)
    if se == 0:

        return 0.0, n1 + n2 - 2, 1.0

    t = (m1 - m2) / se


    num = se2 ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else (n1 + n2 - 2)

    p = 2 * scipy_stats.t.sf(abs(t), df)
    return t, df, float(p)


def hedges_g(a: List[float], b: List[float]) -> float:

    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)

    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    d = (m1 - m2) / pooled_sd


    df = n1 + n2 - 2
    j = 1 - 3 / (4 * df - 1)
    return d * j


def mean_diff_ci(a: List[float], b: List[float], df: float, confidence: float = 0.95):

    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)

    alpha = 1 - confidence
    tcrit = float(scipy_stats.t.ppf(1 - alpha / 2, df))
    diff = m1 - m2
    return diff, (diff - tcrit * se, diff + tcrit * se)


def holm_bonferroni(pvalues: List[float]) -> List[float]:

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def analyze(data: Dict, baseline: str, metric: str) -> List[Dict]:

    rows = []

    for scenario, methods in data.items():
        if baseline not in methods:
            print(f"[skip] scenario '{scenario}': baseline '{baseline}' not present", file=sys.stderr)
            continue
        baseline_vals = methods[baseline].get(metric)
        if baseline_vals is None:
            print(f"[skip] scenario '{scenario}': metric '{metric}' missing for baseline", file=sys.stderr)
            continue

        comparisons = []
        for other, other_metrics in methods.items():
            if other == baseline:
                continue
            other_vals = other_metrics.get(metric)
            if other_vals is None:
                print(f"[skip] scenario '{scenario}': metric '{metric}' missing for '{other}'", file=sys.stderr)
                continue

            t, df, p_raw = welch_ttest(baseline_vals, other_vals)
            g = hedges_g(baseline_vals, other_vals)
            diff, ci = mean_diff_ci(baseline_vals, other_vals, df)

            comparisons.append({
                "scenario": scenario,
                "metric": metric,
                "baseline": baseline,
                "comparison": other,
                "n_baseline": len(baseline_vals),
                "n_comparison": len(other_vals),
                "mean_baseline": sum(baseline_vals) / len(baseline_vals),
                "mean_comparison": sum(other_vals) / len(other_vals),
                "mean_diff": diff,
                "t_statistic": t,
                "df_welch_satterthwaite": df,
                "p_raw": p_raw,
                "hedges_g": g,
                "ci95_low": ci[0],
                "ci95_high": ci[1],
            })


        if comparisons:
            raw_ps = [c["p_raw"] for c in comparisons]
            corrected = holm_bonferroni(raw_ps)
            for c, p_holm in zip(comparisons, corrected):
                c["p_holm"] = float(p_holm)
                c["significant_at_0.05"] = bool(p_holm < 0.05)
            rows.extend(comparisons)

    return rows


def print_summary(rows: List[Dict]):
    print("\n" + "=" * 100)
    print("Statistical summary (Welch's t-test, Holm-Bonferroni corrected within scenario)")
    print("=" * 100)
    header = f"{'Scenario':<15}{'Comparison':<28}{'t':>8}{'df':>8}{'p_holm':>10}{'g':>8}{'95% CI':>22}  sig"
    print(header)
    print("-" * len(header))
    for r in rows:
        comp = f"{r['baseline']} vs {r['comparison']}"
        ci = f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
        sig = "*" if r["significant_at_0.05"] else ""
        print(f"{r['scenario']:<15}{comp:<28}{r['t_statistic']:>8.2f}{r['df_welch_satterthwaite']:>8.1f}"
              f"{r['p_holm']:>10.4f}{r['hedges_g']:>8.2f}{ci:>22}  {sig}")
    print("=" * 100)
    print("* = significant at Holm-corrected alpha=0.05\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Path to raw per-run results JSON (scenario -> method -> metric -> [values])")
    ap.add_argument("--baseline", required=True, help="Method name to compare all others against, e.g. LEWR")
    ap.add_argument("--metric", required=True, help="Metric name to test, e.g. structural_f1")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <input>_stats.json)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = analyze(data, args.baseline, args.metric)
    if not rows:
        print("No comparisons could be computed — check method/metric names in the input file.", file=sys.stderr)
        sys.exit(1)

    print_summary(rows)

    out_path = Path(args.out) if args.out else Path(args.input).with_suffix("").with_name(
        Path(args.input).stem + "_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Full report written to {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV report written to {csv_path}")


if __name__ == "__main__":
    main()
