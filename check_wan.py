#!/usr/bin/env python3
"""Competition-oriented checker for Wan 2.2 W4A4 artifacts.

This script validates:
1) Constraint compliance (quant type + high-precision layer cap)
2) Proxy quality/speed targets for fast iteration
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Wan W4A4 quality/compliance")
    p.add_argument("--profile", choices=["practical", "strict"], default="practical",
                   help="Threshold preset: practical for iterative tuning, strict for stretch target")
    p.add_argument("--output-dir", default="./quantized_wan_output",
                   help="Directory containing quantization_metadata.json and vbench_results")
    p.add_argument("--metadata", default=None,
                   help="Optional explicit metadata JSON path")
    p.add_argument("--results", default=None,
                   help="Optional explicit vbench_results.json path")
    p.add_argument("--speedup-min", type=float, default=None)
    p.add_argument("--cosine-min", type=float, default=None)
    p.add_argument("--relative-l2-max", type=float, default=None)
    p.add_argument("--proxy-total-min", type=float, default=None)
    args = p.parse_args()

    if args.profile == "strict":
        defaults = {
            "speedup_min": 3.0,
            "cosine_min": 0.90,
            "relative_l2_max": 0.45,
            "proxy_total_min": 0.65,
        }
    else:
        defaults = {
            "speedup_min": 0.90,
            "cosine_min": 0.89,
            "relative_l2_max": 0.50,
            "proxy_total_min": 0.62,
        }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def evaluate_compliance(metadata: Dict) -> Tuple[bool, Dict[str, str]]:
    verdicts: Dict[str, str] = {}
    ok = True

    quant_type = str(metadata.get("quant_type", "")).lower()
    max_hp = int(metadata.get("max_high_precision_layers", -1))

    if quant_type not in {"hifx4", "mxfp4"}:
        ok = False
        verdicts["quant_type"] = f"FAIL: unsupported quant_type={quant_type}"
    else:
        verdicts["quant_type"] = f"PASS: quant_type={quant_type}"

    hp_limit = 2 if quant_type == "hifx4" else 5
    if max_hp < 0:
        ok = False
        verdicts["high_precision_layers"] = "FAIL: missing max_high_precision_layers"
    elif max_hp > hp_limit:
        ok = False
        verdicts["high_precision_layers"] = (
            f"FAIL: max_high_precision_layers={max_hp} exceeds {quant_type} limit {hp_limit}"
        )
    else:
        verdicts["high_precision_layers"] = (
            f"PASS: max_high_precision_layers={max_hp} within {quant_type} limit {hp_limit}"
        )

    return ok, verdicts


def evaluate_proxy_performance(results: Dict, args: argparse.Namespace) -> Tuple[bool, Dict[str, str]]:
    verdicts: Dict[str, str] = {}
    ok = True

    proxy = results.get("transformer_proxy_metrics", {})
    metrics = results.get("metrics", {})

    speedup = float(proxy.get("speedup", 0.0))
    cosine = float(proxy.get("transformer_cosine_similarity", 0.0))
    rel_l2 = float(proxy.get("transformer_relative_l2", 1e9))
    total = float(metrics.get("total", 0.0))

    if speedup >= args.speedup_min:
        verdicts["speedup"] = f"PASS: speedup={speedup:.4f} >= {args.speedup_min:.4f}"
    else:
        ok = False
        verdicts["speedup"] = f"FAIL: speedup={speedup:.4f} < {args.speedup_min:.4f}"

    if cosine >= args.cosine_min:
        verdicts["cosine"] = f"PASS: cosine={cosine:.4f} >= {args.cosine_min:.4f}"
    else:
        ok = False
        verdicts["cosine"] = f"FAIL: cosine={cosine:.4f} < {args.cosine_min:.4f}"

    if rel_l2 <= args.relative_l2_max:
        verdicts["relative_l2"] = f"PASS: relative_l2={rel_l2:.4f} <= {args.relative_l2_max:.4f}"
    else:
        ok = False
        verdicts["relative_l2"] = f"FAIL: relative_l2={rel_l2:.4f} > {args.relative_l2_max:.4f}"

    if total >= args.proxy_total_min:
        verdicts["proxy_total"] = f"PASS: total={total:.4f} >= {args.proxy_total_min:.4f}"
    else:
        ok = False
        verdicts["proxy_total"] = f"FAIL: total={total:.4f} < {args.proxy_total_min:.4f}"

    return ok, verdicts


def main() -> int:
    args = parse_args()

    metadata_path = args.metadata or os.path.join(args.output_dir, "quantization_metadata.json")
    results_path = args.results or os.path.join(args.output_dir, "vbench_results", "vbench_results.json")

    if not os.path.isfile(metadata_path):
        print(f"[FATAL] Metadata not found: {metadata_path}")
        return 2
    if not os.path.isfile(results_path):
        print(f"[FATAL] Results not found: {results_path}")
        return 2

    metadata = load_json(metadata_path)
    results = load_json(results_path)

    comp_ok, comp_detail = evaluate_compliance(metadata)
    perf_ok, perf_detail = evaluate_proxy_performance(results, args)

    print("=" * 68)
    print("Wan 2.2 W4A4 Check Report")
    print("=" * 68)
    print("[Compliance]")
    for key in ("quant_type", "high_precision_layers"):
        print(f"  - {comp_detail.get(key, 'N/A')}")

    print("\n[Proxy Performance]")
    print(f"  - Profile: {args.profile}")
    for key in ("speedup", "cosine", "relative_l2", "proxy_total"):
        print(f"  - {perf_detail.get(key, 'N/A')}")

    print("\n[Overall]")
    if comp_ok and perf_ok:
        print("  PASS: compliant and performance target reached.")
    elif comp_ok:
        print("  PARTIAL: compliant but performance target not fully reached.")
        print("  Suggestion: rerun with sensitivity enabled (rotation off), then tune kept high-precision layers.")
        print("  Optional: after baseline improves, A/B test rotation separately.")
    else:
        print("  FAIL: compliance issue found.")

    return 0 if (comp_ok and perf_ok) else 1


if __name__ == "__main__":
    sys.exit(main())