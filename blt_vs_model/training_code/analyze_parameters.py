"""
Parameter Analysis Script for BLT-VS Models

Instantiates one or more BLT-VS model configurations and prints a detailed
breakdown of parameter counts per area, per connection type, and per
bottleneck module.

Usage Examples:
---------------
    # Single model with no bottlenecks (baseline)
    python analyze_parameters.py

    # Single model with specific bottlenecks
    python analyze_parameters.py --configs "V1->V2:16,V2->V3:16"

    # Compare multiple configurations side-by-side
    python analyze_parameters.py \
        --configs "none" "V1->V2:16" "V1->V2:16,V2->V3:16,V3->V4:16,V4->LOC:16,LGN->V1:16"

    # Load from an existing training run
    python analyze_parameters.py \
        --run_name "blt_vs_bottleneck__miniecoset__ts12__bnall16__20260402_123451"

    # Compare an existing run against new configs
    python analyze_parameters.py \
        --run_name "blt_vs_bottleneck__miniecoset__ts12__bnall16__20260402_123451" \
        --configs "none" "V1->V2:32,V2->V3:32"
"""

import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# ============================
# ARGUMENTS
# ============================

parser = argparse.ArgumentParser(
    description="Analyze parameter counts for BLT-VS models",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--configs",
    type=str,
    nargs="*",
    default=None,
    help=(
        'One or more bottleneck configs. Use "none" or "" for no bottlenecks. '
        'Example: "V1->V2:16,V2->V3:16"'
    ),
)
parser.add_argument(
    "--run_name",
    type=str,
    nargs="*",
    default=None,
    help="One or more existing run folder names (under logs/perf_logs/) to load configs from.",
)
parser.add_argument("--timesteps", type=int, default=12)
parser.add_argument("--num_classes", type=int, default=100)
parser.add_argument("--lateral_connections", type=int, default=1)
parser.add_argument("--topdown_connections", type=int, default=0)
parser.add_argument("--skip_connections", type=int, default=0)
parser.add_argument("--bio_unroll", type=int, default=1)
parser.add_argument("--readout_type", type=str, default="multi")
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument(
    "--no_save",
    action="store_true",
    help="Do NOT save the report to a text file (by default it is saved).",
)

args = parser.parse_args()

# ============================
# IMPORTS (model)
# ============================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.blt_vs_bottleneck_modular import BLT_VS_ModularBottlenecks


# ============================
# HELPERS
# ============================

def parse_bottlenecks(s: str):
    """Parse a bottleneck string like 'V1->V2:16,V2->V3:32' into a dict."""
    s = (s or "").strip()
    if s in ("", "none", "None"):
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        edge, ch = item.split(":")
        out[edge.strip()] = int(ch.strip())
    return out


def bottleneck_label(bn_dict):
    """Create a short human-readable label for a bottleneck config."""
    if not bn_dict:
        return "No Bottlenecks (Baseline)"
    parts = []
    for edge, ch in bn_dict.items():
        parts.append(f"{edge}:{ch}")
    return ", ".join(parts)



"""
most important function:
p = tensor with weights
numel() = number of values in tensor
requires_grad = wether it is trained
all of this comes form PyTorch
"""
def count_params(module):
    """Count total and trainable parameters of a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def count_params_no_recurse(module):
    """Count parameters that belong directly to this module (not sub-modules)."""
    total = sum(p.numel() for p in module.parameters(recurse=False))
    return total


def format_num(n):
    """Format a number with commas."""
    return f"{n:,}"


def format_pct(part, whole):
    if whole == 0:
        return "0.0%"
    return f"{100.0 * part / whole:.1f}%"


# ============================
# ANALYSIS FUNCTION
# ============================

def analyze_model(bn_dict, args, label=None):
    """Build a model with the given bottleneck config and return analysis dict."""

    if label is None:
        label = bottleneck_label(bn_dict)

    net = BLT_VS_ModularBottlenecks(
        timesteps=args.timesteps,
        num_classes=args.num_classes,
        lateral_connections=args.lateral_connections,
        topdown_connections=args.topdown_connections,
        skip_connections=args.skip_connections,
        bio_unroll=args.bio_unroll,
        image_size=args.image_size,
        readout_type=args.readout_type,
        bottlenecks=bn_dict,
    )

    total_params, trainable_params = count_params(net)

    areas = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC", "Readout"]
    area_details = OrderedDict()

    for area in areas:
        layer = net.connections[area]
        layer_total, layer_trainable = count_params(layer)

        detail = {
            "total": layer_total,
            "trainable": layer_trainable,
            "sublayers": OrderedDict(),
        }

        # Enumerate all named children and direct parameters
        for name, child in layer.named_children():
            child_total, _ = count_params(child)
            if child_total > 0:
                detail["sublayers"][name] = child_total

        # Check for direct parameters (e.g. bias)
        direct = count_params_no_recurse(layer)
        if direct > 0:
            detail["sublayers"]["(direct params)"] = direct

        area_details[area] = detail

    # Bottleneck modules
    bn_details = OrderedDict()
    bn_total = 0
    for edge_name, module in net.bottlenecks.items():
        edge_params, _ = count_params(module)
        bn_details[edge_name] = edge_params
        bn_total += edge_params

    # Readout weights (for single readout)
    readout_weight_params = 0
    if hasattr(net, "readout_weights"):
        readout_weight_params = net.readout_weights.numel()

    # Channel info
    channel_info = {area: ch for area, ch in zip(areas, net.channel_sizes)}

    return {
        "label": label,
        "bottlenecks": bn_dict,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "area_details": area_details,
        "bn_details": bn_details,
        "bn_total": bn_total,
        "readout_weight_params": readout_weight_params,
        "channel_sizes": channel_info,
    }


# ============================
# REPORT GENERATION
# ============================

def print_single_report(result, file=sys.stdout):
    """Print a detailed report for a single model config."""
    p = lambda *a, **kw: print(*a, **kw, file=file)

    p("=" * 72)
    p(f"  CONFIG: {result['label']}")
    p("=" * 72)
    p()

    # Channel sizes
    p("  Channel sizes per area:")
    for area, ch in result["channel_sizes"].items():
        p(f"    {area:10s} : {ch}")
    p()

    # Total
    p(f"  Total parameters:     {format_num(result['total_params'])}")
    p(f"  Trainable parameters: {format_num(result['trainable_params'])}")
    p()

    # Per-area breakdown
    p("  " + "-" * 68)
    p(f"  {'Area':<12s} {'Params':>12s} {'% of Total':>12s}   Sub-layer breakdown")
    p("  " + "-" * 68)

    for area, detail in result["area_details"].items():
        pct = format_pct(detail["total"], result["total_params"])
        sublayer_strs = []
        for sl_name, sl_params in detail["sublayers"].items():
            sublayer_strs.append(f"{sl_name}={format_num(sl_params)}")
        sl_str = ", ".join(sublayer_strs) if sublayer_strs else "-"

        p(f"  {area:<12s} {format_num(detail['total']):>12s} {pct:>12s}   {sl_str}")

    p("  " + "-" * 68)

    # Bottleneck modules
    if result["bn_details"]:
        p()
        p("  Bottleneck modules:")
        for edge, params in result["bn_details"].items():
            pct = format_pct(params, result["total_params"])
            p(f"    {edge:<16s} : {format_num(params):>10s}  ({pct})")
        p(f"    {'TOTAL':<16s} : {format_num(result['bn_total']):>10s}  ({format_pct(result['bn_total'], result['total_params'])})")
    else:
        p()
        p("  No bottleneck modules.")

    if result["readout_weight_params"] > 0:
        p(f"\n  Readout weights: {format_num(result['readout_weight_params'])}")

    p()


def print_comparison_table(results, file=sys.stdout):
    """Print a side-by-side comparison of multiple configs."""
    p = lambda *a, **kw: print(*a, **kw, file=file)

    if len(results) < 2:
        return

    p()
    p("=" * 72)
    p("  COMPARISON TABLE")
    p("=" * 72)
    p()

    # Short labels
    labels = []
    for i, r in enumerate(results):
        short = r["label"]
        if len(short) > 40:
            short = short[:37] + "..."
        labels.append(f"[{i+1}] {short}")

    # Header
    col_w = max(20, max(len(l) for l in labels) + 2)
    header = f"  {'':20s}" + "".join(f"{l:>{col_w}s}" for l in labels)
    p(header)
    p("  " + "-" * (20 + col_w * len(results)))

    # Total params
    row = f"  {'Total params':20s}"
    for r in results:
        row += f"{format_num(r['total_params']):>{col_w}s}"
    p(row)

    # Trainable
    row = f"  {'Trainable':20s}"
    for r in results:
        row += f"{format_num(r['trainable_params']):>{col_w}s}"
    p(row)

    # Delta vs first (baseline)
    baseline = results[0]["total_params"]
    row = f"  {'Delta vs [1]':20s}"
    for r in results:
        delta = r["total_params"] - baseline
        sign = "+" if delta >= 0 else ""
        row += f"{sign}{format_num(delta):>{col_w - 1}s}"
    p(row)

    row = f"  {'Delta %':20s}"
    for r in results:
        delta = r["total_params"] - baseline
        if baseline > 0:
            pct = 100.0 * delta / baseline
            sign = "+" if pct >= 0 else ""
            row += f"{sign}{pct:.1f}%".rjust(col_w)
        else:
            row += f"{'N/A':>{col_w}s}"
    p(row)

    p("  " + "-" * (20 + col_w * len(results)))

    # Per-area
    areas = list(results[0]["area_details"].keys())
    for area in areas:
        row = f"  {area:20s}"
        for r in results:
            params = r["area_details"][area]["total"]
            row += f"{format_num(params):>{col_w}s}"
        p(row)

    # Bottleneck total
    p("  " + "-" * (20 + col_w * len(results)))
    row = f"  {'Bottleneck total':20s}"
    for r in results:
        row += f"{format_num(r['bn_total']):>{col_w}s}"
    p(row)

    p()


# ============================
# MAIN
# ============================

if __name__ == "__main__":

    results = []

    # --- Load from existing runs ---
    if args.run_name:
        for rn in args.run_name:
            config_path = os.path.join("logs", "perf_logs", rn, "config.json")
            if not os.path.exists(config_path):
                print(f"WARNING: Config not found for run '{rn}': {config_path}", file=sys.stderr)
                continue

            with open(config_path, "r") as f:
                hyp = json.load(f)

            bn_dict = hyp.get("network", {}).get("bottlenecks", {})
            # Convert string keys to int values if needed
            bn_dict = {k: int(v) for k, v in bn_dict.items()}

            # Use run's actual settings
            run_args = argparse.Namespace(
                timesteps=hyp["network"]["timesteps"],
                num_classes=int(hyp["dataset"]["n_classes"]),
                lateral_connections=hyp["network"]["lateral_connections"],
                topdown_connections=hyp["network"]["topdown_connections"],
                skip_connections=hyp["network"]["skip_connections"],
                bio_unroll=hyp["network"]["bio_unroll"],
                readout_type=hyp["network"].get("readout_type", "multi"),
                image_size=224,
            )

            label = f"Run: {rn}"
            if bn_dict:
                label += f" ({bottleneck_label(bn_dict)})"
            else:
                label += " (no BN)"

            result = analyze_model(bn_dict, run_args, label=label)
            results.append(result)

    # --- From --configs ---
    if args.configs is not None:
        for cfg_str in args.configs:
            bn_dict = parse_bottlenecks(cfg_str)
            label = bottleneck_label(bn_dict)
            result = analyze_model(bn_dict, args, label=label)
            results.append(result)

    # --- Default: just baseline ---
    if not results:
        bn_dict = {}
        result = analyze_model(bn_dict, args, label="No Bottlenecks (Baseline)")
        results.append(result)

    # --- Output ---
    output_lines = []

    import io
    buf = io.StringIO()

    print("\n" + "=" * 72, file=buf)
    print("  BLT-VS PARAMETER ANALYSIS", file=buf)
    print(f"  Timesteps={args.timesteps}, Classes={args.num_classes}, "
          f"Lateral={args.lateral_connections}, TD={args.topdown_connections}, "
          f"Skip={args.skip_connections}, BioUnroll={args.bio_unroll}", file=buf)
    print("=" * 72, file=buf)

    for result in results:
        print_single_report(result, file=buf)

    if len(results) >= 2:
        print_comparison_table(results, file=buf)

    report = buf.getvalue()
    print(report)

    if not args.no_save:
        from datetime import datetime
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter_analysis")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"param_analysis_{timestamp}.txt")
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Report saved to {out_path}")
