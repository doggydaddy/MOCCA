"""Assemble the covariate sensitivity series into one comparison table.

Each member is a completed `run_bundle_fwer.py` output directory differing
only in its design matrix. Reading them side by side is what makes the
adjustment decision visible: the conventional age+sex model is the only one of
the four that finds nothing, and no single run shows that on its own.

The `coherence` column -- the top bundle's share of all retained suprathreshold
edges -- is included because it distinguishes two very different ways a model
can lose significance: by thresholding fewer edges, or by keeping the same
edges while destroying their spatial contiguity so they no longer assemble
into a bundle. In this dataset the joint model does the latter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def summarise(name: str, run_dir: Path, alpha: float) -> dict[str, object]:
    maxima = pd.read_csv(run_dir / "permutation_bundle_maxima.csv")
    observed_bundles = pd.read_csv(run_dir / "observed_bundles_fwer.csv")
    if not len(observed_bundles):
        raise ValueError(f"{run_dir} reports no observed bundles")
    observed = maxima.loc[maxima["observed"] == True].iloc[0]  # noqa: E712
    null = maxima.loc[maxima["observed"] != True]              # noqa: E712
    top = observed_bundles.sort_values("p_fwer").iloc[0]
    null_q95 = float(null["max_statistic"].quantile(0.95))
    config_path = run_dir / "bundle_fwer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "model": name,
        "edge_statistic": config.get("edge_statistic"),
        "significant": int((observed_bundles["p_fwer"] <= alpha).sum()),
        "top_p_fwer": float(top["p_fwer"]),
        "suprathreshold_edges": int(observed["threshold_edges"]),
        "retained_edges": int(observed["retained_edges"]),
        "bundles": int(observed["bundles"]),
        "top_bundle_edges": int(top["edge_count"]),
        "top_bundle_mass": float(top["mass"]),
        "null_q95": null_q95,
        "observed_over_null_q95": float(top["mass"]) / null_q95,
        "coherence": int(top["edge_count"]) / int(observed["retained_edges"]),
        "omnibus_p_suprathreshold_edges": float(
            (1 + (null["threshold_edges"] >= observed["threshold_edges"]).sum())
            / (len(null) + 1)
        ),
        "omnibus_p_bundles": float(
            (1 + (null["bundles"] >= observed["bundles"]).sum()) / (len(null) + 1)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs", nargs="+", metavar="NAME=DIR",
        help="labelled run directories, e.g. 'group + age=/path/to/inference'",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    rows = []
    for entry in args.runs:
        if "=" not in entry:
            raise ValueError(f"expected NAME=DIR, got {entry!r}")
        name, _, directory = entry.partition("=")
        rows.append(summarise(name, Path(directory).resolve(), args.alpha))
    table = pd.DataFrame(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    pd.set_option("display.width", 250)
    print(table[[
        "model", "significant", "top_p_fwer", "suprathreshold_edges",
        "retained_edges", "top_bundle_edges", "top_bundle_mass", "null_q95",
        "observed_over_null_q95", "coherence",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\ncoherence = top bundle's share of all retained suprathreshold edges")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
