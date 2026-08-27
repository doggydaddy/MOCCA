#!/usr/bin/env python3
"""Create directly loadable COFFEE-DAC GUI caches from bundle-grid FWER output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


EDGE_COLUMNS = [
    "i1", "j1", "k1", "i2", "j2", "k2", "pvalue", "tstat",
    "bundle", "network",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_slug(value: float) -> str:
    return f"p_{value:.10g}".replace(".", "p")


def write_selected_edges(
    source: Path,
    raw_output: Path,
    processed_output: Path,
    selected: pd.DataFrame,
    chunk_size: int,
) -> int:
    """Stream selected bundles into raw and already-processed GUI CSVs."""

    selected = selected.sort_values(
        ["sign", "edge_count", "bundle"], ascending=[False, False, True]
    ).reset_index(drop=True)
    bundle_map = {
        int(row.bundle): compact for compact, row in selected.iterrows()
    }
    p_map = {
        int(row.bundle): float(row.p_grid_fwer) for _, row in selected.iterrows()
    }
    present_signs = sorted(selected["sign"].astype(int).unique(), reverse=True)
    network_map = {sign: network for network, sign in enumerate(present_signs)}
    sign_map = {
        int(row.bundle): int(row.sign) for _, row in selected.iterrows()
    }

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_temp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=raw_output.parent, delete=False
    )
    processed_temp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=processed_output.parent, delete=False
    )
    raw_temp_path = Path(raw_temp.name)
    processed_temp_path = Path(processed_temp.name)
    raw_temp.close()
    processed_temp.close()

    rows_written = 0
    first = True
    try:
        for chunk in pd.read_csv(source, chunksize=chunk_size):
            chunk = chunk[chunk["bundle"].isin(bundle_map)].copy()
            if chunk.empty:
                continue
            source_bundle = chunk["bundle"].astype(int)
            chunk["pvalue"] = source_bundle.map(p_map).astype(float)
            chunk["bundle"] = source_bundle.map(bundle_map).astype(int)
            chunk["network"] = source_bundle.map(
                lambda value: network_map[sign_map[int(value)]]
            ).astype(int)
            chunk = chunk[EDGE_COLUMNS]
            chunk.iloc[:, :6] = chunk.iloc[:, :6].astype(int)
            chunk[EDGE_COLUMNS[:8]].to_csv(
                raw_temp_path, mode="a", header=first, index=False
            )
            chunk.to_csv(
                processed_temp_path, mode="a", header=first, index=False
            )
            rows_written += len(chunk)
            first = False
        if first:
            pd.DataFrame(columns=EDGE_COLUMNS[:8]).to_csv(raw_temp_path, index=False)
            pd.DataFrame(columns=EDGE_COLUMNS).to_csv(processed_temp_path, index=False)
        raw_temp_path.replace(raw_output)
        processed_temp_path.replace(processed_output)
    finally:
        raw_temp_path.unlink(missing_ok=True)
        processed_temp_path.unlink(missing_ok=True)
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_results", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--top-bundles-per-threshold", type=int, default=0,
        help=("also export the N smallest-p_grid_fwer bundles per threshold "
              "for explicitly exploratory inspection"),
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie between zero and one")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if args.top_bundles_per_threshold < 0:
        raise ValueError("--top-bundles-per-threshold cannot be negative")

    source_root = args.grid_results.resolve()
    output_root = args.output_dir.resolve()
    bundles_path = source_root / "observed_bundles_grid_fwer.csv"
    config_path = source_root / "bundle_fwer_config.json"
    if not bundles_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("grid result directory lacks bundles or configuration")
    bundles = pd.read_csv(bundles_path)
    config = json.loads(config_path.read_text())
    required = {
        "cluster_forming_p", "threshold_index", "bundle", "sign",
        "edge_count", "p_grid_fwer",
    }
    if not required.issubset(bundles.columns):
        raise ValueError(f"bundle table lacks columns: {sorted(required - set(bundles.columns))}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    for threshold_index, probability in enumerate(config["cluster_forming_p_grid"]):
        threshold_rows = bundles[
            np.isclose(bundles["cluster_forming_p"], probability, rtol=0, atol=1e-15)
        ]
        significant = threshold_rows[
            threshold_rows["p_grid_fwer"] <= args.alpha
        ].copy()
        selected = significant
        if args.top_bundles_per_threshold:
            exploratory = threshold_rows.nsmallest(
                args.top_bundles_per_threshold,
                ["p_grid_fwer", "statistic"],
            )
            selected = pd.concat([significant, exploratory]).drop_duplicates(
                subset=["bundle"]
            )
        summary = {
            "threshold_index": threshold_index,
            "cluster_forming_p": probability,
            "significant_bundles": int(len(significant)),
            "significant_positive_bundles": int((significant["sign"] > 0).sum()),
            "significant_negative_bundles": int((significant["sign"] < 0).sum()),
            "selected_bundles": int(len(selected)),
            "expected_edges": int(selected["edge_count"].sum()),
            "exported_edges": 0,
            "minimum_p_grid_fwer": (
                float(selected["p_grid_fwer"].min()) if len(selected) else None
            ),
            "raw_csv": None,
        }
        if len(selected):
            slug = threshold_slug(probability)
            selection_suffix = (
                f"_exploratory_top{args.top_bundles_per_threshold}"
                if args.top_bundles_per_threshold else ""
            )
            raw_output = output_root / (
                f"LTLEvsRTLE_gridFWER_{slug}{selection_suffix}.csv"
            )
            processed_output = raw_output.with_name(
                f"{raw_output.stem}_v2_processed.csv"
            )
            source_edges = (
                source_root / "thresholds" / slug / "observed_edges_bundled.csv"
            )
            if not source_edges.is_file():
                raise FileNotFoundError(source_edges)
            exported = write_selected_edges(
                source_edges, raw_output, processed_output, selected, args.chunk_size
            )
            if exported != summary["expected_edges"]:
                raise RuntimeError(
                    f"{slug}: expected {summary['expected_edges']} selected edges, "
                    f"exported {exported}"
                )
            linkage_output = raw_output.with_name(f"{raw_output.stem}_v2_linkage.npy")
            np.save(linkage_output, np.array([]))
            params_output = raw_output.with_name(f"{raw_output.stem}_v2_params.json")
            network_count = int(selected["sign"].nunique())
            manifest = {
                "schema_version": 1,
                "pipeline": "bundle-grid-fwer-visualization-export",
                "created_at": created_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "invocation": "03_prepResultsForVisualization/prepare_bundle_grid_fwer.py",
                "parameters": {
                    "nr_networks": network_count,
                    "min_network_size": config["min_size"],
                    "min_cluster_voxels": config["min_cluster_voxels"],
                    "neighbor_dist": config["neighbor_dist"],
                    "strict_bundles": config["strict_bundles"],
                    "top_n": None,
                    "tstat_threshold": None,
                    "grid_fwer_alpha": args.alpha,
                    "exploratory_top_bundles_per_threshold": (
                        args.top_bundles_per_threshold
                    ),
                    "cluster_forming_p": probability,
                    "threshold_index": threshold_index,
                    "two_sided": config["two_sided"],
                },
                "input": {
                    "path": str(raw_output),
                    "sha256": sha256_file(raw_output),
                    "rows": exported,
                    "statistical_source": str(source_root),
                },
                "results": {
                    "retained_edges": exported,
                    "bundles": int(len(selected)),
                    "networks": network_count,
                },
                "outputs": {
                    "processed_csv": processed_output.name,
                    "processed_csv_sha256": sha256_file(processed_output),
                    "linkage_npy": linkage_output.name,
                    "linkage_npy_sha256": sha256_file(linkage_output),
                },
                "inference": {
                    "correction": config["grid_correction"],
                    "alpha": args.alpha,
                    "contains_nonsignificant_exploratory_bundles": bool(
                        args.top_bundles_per_threshold
                    ),
                    "pvalue_column": "bundle-level p_grid_fwer",
                    "positive_effect": "LTLE > RTLE",
                    "negative_effect": "RTLE > LTLE",
                    "selected_source_bundles": selected[
                        ["bundle", "sign", "edge_count", "mass", "p_grid_fwer"]
                    ].to_dict(orient="records"),
                },
                "recuts": [],
            }
            params_output.write_text(json.dumps(manifest, indent=2) + "\n")
            summary["exported_edges"] = exported
            summary["raw_csv"] = raw_output.name
        summary_rows.append(summary)

    pd.DataFrame(summary_rows).to_csv(output_root / "visualization_summary.csv", index=False)
    (output_root / "README.md").write_text(
        "# LTLE vs RTLE grid-FWER visualization exports\n\n"
        + (
            f"This is an **exploratory, non-significance-filtered** export of "
            f"the {args.top_bundles_per_threshold} smallest adjusted-p bundles "
            "per threshold. No exported bundle should be described as "
            f"significant unless it independently has `p_grid_fwer <= {args.alpha:g}`. "
            if args.top_bundles_per_threshold else
            f"Only bundles with `p_grid_fwer <= {args.alpha:g}` are exported. "
        )
        + "Each "
        "cluster-forming threshold is a separate representation of the corrected "
        "result; do not pool edge counts across thresholds.\n\n"
        "Open `04_coffee-dac/pyqt_launcher.py`, select one of the CSV files that "
        "does **not** end in `_v2_processed.csv`, then choose **Load existing v2 "
        "results (fast)**. The accompanying processed CSV already contains the "
        "permutation-derived bundles. Positive effects are LTLE > RTLE; negative "
        "effects are RTLE > LTLE. The `pvalue` column contains bundle-level "
        "grid-adjusted FWER p-values, repeated for each member edge.\n"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
