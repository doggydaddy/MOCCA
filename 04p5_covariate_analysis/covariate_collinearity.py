"""How much of the group contrast each covariate set explains.

Written to test, and reject, the leading explanation for why age and sex
jointly remove an effect that neither removes alone. If collinearity with
group were responsible, the joint variance inflation factor would be large;
measured here it is 1.12, a ~6% inflation of the group coefficient's standard
error, which cannot account for the observed shift in p_FWER.

Reported for completeness so the rejection is auditable rather than asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def explained(target: np.ndarray, columns: list[np.ndarray]) -> float:
    design = np.column_stack([np.ones(target.size)] + columns)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    centred = target - target.mean()
    return float(1.0 - residual @ residual / (centred @ centred))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("design", type=Path, help="design.npz containing all covariates")
    parser.add_argument("--group-column", default="group")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stored = np.load(args.design.resolve(), allow_pickle=True)
    names = [str(name) for name in stored["column_names"]]
    matrix = stored["matrix"]
    if args.group_column not in names:
        raise ValueError(f"{args.design} has no '{args.group_column}' column")
    group = matrix[:, names.index(args.group_column)]
    covariates = {
        name: matrix[:, index]
        for index, name in enumerate(names)
        if name not in (args.group_column, "intercept")
    }
    if not covariates:
        raise ValueError("design carries no covariates to assess")

    results = {}
    for name, column in covariates.items():
        value = explained(group, [column])
        results[name] = {"r_squared": value, "vif": 1.0 / (1.0 - value)}
    joint = explained(group, list(covariates.values()))
    results["joint"] = {"r_squared": joint, "vif": 1.0 / (1.0 - joint)}
    additive = sum(results[name]["r_squared"] for name in covariates)
    results["super_additivity"] = joint - additive
    results["group_sd"] = float(group.std(ddof=0))
    results["group_residual_sd_joint"] = float(np.sqrt(1 - joint) * group.std(ddof=0))
    results["standard_error_inflation_joint"] = float(1.0 / np.sqrt(1 - joint))
    results["condition_number"] = float(np.linalg.cond(matrix))

    for name in covariates:
        r = results[name]
        print(f"  {name:16s} R^2 of {args.group_column} = {r['r_squared']:.4f}   VIF = {r['vif']:.3f}")
    print(f"  {'joint':16s} R^2 of {args.group_column} = {joint:.4f}   VIF = {results['joint']['vif']:.3f}")
    print(f"\n  additive prediction {additive:.4f} vs joint {joint:.4f}"
          f"  -> super-additive by {results['super_additivity']:+.4f}")
    print(f"  group residual SD {results['group_sd']:.4f} -> "
          f"{results['group_residual_sd_joint']:.4f}"
          f"  (standard error x{results['standard_error_inflation_joint']:.3f})")
    print(f"  design condition number {results['condition_number']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
