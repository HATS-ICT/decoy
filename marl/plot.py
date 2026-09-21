"""Plot learning curves from a training run's metrics.csv.

    python -m marl.plot runs/ippo-T-vs-random
    python -m marl.plot runs/a runs/b --out comparison.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Panels: (csv column, axis label, whether a 0-1 axis makes sense)
PANELS = [
    ("return_T", "mean return (T)", False),
    ("win_rate_T", "win rate (T)", True),
    ("plant_rate", "bomb plant rate", True),
    ("kills_per_episode", "kills / episode", False),
    ("T_entropy", "policy entropy (T)", False),
    ("T_explained_variance", "value explained variance", False),
]


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.csv in {run_dir}")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")

    columns = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[key] = values
    return columns


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="output image (default: <first run>/curves.png)")
    args = parser.parse_args(argv)

    out = args.out or (args.runs[0] / "curves.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DECOY - IPPO learning curves", fontsize=14)

    for run_dir in args.runs:
        metrics = read_metrics(run_dir)
        steps = metrics.get("steps", [])
        label = run_dir.name
        for ax, (column, ylabel, unit_axis) in zip(axes.ravel(), PANELS):
            if column not in metrics:
                ax.set_visible(False)
                continue
            ax.plot(steps, metrics[column], label=label, linewidth=1.6)
            ax.set_xlabel("environment steps")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if unit_axis:
                ax.set_ylim(-0.02, 1.02)

    for ax in axes.ravel():
        if ax.get_visible() and len(args.runs) > 1:
            ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
