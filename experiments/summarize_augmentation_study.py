import argparse
import ast
import csv
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.augmentation_study_runs import build_runs


def parse_metric_value(raw_value):
    try:
        return float(raw_value)
    except ValueError:
        try:
            return ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return raw_value


def parse_report(report_path):
    metrics = {}
    for line in report_path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key == "Years":
            continue
        metrics[key] = parse_metric_value(value)
    return metrics


def find_run_directory(output_root, experiment_name, version):
    matches = sorted(output_root.glob(f"{experiment_name}_*/{version}"))
    if not matches:
        return None
    return matches[0]


def format_percent(value):
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def format_delta(value):
    if value is None:
        return ""
    return f"{value * 100:+.2f}"


def format_kl_div(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def write_csv(rows, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "name",
                "label",
                "group",
                "variant",
                "setting",
                "status",
                "metric",
                "accuracy_pct",
                "kl_div",
                "delta_vs_baseline",
                "delta_vs_baseline_pp",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def get_best_rows_by_variant(rows, group):
    best_rows = []
    variant_candidates = {}
    variant_order = []

    for row in rows:
        if row["group"] != group:
            continue
        variant = row["variant"]
        if variant not in variant_candidates:
            variant_candidates[variant] = []
            variant_order.append(variant)
        if row["status"] == "done" and row["metric"] is not None:
            variant_candidates[variant].append(row)

    for variant in variant_order:
        candidates = variant_candidates[variant]
        if not candidates:
            continue
        best_rows.append(max(candidates, key=lambda row: row["metric"]))

    return best_rows


def build_markdown(rows, metric_name, kl_metric_name):
    completed_rows = [row for row in rows if row["status"] == "done"]
    baseline_row = next((row for row in rows if row["group"] == "baseline"), None)
    default_row = next((row for row in rows if row["group"] == "final"), None)
    best_single_rows = get_best_rows_by_variant(rows, "single")

    lines = [
        f"# Augmentation Study Summary ({metric_name}; {kl_metric_name})",
        "",
        "| Variant | Setting | Group | Acc (%) | KL div | Delta vs baseline (pp) | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {variant} | {setting} | {group} | {acc} | {kl_div} | {delta} | {status} |".format(
                variant=row["variant"],
                setting=row["setting"],
                group=row["group"],
                acc=format_percent(row["metric"]),
                kl_div=format_kl_div(row["kl_div"]),
                delta=format_delta(row["delta_vs_baseline"]),
                status=row["status"],
            )
        )

    lines.extend(["", "## Paper Table", ""])
    lines.extend(
        [
            "| Variant | Best setting | Acc (%) | KL div | Delta vs baseline (pp) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    if baseline_row is not None:
        lines.append(
            f"| baseline | {baseline_row['setting']} | {format_percent(baseline_row['metric'])} | {format_kl_div(baseline_row['kl_div'])} | {format_delta(0.0 if baseline_row['metric'] is not None else None)} |"
        )

    if default_row is not None:
        lines.append(
            f"| {default_row['variant']} | {default_row['setting']} | {format_percent(default_row['metric'])} | {format_kl_div(default_row['kl_div'])} | {format_delta(default_row['delta_vs_baseline'])} |"
        )

    for row in best_single_rows:
        lines.append(
            f"| {row['variant']} | {row['setting']} | {format_percent(row['metric'])} | {format_kl_div(row['kl_div'])} | {format_delta(row['delta_vs_baseline'])} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--experiment-name", default="augmentation_study")
    parser.add_argument("--metric", default="acc_ds_avg_seq_len=64")
    parser.add_argument("--kl-metric", default="kl_div_ds_avg_seq_len=64")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_prefix = Path(args.output_prefix) if args.output_prefix else output_root / args.experiment_name

    rows = []
    baseline_metric = None

    for run_id, run in enumerate(build_runs()):
        version = f"{run_id:02d}_{run['name']}"
        run_dir = find_run_directory(output_root, args.experiment_name, version)
        report_path = None if run_dir is None else run_dir / "eval" / "validation_report_seq_len=64.txt"

        metric_value = None
        kl_div_value = None
        status = "missing"
        if report_path is not None and report_path.exists():
            report_metrics = parse_report(report_path)
            metric_value = report_metrics.get(args.metric)
            kl_div_value = report_metrics.get(args.kl_metric)
            status = "done" if metric_value is not None else "missing-metric"

        if run["group"] == "baseline":
            baseline_metric = metric_value

        rows.append(
            {
                "id": run_id,
                "name": run["name"],
                "label": run["label"],
                "group": run["group"],
                "variant": run.get("variant", run["label"]),
                "setting": run.get("setting", ""),
                "status": status,
                "metric": metric_value,
                "accuracy_pct": format_percent(metric_value),
                "kl_div": kl_div_value,
                "delta_vs_baseline": None,
                "delta_vs_baseline_pp": "",
                "output_dir": "" if run_dir is None else str(run_dir),
            }
        )

    for row in rows:
        if row["metric"] is not None and baseline_metric is not None:
            row["delta_vs_baseline"] = row["metric"] - baseline_metric
            row["delta_vs_baseline_pp"] = format_delta(row["delta_vs_baseline"])

    write_csv(rows, output_prefix.with_name(output_prefix.name + "_summary.csv"))
    output_prefix.with_name(output_prefix.name + "_summary.md").write_text(build_markdown(rows, args.metric, args.kl_metric))


if __name__ == "__main__":
    main()