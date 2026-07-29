import argparse
import ast
import csv
import math
from pathlib import Path

import duckdb
import yaml


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
        if key != "Years":
            metrics[key] = parse_metric_value(value)
    return metrics


def kl_divergence(pred_counts, true_counts):
    labels = sorted(set(pred_counts) | set(true_counts))
    pred_values = [pred_counts.get(label, 0.0) + 1e-7 for label in labels]
    true_values = [true_counts.get(label, 0.0) + 1e-7 for label in labels]
    pred_total = sum(pred_values)
    true_total = sum(true_values)
    return sum(
        (pred_value / pred_total) * math.log((pred_value / pred_total) / (true_value / true_total))
        for pred_value, true_value in zip(pred_values, true_values)
    )


def parse_prediction_metrics(eval_dir, seq_len):
    prediction_paths = sorted(eval_dir.glob(f"prediction_seq_len={seq_len}_year=*.sqlite"))
    if not prediction_paths:
        return None, None

    accuracies = []
    kl_divs = []
    for prediction_path in prediction_paths:
        accuracy = duckdb.query(
            f"SELECT avg(correct) FROM sqlite_scan('{prediction_path}', 'val')"
        ).fetchone()[0]
        true_counts = dict(
            duckdb.query(
                f"SELECT y_true, count(*) FROM sqlite_scan('{prediction_path}', 'val') GROUP BY y_true"
            ).fetchall()
        )
        pred_counts = dict(
            duckdb.query(
                f"SELECT y_pred, count(*) FROM sqlite_scan('{prediction_path}', 'val') GROUP BY y_pred"
            ).fetchall()
        )
        accuracies.append(accuracy)
        kl_divs.append(kl_divergence(pred_counts, true_counts))

    return sum(accuracies) / len(accuracies), sum(kl_divs) / len(kl_divs)


def iter_run_directories(output_root, experiment_name):
    for experiment_dir in sorted(output_root.glob(f"{experiment_name}_*")):
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "config.yaml").exists():
                yield run_dir


def format_percent(value):
    return "" if value is None else f"{value * 100:.2f}"


def format_delta(value):
    return "" if value is None else f"{value * 100:+.2f}"


def format_decline(value):
    return "" if value is None else f"{value * 100:.2f}"


def format_kl_div(value):
    return "" if value is None else f"{value:.4f}"


def escape_latex(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def load_run_metadata(run_dir):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    data_config = config.get("data", {})
    omitted_band = data_config.get("omitted_band", "")

    return {
        "name": run_dir.name,
        "omitted_band": omitted_band,
        "input_band_indices": data_config.get("input_band_indices"),
        "satellite_input_channels": data_config.get("satellite_input_channels"),
    }


def sort_rows(rows):
    return sorted(rows, key=lambda row: (row["omitted_band"] != "none", row["omitted_band"], row["name"]))


def sort_comparison_rows(rows):
    return sorted(
        [row for row in rows if row["omitted_band"] != "none"],
        key=lambda row: (row["decline"] is None, -(row["decline"] or 0.0), row["omitted_band"]),
    )


def write_csv(rows, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "omitted_band",
                "input_band_indices",
                "satellite_input_channels",
                "status",
                "metric",
                "accuracy_pct",
                "kl_div",
                "delta_vs_baseline",
                "delta_vs_baseline_pp",
                "decline",
                "decline_pp",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_latex_table(rows):
    row_end = r"\\"
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Omitted band & Acc (\%) & $\Delta$ vs. baseline (pp) " + row_end,
        r"\hline",
    ]

    for row in rows:
        lines.append(
            f"{escape_latex(row['omitted_band'])} & {format_percent(row['metric'])} & {format_delta(row['delta_vs_baseline'])} "
            + row_end
        )

    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def build_markdown(rows, metric_name, kl_metric_name):
    baseline = next((row for row in rows if row["omitted_band"] == "none"), None)
    comparison_rows = sort_comparison_rows(rows)

    lines = [f"# Input Band Ablation Summary ({metric_name}; {kl_metric_name})", ""]
    if baseline is not None:
        lines.extend(
            [
                f"Baseline all bands: {format_percent(baseline['metric'])}% acc, {format_kl_div(baseline['kl_div'])} KL div.",
                "",
            ]
        )

    lines.extend(
        [
            "| Run | Omitted band | Input bands | Acc (%) | KL div | Delta vs baseline (pp) | Accuracy decline (pp) | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {name} | {omitted_band} | {input_bands} | {acc} | {kl_div} | {delta} | {decline} | {status} |".format(
                name=row["name"],
                omitted_band=row["omitted_band"],
                input_bands="" if row["input_band_indices"] is None else row["input_band_indices"],
                acc=format_percent(row["metric"]),
                kl_div=format_kl_div(row["kl_div"]),
                delta=format_delta(row["delta_vs_baseline"]),
                decline=format_decline(row["decline"]),
                status=row["status"],
            )
        )

    lines.extend(
        [
            "",
            "## Sorted Comparison",
            "",
            "| Omitted band | Acc (%) | KL div | Delta vs baseline (pp) | Accuracy decline (pp) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['omitted_band']} | {format_percent(row['metric'])} | {format_kl_div(row['kl_div'])} | {format_delta(row['delta_vs_baseline'])} | {format_decline(row['decline'])} |"
        )

    lines.extend(["", "## LaTeX Table", "", "```latex", build_latex_table(comparison_rows), "```"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--experiment-name", default="input_band_ablation")
    parser.add_argument("--metric", default="acc_ds_avg_seq_len=64")
    parser.add_argument("--kl-metric", default="kl_div_ds_avg_seq_len=64")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_prefix = Path(args.output_prefix) if args.output_prefix else output_root / args.experiment_name
    rows = []
    baseline_metric = None

    for run_dir in iter_run_directories(output_root, args.experiment_name):
        run = load_run_metadata(run_dir)
        report_path = run_dir / "eval" / "validation_report_seq_len=64.txt"
        metric_value = None
        kl_div_value = None
        status = "missing"

        if report_path.exists():
            report_metrics = parse_report(report_path)
            metric_value = report_metrics.get(args.metric)
            kl_div_value = report_metrics.get(args.kl_metric)
            status = "done" if metric_value is not None else "missing-metric"

        if metric_value is None:
            metric_value, kl_div_value = parse_prediction_metrics(run_dir / "eval", args.seq_len)
            if metric_value is not None:
                status = "done-from-predictions"

        if run["omitted_band"] == "none":
            baseline_metric = metric_value

        rows.append(
            {
                "name": run["name"],
                "omitted_band": run["omitted_band"],
                "input_band_indices": run["input_band_indices"],
                "satellite_input_channels": run["satellite_input_channels"],
                "status": status,
                "metric": metric_value,
                "accuracy_pct": format_percent(metric_value),
                "kl_div": kl_div_value,
                "delta_vs_baseline": None,
                "delta_vs_baseline_pp": "",
                "decline": None,
                "decline_pp": "",
                "output_dir": str(run_dir),
            }
        )

    rows = sort_rows(rows)
    for row in rows:
        if row["metric"] is not None and baseline_metric is not None:
            row["delta_vs_baseline"] = row["metric"] - baseline_metric
            row["delta_vs_baseline_pp"] = format_delta(row["delta_vs_baseline"])
            row["decline"] = baseline_metric - row["metric"]
            row["decline_pp"] = format_decline(row["decline"])

    write_csv(rows, output_prefix.with_name(output_prefix.name + "_summary.csv"))
    output_prefix.with_name(output_prefix.name + "_summary.md").write_text(build_markdown(rows, args.metric, args.kl_metric))


if __name__ == "__main__":
    main()