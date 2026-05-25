import argparse
import ast
import csv
from pathlib import Path

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
    omitted_season = data_config.get("omitted_season", "")

    return {
        "name": run_dir.name,
        "omitted_season": omitted_season,
        "where": data_config.get("where", ""),
        "val_where": data_config.get("val_where", ""),
    }


def sort_rows(rows):
    season_order = {"none": 0, "spring": 1, "summer": 2, "autumn": 3, "winter": 4}
    return sorted(rows, key=lambda row: (season_order.get(row["omitted_season"], 99), row["name"]))


def sort_comparison_rows(rows):
    return sorted(
        [row for row in rows if row["omitted_season"] != "none"],
        key=lambda row: (row["decline"] is None, -(row["decline"] or 0.0), row["omitted_season"]),
    )


def write_csv(rows, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "omitted_season",
                "where",
                "val_where",
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
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Omitted season & Acc (\%) & KL div & $\Delta$ vs. baseline (pp) & Accuracy decline (pp) " + row_end,
        r"\hline",
    ]

    for row in rows:
        lines.append(
            f"{escape_latex(row['omitted_season'])} & {format_percent(row['metric'])} & {format_kl_div(row['kl_div'])} & {format_delta(row['delta_vs_baseline'])} & {format_decline(row['decline'])} "
            + row_end
        )

    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def build_markdown(rows, metric_name, kl_metric_name):
    baseline = next((row for row in rows if row["omitted_season"] == "none"), None)
    comparison_rows = sort_comparison_rows(rows)

    lines = [f"# Season Ablation Summary ({metric_name}; {kl_metric_name})", ""]
    if baseline is not None:
        lines.extend(
            [
                f"Baseline full year: {format_percent(baseline['metric'])}% acc, {format_kl_div(baseline['kl_div'])} KL div.",
                "",
            ]
        )

    lines.extend(
        [
            "| Run | Omitted season | Where | Acc (%) | KL div | Delta vs baseline (pp) | Accuracy decline (pp) | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {name} | {omitted_season} | {where} | {acc} | {kl_div} | {delta} | {decline} | {status} |".format(
                name=row["name"],
                omitted_season=row["omitted_season"],
                where=row["where"],
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
            "| Omitted season | Acc (%) | KL div | Delta vs baseline (pp) | Accuracy decline (pp) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['omitted_season']} | {format_percent(row['metric'])} | {format_kl_div(row['kl_div'])} | {format_delta(row['delta_vs_baseline'])} | {format_decline(row['decline'])} |"
        )

    lines.extend(["", "## LaTeX Table", "", "```latex", build_latex_table(comparison_rows), "```"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--experiment-name", default="season_ablation")
    parser.add_argument("--metric", default="acc_ds_avg_seq_len=64")
    parser.add_argument("--kl-metric", default="kl_div_ds_avg_seq_len=64")
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

        if run["omitted_season"] == "none":
            baseline_metric = metric_value

        rows.append(
            {
                "name": run["name"],
                "omitted_season": run["omitted_season"],
                "where": run["where"],
                "val_where": run["val_where"],
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
