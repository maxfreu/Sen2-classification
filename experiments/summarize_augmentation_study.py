import argparse
import ast
import csv
from pathlib import Path

import yaml

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def iter_run_directories(output_root, experiment_name):
    for experiment_dir in sorted(output_root.glob(f"{experiment_name}_*")):
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "config.yaml").exists():
                yield run_dir


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


def escape_latex(value):
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


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


def get_paper_table_rows(rows):
    baseline_row = next((row for row in rows if row["group"] == "baseline"), None)
    final_rows = [
        row
        for row in rows
        if row["group"] == "final" and row["status"] == "done" and row["metric"] is not None
    ]
    best_single_rows = get_best_rows_by_variant(rows, "single")

    paper_rows = []
    if baseline_row is not None:
        paper_rows.append(
            {
                "variant": baseline_row["variant"],
                "setting": baseline_row["setting"],
                "metric": baseline_row["metric"],
                "kl_div": baseline_row["kl_div"],
                "delta_vs_baseline": 0.0 if baseline_row["metric"] is not None else None,
            }
        )

    for row in final_rows:
        paper_rows.append(
            {
                "variant": row["variant"],
                "setting": row["setting"],
                "metric": row["metric"],
                "kl_div": row["kl_div"],
                "delta_vs_baseline": row["delta_vs_baseline"],
            }
        )

    for row in best_single_rows:
        paper_rows.append(
            {
                "variant": row["variant"],
                "setting": row["setting"],
                "metric": row["metric"],
                "kl_div": row["kl_div"],
                "delta_vs_baseline": row["delta_vs_baseline"],
            }
        )

    return sorted(
        paper_rows,
        key=lambda row: (
            row["delta_vs_baseline"] is None,
            row["delta_vs_baseline"] if row["delta_vs_baseline"] is not None else float("inf"),
            row["variant"],
        ),
    )


def build_latex_table(rows):
    header = [
        "Variant",
        "Acc (\\%)",
        "$\\Delta$ vs. baseline (pp)",
        "KL div",
    ]
    body_rows = [
        [
            escape_latex(row["variant"]),
            format_percent(row["metric"]),
            format_delta(row["delta_vs_baseline"]),
            format_kl_div(row["kl_div"]),
        ]
        for row in rows
    ]
    widths = [max(len(values[index]) for values in [header, *body_rows]) for index in range(len(header))]

    def format_row(values):
        padded = [value.ljust(width) for value, width in zip(values, widths)]
        return " & ".join(padded) + r" \\" 

    lines = [
        "\\begin{tabular}{lrrr}",
        "\\hline",
        format_row(header),
        "\\hline",
    ]

    for row in body_rows:
        lines.append(format_row(row))

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def parse_variant_and_setting(group, label):
    if group == "baseline":
        return "baseline", "none"
    if label.endswith(")") and " (" in label:
        variant, setting = label.rsplit(" (", 1)
        return variant, setting[:-1]
    return label, ""


def load_run_metadata(run_dir):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    data_config = config.get("data", {})
    study_run = data_config.get("augmentation_study_run", {})

    group = study_run.get("group", "")
    label = study_run.get("label", run_dir.name)
    variant = study_run.get("variant")
    setting = study_run.get("setting")
    if variant is None or setting is None:
        variant, setting = parse_variant_and_setting(group, label)

    return {
        "id": study_run.get("id"),
        "name": study_run.get("name", run_dir.name),
        "label": label,
        "group": group,
        "variant": variant,
        "setting": setting,
    }


def sort_rows(rows):
    group_order = {"baseline": 0, "final": 1, "single": 2}
    return sorted(
        rows,
        key=lambda row: (
            group_order.get(row["group"], 99),
            row["variant"],
            row["setting"],
            row["name"],
        ),
    )


def build_markdown(rows, metric_name, kl_metric_name):
    paper_rows = get_paper_table_rows(rows)

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
            "| Variant | Acc (%) | Delta vs baseline (pp) | KL div |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for row in paper_rows:
        lines.append(
            f"| {row['variant']} | {format_percent(row['metric'])} | {format_delta(row['delta_vs_baseline'])} | {format_kl_div(row['kl_div'])} |"
        )

    lines.extend([
        "",
        "## LaTeX Table",
        "",
        "```latex",
        build_latex_table(paper_rows),
        "```",
    ])

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

        if run["group"] == "baseline":
            baseline_metric = metric_value

        rows.append(
            {
                "id": run["id"],
                "name": run["name"],
                "label": run["label"],
                "group": run["group"],
                "variant": run["variant"],
                "setting": run["setting"],
                "status": status,
                "metric": metric_value,
                "accuracy_pct": format_percent(metric_value),
                "kl_div": kl_div_value,
                "delta_vs_baseline": None,
                "delta_vs_baseline_pp": "",
                "output_dir": "" if run_dir is None else str(run_dir),
            }
        )

    rows = sort_rows(rows)

    for row in rows:
        if row["metric"] is not None and baseline_metric is not None:
            row["delta_vs_baseline"] = row["metric"] - baseline_metric
            row["delta_vs_baseline_pp"] = format_delta(row["delta_vs_baseline"])

    write_csv(rows, output_prefix.with_name(output_prefix.name + "_summary.csv"))
    output_prefix.with_name(output_prefix.name + "_summary.md").write_text(build_markdown(rows, args.metric, args.kl_metric))


if __name__ == "__main__":
    main()