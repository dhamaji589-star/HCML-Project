"""Summarize downloaded MAD22 method result packages for report writing."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_FILES = {
    "OpenCV": [
        "results/extracted/opencv/hcml_project/metadata/mad22_opencv_smoke_generated_recovery_eval_original_schedule_20.csv",
        "results/extracted_subset2/opencv/hcml_project/metadata/mad22_opencv_subset2_generated_recovery_eval.csv",
        "results/extracted_subset3/opencv/hcml_project/metadata/mad22_opencv_subset3_generated_recovery_eval.csv",
    ],
    "FaceMorpher": [
        "results/extracted/facemorpher/hcml_project/metadata/mad22_facemorpher_subset_generated_recovery_eval.csv",
        "results/extracted_subset2/facemorpher/hcml_project/metadata/mad22_facemorpher_subset2_generated_recovery_eval.csv",
        "results/extracted_subset3/facemorpher/hcml_project/metadata/mad22_facemorpher_subset3_generated_recovery_eval.csv",
    ],
    "MIPGAN-I": [
        "results/extracted/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset_generated_recovery_eval.csv",
        "results/extracted_subset2/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset2_generated_recovery_eval.csv",
        "results/extracted_subset3/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset3_generated_recovery_eval.csv",
    ],
    "MIPGAN-II": [
        "results/extracted/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset_generated_recovery_eval.csv",
        "results/extracted_subset2/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset2_generated_recovery_eval.csv",
        "results/extracted_subset3/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset3_generated_recovery_eval.csv",
    ],
    "WebMorph": [
        "results/extracted/webmorph/hcml_project/metadata/mad22_webmorph_subset_generated_recovery_eval.csv",
        "results/extracted_subset2/webmorph/hcml_project/metadata/mad22_webmorph_subset2_generated_recovery_eval.csv",
        "results/extracted_subset3/webmorph/hcml_project/metadata/mad22_webmorph_subset3_generated_recovery_eval.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready result tables.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/summary_method_results.csv"),
        help="Summary CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summary_method_results.md"),
        help="Summary Markdown path.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def summarize_method(method: str, paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        rows.extend(read_rows(path))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)

    summaries = []
    for setting in ["negfacediff", "adaptdiff"]:
        setting_rows = grouped[setting]
        total = len(setting_rows)
        successes = sum(
            row["closer_to_hidden"].strip().lower() == "true"
            for row in setting_rows
        )
        margins = [float(row["margin_hidden_minus_known"]) for row in setting_rows]
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        summaries.append(
            {
                "method": method,
                "setting": setting,
                "successes": str(successes),
                "trials": str(total),
                "success_rate": f"{successes / total:.3f}" if total else "0.000",
                "mean_margin_hidden_minus_known": f"{mean_margin:.6f}",
            }
        )
    return summaries


def pivot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][row["setting"]] = row

    pivoted = []
    for method in METHOD_FILES:
        neg = by_method[method]["negfacediff"]
        adapt = by_method[method]["adaptdiff"]
        pivoted.append(
            {
                "method": method,
                "negfacediff_success": f"{neg['successes']}/{neg['trials']}",
                "negfacediff_margin": neg["mean_margin_hidden_minus_known"],
                "adaptdiff_success": f"{adapt['successes']}/{adapt['trials']}",
                "adaptdiff_margin": adapt["mean_margin_hidden_minus_known"],
            }
        )
    return pivoted


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "negfacediff_success",
        "negfacediff_margin",
        "adaptdiff_success",
        "adaptdiff_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# MAD22 Subset Results",
        "",
        "| Method | NegFaceDiff Success | NegFaceDiff Margin | AdaptDiff Success | AdaptDiff Margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {negfacediff_success} | {negfacediff_margin} | "
            "{adaptdiff_success} | {adaptdiff_margin} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Each method uses 30 morph images, evaluated as 60 directed recovery trials.",
            "The margin is cosine(generated, hidden identity) minus cosine(generated, known identity).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = []
    for method, path_texts in METHOD_FILES.items():
        summaries.extend(summarize_method(method, [Path(path_text) for path_text in path_texts]))

    pivoted = pivot_rows(summaries)
    write_csv(pivoted, args.output_csv)
    write_markdown(pivoted, args.output_md)

    print(f"Summary CSV: {args.output_csv}")
    print(f"Summary Markdown: {args.output_md}")
    for row in pivoted:
        print(
            f"{row['method']}: NegFaceDiff {row['negfacediff_success']} "
            f"({row['negfacediff_margin']}), AdaptDiff {row['adaptdiff_success']} "
            f"({row['adaptdiff_margin']})"
        )


if __name__ == "__main__":
    main()
