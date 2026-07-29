"""Create a small directed-trial subset for fast pipeline smoke tests."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small smoke-test subset from directed MAD22 trial metadata."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_trials.csv"),
        help="Full directed trial CSV created by build_mad22_metadata.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_trials.csv"),
        help="Small directed smoke-test CSV to write.",
    )
    parser.add_argument(
        "--num-morphs",
        type=int,
        default=10,
        help="Number of unique morph images to include. Each morph contributes two rows.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for deterministic first-mode subset selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when selecting morphs.",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Choose the first morphs deterministically or a seeded random subset.",
    )
    return parser.parse_args()


def read_trials(input_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"Input CSV has no header: {input_csv}")

    return rows, fieldnames


def group_by_morph(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["morph_path"]].append(row)
    return dict(grouped)


def select_morphs(
    grouped: dict[str, list[dict[str, str]]],
    num_morphs: int,
    selection: str,
    seed: int,
    start_index: int,
) -> list[str]:
    morph_paths = sorted(grouped)
    if num_morphs <= 0:
        raise ValueError("--num-morphs must be greater than zero")
    if start_index < 0:
        raise ValueError("--start-index must be zero or greater")
    if start_index + num_morphs > len(morph_paths):
        raise ValueError(
            f"Requested {num_morphs} morphs from start index {start_index}, "
            f"but only {len(morph_paths)} are available."
        )

    if selection == "first":
        return morph_paths[start_index : start_index + num_morphs]

    rng = random.Random(seed)
    return sorted(rng.sample(morph_paths, num_morphs))


def write_subset(output_csv: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_trials(args.input_csv)
    grouped = group_by_morph(rows)
    selected_morphs = select_morphs(
        grouped=grouped,
        num_morphs=args.num_morphs,
        selection=args.selection,
        seed=args.seed,
        start_index=args.start_index,
    )

    subset_rows: list[dict[str, str]] = []
    for morph_path in selected_morphs:
        pair_rows = sorted(grouped[morph_path], key=lambda row: row["direction"])
        subset_rows.extend(pair_rows)

    write_subset(args.output_csv, subset_rows, fieldnames)

    print(f"Input trials: {len(rows)}")
    print(f"Unique morphs in input: {len(grouped)}")
    print(f"Selected morphs: {len(selected_morphs)}")
    print(f"Smoke-test directed trials: {len(subset_rows)}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
