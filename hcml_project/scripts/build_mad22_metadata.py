"""Build paired metadata for hidden-identity recovery experiments on MAD22.

The MAD22 morph filenames encode the two source images:

    001_08-vs-010_08.jpg

This script turns each usable morph into two directed recovery trials:

    M_AB + 001_08 -> recover 010_08
    M_AB + 010_08 -> recover 001_08
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


MORPH_NAME_RE = re.compile(r"^(?P<a>\d{3}_\d{2})-vs-(?P<b>\d{3}_\d{2})$")


@dataclass(frozen=True)
class MorphPair:
    morph_path: Path
    source_a_path: Path
    source_b_path: Path
    source_a_id: str
    source_b_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create directed morph/known/hidden metadata rows from MAD22."
    )
    parser.add_argument(
        "--mad22-root",
        type=Path,
        default=Path("MAD22"),
        help="Path to the MAD22 directory.",
    )
    parser.add_argument(
        "--method",
        default="OpenCV",
        help="Morphing method folder under MAD22/original_sorted.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_trials.csv"),
        help="CSV path to write directed recovery trials.",
    )
    parser.add_argument(
        "--skipped-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_skipped.csv"),
        help="CSV path to write skipped morphs and reasons.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of paths relative to the project root.",
    )
    return parser.parse_args()


def display_path(path: Path, project_root: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def find_source_image(bonafide_dir: Path, source_id: str) -> Path:
    return bonafide_dir / f"{source_id}.jpg"


def collect_pairs(mad22_root: Path, method: str) -> tuple[list[MorphPair], list[dict[str, str]]]:
    original_sorted = mad22_root / "original_sorted"
    bonafide_dir = original_sorted / "BonaFide"
    morph_dir = original_sorted / method

    if not bonafide_dir.is_dir():
        raise FileNotFoundError(f"BonaFide directory not found: {bonafide_dir}")
    if not morph_dir.is_dir():
        raise FileNotFoundError(f"Morph method directory not found: {morph_dir}")

    pairs: list[MorphPair] = []
    skipped: list[dict[str, str]] = []

    for morph_path in sorted(morph_dir.iterdir()):
        if not morph_path.is_file():
            continue

        match = MORPH_NAME_RE.match(morph_path.stem)
        if match is None:
            skipped.append(
                {
                    "morph_path": str(morph_path),
                    "reason": "filename_does_not_match_expected_pattern",
                    "source_a_path": "",
                    "source_b_path": "",
                }
            )
            continue

        source_a_id = match.group("a")
        source_b_id = match.group("b")
        source_a_path = find_source_image(bonafide_dir, source_a_id)
        source_b_path = find_source_image(bonafide_dir, source_b_id)

        missing = []
        if not source_a_path.is_file():
            missing.append(source_a_path.name)
        if not source_b_path.is_file():
            missing.append(source_b_path.name)

        if missing:
            skipped.append(
                {
                    "morph_path": str(morph_path),
                    "reason": "missing_source_images:" + "|".join(missing),
                    "source_a_path": str(source_a_path),
                    "source_b_path": str(source_b_path),
                }
            )
            continue

        pairs.append(
            MorphPair(
                morph_path=morph_path,
                source_a_path=source_a_path,
                source_b_path=source_b_path,
                source_a_id=source_a_id,
                source_b_id=source_b_id,
            )
        )

    return pairs, skipped


def write_trials_csv(
    pairs: list[MorphPair],
    output_csv: Path,
    project_root: Path,
    method: str,
    absolute_paths: bool,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "morph_method",
        "direction",
        "morph_path",
        "known_path",
        "hidden_path",
        "known_id",
        "hidden_id",
        "source_a_id",
        "source_b_id",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, pair in enumerate(pairs):
            base_trial_id = f"{method}_{pair.source_a_id}_vs_{pair.source_b_id}"
            rows = [
                {
                    "trial_id": f"{base_trial_id}_A_to_B",
                    "direction": "A_to_B",
                    "known_path": pair.source_a_path,
                    "hidden_path": pair.source_b_path,
                    "known_id": pair.source_a_id,
                    "hidden_id": pair.source_b_id,
                },
                {
                    "trial_id": f"{base_trial_id}_B_to_A",
                    "direction": "B_to_A",
                    "known_path": pair.source_b_path,
                    "hidden_path": pair.source_a_path,
                    "known_id": pair.source_b_id,
                    "hidden_id": pair.source_a_id,
                },
            ]

            for row in rows:
                writer.writerow(
                    {
                        "trial_id": row["trial_id"],
                        "morph_method": method,
                        "direction": row["direction"],
                        "morph_path": display_path(pair.morph_path, project_root, absolute_paths),
                        "known_path": display_path(row["known_path"], project_root, absolute_paths),
                        "hidden_path": display_path(row["hidden_path"], project_root, absolute_paths),
                        "known_id": row["known_id"],
                        "hidden_id": row["hidden_id"],
                        "source_a_id": pair.source_a_id,
                        "source_b_id": pair.source_b_id,
                    }
                )


def write_skipped_csv(
    skipped: list[dict[str, str]],
    skipped_csv: Path,
    project_root: Path,
    absolute_paths: bool,
) -> None:
    skipped_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["morph_path", "reason", "source_a_path", "source_b_path"]

    with skipped_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in skipped:
            output_row = dict(row)
            for key in ("morph_path", "source_a_path", "source_b_path"):
                if output_row[key]:
                    output_row[key] = display_path(Path(output_row[key]), project_root, absolute_paths)
            writer.writerow(output_row)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()

    pairs, skipped = collect_pairs(args.mad22_root, args.method)
    write_trials_csv(
        pairs=pairs,
        output_csv=args.output_csv,
        project_root=project_root,
        method=args.method,
        absolute_paths=args.absolute_paths,
    )
    write_skipped_csv(
        skipped=skipped,
        skipped_csv=args.skipped_csv,
        project_root=project_root,
        absolute_paths=args.absolute_paths,
    )

    print(f"Method: {args.method}")
    print(f"Usable morphs: {len(pairs)}")
    print(f"Directed trials: {len(pairs) * 2}")
    print(f"Skipped morphs: {len(skipped)}")
    print(f"Trials CSV: {args.output_csv}")
    print(f"Skipped CSV: {args.skipped_csv}")


if __name__ == "__main__":
    main()
