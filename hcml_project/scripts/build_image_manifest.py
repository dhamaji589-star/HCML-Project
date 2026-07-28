"""Build a unique image manifest from directed recovery trial metadata.

The trial CSV repeats images across rows. For example, the same bona fide
source image can be the known identity in one trial and the hidden target in
another. This manifest lets later steps align and embed every image only once.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageRecord:
    image_path: str
    image_type: str
    roles: set[str] = field(default_factory=set)
    identity_ids: set[str] = field(default_factory=set)
    trial_ids: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a unique image manifest from MAD22 recovery trials."
    )
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_trials.csv"),
        help="Directed trial CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_images.csv"),
        help="Unique image manifest CSV to write.",
    )
    return parser.parse_args()


def add_image(
    records: dict[str, ImageRecord],
    image_path: str,
    image_type: str,
    role: str,
    trial_id: str,
    identity_id: str = "",
) -> None:
    if image_path not in records:
        records[image_path] = ImageRecord(image_path=image_path, image_type=image_type)

    record = records[image_path]
    record.roles.add(role)
    record.trial_ids.add(trial_id)
    if identity_id:
        record.identity_ids.add(identity_id)


def build_manifest(trials_csv: Path) -> list[ImageRecord]:
    if not trials_csv.is_file():
        raise FileNotFoundError(f"Trials CSV not found: {trials_csv}")

    records: dict[str, ImageRecord] = {}

    with trials_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            trial_id = row["trial_id"]

            add_image(
                records=records,
                image_path=row["morph_path"],
                image_type="morph",
                role="morph",
                trial_id=trial_id,
            )
            add_image(
                records=records,
                image_path=row["known_path"],
                image_type="bona_fide",
                role="known",
                trial_id=trial_id,
                identity_id=row["known_id"],
            )
            add_image(
                records=records,
                image_path=row["hidden_path"],
                image_type="bona_fide",
                role="hidden",
                trial_id=trial_id,
                identity_id=row["hidden_id"],
            )

    return [records[path] for path in sorted(records)]


def write_manifest(records: list[ImageRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_index",
        "image_path",
        "image_type",
        "roles",
        "identity_ids",
        "num_trials",
        "trial_ids",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "image_index": index,
                    "image_path": record.image_path,
                    "image_type": record.image_type,
                    "roles": "|".join(sorted(record.roles)),
                    "identity_ids": "|".join(sorted(record.identity_ids)),
                    "num_trials": len(record.trial_ids),
                    "trial_ids": "|".join(sorted(record.trial_ids)),
                }
            )


def main() -> None:
    args = parse_args()
    records = build_manifest(args.trials_csv)
    write_manifest(records, args.output_csv)

    morph_count = sum(1 for record in records if record.image_type == "morph")
    source_count = sum(1 for record in records if record.image_type == "bona_fide")

    print(f"Trials CSV: {args.trials_csv}")
    print(f"Unique images: {len(records)}")
    print(f"Unique morph images: {morph_count}")
    print(f"Unique bona fide images: {source_count}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
