"""Create a qualitative comparison figure from downloaded MAD22 result packages.

The generated images are read from results/extracted/*, while the original
known/morph/hidden images are read from the local MAD22 paths in the trial CSVs.
This avoids storing the original dataset images inside the result archives.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHODS = [
    {
        "name": "OpenCV",
        "root": Path("results/extracted/opencv"),
        "trials": Path("hcml_project/metadata/mad22_opencv_smoke_trials.csv"),
        "report": Path(
            "hcml_project/outputs/generated_smoke_elasticface_arc_original_schedule_20/sampling_report.csv"
        ),
    },
    {
        "name": "FaceMorpher",
        "root": Path("results/extracted/facemorpher"),
        "trials": Path("hcml_project/metadata/mad22_facemorpher_subset_trials.csv"),
        "report": Path(
            "hcml_project/outputs/generated_mad22_facemorpher_subset_elasticface_arc/sampling_report.csv"
        ),
    },
    {
        "name": "MIPGAN-I",
        "root": Path("results/extracted/mipgan_i"),
        "trials": Path("hcml_project/metadata/mad22_mipgan_i_subset_trials.csv"),
        "report": Path(
            "hcml_project/outputs/generated_mad22_mipgan_i_subset_elasticface_arc/sampling_report.csv"
        ),
    },
    {
        "name": "MIPGAN-II",
        "root": Path("results/extracted/mipgan_ii"),
        "trials": Path("hcml_project/metadata/mad22_mipgan_ii_subset_trials.csv"),
        "report": Path(
            "hcml_project/outputs/generated_mad22_mipgan_ii_subset_elasticface_arc/sampling_report.csv"
        ),
    },
    {
        "name": "WebMorph",
        "root": Path("results/extracted/webmorph"),
        "trials": Path("hcml_project/metadata/mad22_webmorph_subset_trials.csv"),
        "report": Path(
            "hcml_project/outputs/generated_mad22_webmorph_subset_elasticface_arc/sampling_report.csv"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a qualitative result figure.")
    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
        help="Trial index to use from each method trial CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qualitative_examples.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=150,
        help="Square image tile size in pixels.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def existing_path(path: Path) -> Path:
    if path.is_file():
        return path
    raise FileNotFoundError(f"Image not found: {path}")


def local_or_extracted(root: Path, path: Path) -> Path:
    extracted = root / path
    if extracted.is_file():
        return extracted
    return existing_path(path)


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((x, y), text, fill="black", font=font, spacing=2)


def select_example(method: dict[str, object], trial_index: int) -> list[tuple[Path, str]]:
    root = method["root"]
    assert isinstance(root, Path)
    trials = read_csv(local_or_extracted(root, method["trials"]))  # type: ignore[arg-type]
    report = read_csv(local_or_extracted(root, method["report"]))  # type: ignore[arg-type]

    info = trials[trial_index % len(trials)]
    trial_id = info["trial_id"]
    rows = [row for row in report if row["trial_id"] == trial_id]
    by_setting = {row["setting"]: row for row in rows}

    return [
        (existing_path(Path(info["known_path"])), f"{method['name']}\nKnown"),
        (existing_path(Path(info["morph_path"])), "Morph"),
        (existing_path(Path(info["hidden_path"])), "Hidden"),
        (local_or_extracted(root, Path(by_setting["negfacediff"]["output_path"])), "NegFaceDiff"),
        (local_or_extracted(root, Path(by_setting["adaptdiff"]["output_path"])), "AdaptDiff"),
    ]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [select_example(method, args.trial_index) for method in METHODS]
    columns = 5
    label_height = 38
    padding = 12
    width = columns * args.tile_size + (columns + 1) * padding
    height = len(rows) * (args.tile_size + label_height + padding) + padding
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)

    for row_index, row_items in enumerate(rows):
        y = padding + row_index * (args.tile_size + label_height + padding)
        for column_index, (path, title) in enumerate(row_items):
            x = padding + column_index * (args.tile_size + padding)
            label(draw, x, y, title)
            figure.paste(fit_image(path, args.tile_size), (x, y + label_height))

    figure.save(args.output)
    print(f"Qualitative figure: {args.output}")


if __name__ == "__main__":
    main()
