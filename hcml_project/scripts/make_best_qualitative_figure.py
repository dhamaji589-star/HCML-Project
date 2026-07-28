"""Create a qualitative figure from high-scoring generated examples."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCES = {
    "OpenCV": [
        (
            Path("results/extracted/opencv"),
            Path("hcml_project/metadata/mad22_opencv_smoke_trials.csv"),
            Path(
                "results/extracted/opencv/hcml_project/metadata/"
                "mad22_opencv_smoke_generated_recovery_eval_original_schedule_20.csv"
            ),
        ),
        (
            Path("results/extracted_subset2/opencv"),
            Path("results/extracted_subset2/opencv/hcml_project/metadata/mad22_opencv_subset2_trials.csv"),
            Path("results/extracted_subset2/opencv/hcml_project/metadata/mad22_opencv_subset2_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset3/opencv"),
            Path("results/extracted_subset3/opencv/hcml_project/metadata/mad22_opencv_subset3_trials.csv"),
            Path("results/extracted_subset3/opencv/hcml_project/metadata/mad22_opencv_subset3_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset4/opencv"),
            Path("results/extracted_subset4/opencv/hcml_project/metadata/mad22_opencv_subset4_trials.csv"),
            Path("results/extracted_subset4/opencv/hcml_project/metadata/mad22_opencv_subset4_generated_recovery_eval.csv"),
        ),
    ],
    "FaceMorpher": [
        (
            Path("results/extracted/facemorpher"),
            Path("results/extracted/facemorpher/hcml_project/metadata/mad22_facemorpher_subset_trials.csv"),
            Path("results/extracted/facemorpher/hcml_project/metadata/mad22_facemorpher_subset_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset2/facemorpher"),
            Path("results/extracted_subset2/facemorpher/hcml_project/metadata/mad22_facemorpher_subset2_trials.csv"),
            Path("results/extracted_subset2/facemorpher/hcml_project/metadata/mad22_facemorpher_subset2_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset3/facemorpher"),
            Path("results/extracted_subset3/facemorpher/hcml_project/metadata/mad22_facemorpher_subset3_trials.csv"),
            Path("results/extracted_subset3/facemorpher/hcml_project/metadata/mad22_facemorpher_subset3_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset4/facemorpher"),
            Path("results/extracted_subset4/facemorpher/hcml_project/metadata/mad22_facemorpher_subset4_trials.csv"),
            Path("results/extracted_subset4/facemorpher/hcml_project/metadata/mad22_facemorpher_subset4_generated_recovery_eval.csv"),
        ),
    ],
    "MIPGAN-I": [
        (
            Path("results/extracted/mipgan_i"),
            Path("results/extracted/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset_trials.csv"),
            Path("results/extracted/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset2/mipgan_i"),
            Path("results/extracted_subset2/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset2_trials.csv"),
            Path("results/extracted_subset2/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset2_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset3/mipgan_i"),
            Path("results/extracted_subset3/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset3_trials.csv"),
            Path("results/extracted_subset3/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset3_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset4/mipgan_i"),
            Path("results/extracted_subset4/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset4_trials.csv"),
            Path("results/extracted_subset4/mipgan_i/hcml_project/metadata/mad22_mipgan_i_subset4_generated_recovery_eval.csv"),
        ),
    ],
    "MIPGAN-II": [
        (
            Path("results/extracted/mipgan_ii"),
            Path("results/extracted/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset_trials.csv"),
            Path("results/extracted/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset2/mipgan_ii"),
            Path("results/extracted_subset2/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset2_trials.csv"),
            Path("results/extracted_subset2/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset2_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset3/mipgan_ii"),
            Path("results/extracted_subset3/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset3_trials.csv"),
            Path("results/extracted_subset3/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset3_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset4/mipgan_ii"),
            Path("results/extracted_subset4/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset4_trials.csv"),
            Path("results/extracted_subset4/mipgan_ii/hcml_project/metadata/mad22_mipgan_ii_subset4_generated_recovery_eval.csv"),
        ),
    ],
    "WebMorph": [
        (
            Path("results/extracted/webmorph"),
            Path("results/extracted/webmorph/hcml_project/metadata/mad22_webmorph_subset_trials.csv"),
            Path("results/extracted/webmorph/hcml_project/metadata/mad22_webmorph_subset_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset2/webmorph"),
            Path("results/extracted_subset2/webmorph/hcml_project/metadata/mad22_webmorph_subset2_trials.csv"),
            Path("results/extracted_subset2/webmorph/hcml_project/metadata/mad22_webmorph_subset2_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset3/webmorph"),
            Path("results/extracted_subset3/webmorph/hcml_project/metadata/mad22_webmorph_subset3_trials.csv"),
            Path("results/extracted_subset3/webmorph/hcml_project/metadata/mad22_webmorph_subset3_generated_recovery_eval.csv"),
        ),
        (
            Path("results/extracted_subset4/webmorph"),
            Path("results/extracted_subset4/webmorph/hcml_project/metadata/mad22_webmorph_subset4_trials.csv"),
            Path("results/extracted_subset4/webmorph/hcml_project/metadata/mad22_webmorph_subset4_generated_recovery_eval.csv"),
        ),
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def resolve_image(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    extracted = root / path
    if extracted.is_file():
        return extracted
    if path.is_file():
        return path
    raise FileNotFoundError(path_text)


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def select_best(method: str) -> tuple[Path, dict[str, str], dict[str, dict[str, str]]]:
    candidates = []
    for root, trials_path, eval_path in SOURCES[method]:
        trials = {row["trial_id"]: row for row in read_csv(trials_path)}
        rows = read_csv(eval_path)
        grouped: dict[str, dict[str, str]] = {}
        for row in rows:
            grouped.setdefault(row["trial_id"], {})[row["setting"]] = row
        for trial_id, by_setting in grouped.items():
            adapt = by_setting.get("adaptdiff")
            neg = by_setting.get("negfacediff")
            if not adapt or not neg or trial_id not in trials:
                continue
            if adapt["closer_to_hidden"].strip().lower() != "true":
                continue
            candidates.append((float(adapt["margin_hidden_minus_known"]), root, trials[trial_id], by_setting))
    if not candidates:
        raise RuntimeError(f"No successful AdaptDiff examples found for {method}")
    _margin, root, trial, by_setting = max(candidates, key=lambda item: item[0])
    return root, trial, by_setting


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont) -> None:
    draw.multiline_text((x, y), text, fill="black", font=font, spacing=2)


def main() -> None:
    output = Path("results/qualitative_best_examples.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    tile_size = 170
    label_height = 50
    padding = 12
    columns = 5
    rows = len(SOURCES)
    width = columns * tile_size + (columns + 1) * padding
    height = rows * (tile_size + label_height + padding) + padding
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for row_index, method in enumerate(SOURCES):
        root, trial, by_setting = select_best(method)
        items = [
            (Path(trial["known_path"]), f"{method}\nKnown"),
            (Path(trial["morph_path"]), "Morph"),
            (Path(trial["hidden_path"]), "Hidden"),
            (
                Path(by_setting["negfacediff"]["generated_path"]),
                "NegFaceDiff\nm={:.3f}".format(float(by_setting["negfacediff"]["margin_hidden_minus_known"])),
            ),
            (
                Path(by_setting["adaptdiff"]["generated_path"]),
                "AdaptDiff\nm={:.3f}".format(float(by_setting["adaptdiff"]["margin_hidden_minus_known"])),
            ),
        ]
        y = padding + row_index * (tile_size + label_height + padding)
        for column_index, (path, title) in enumerate(items):
            x = padding + column_index * (tile_size + padding)
            draw_label(draw, x, y, title, font)
            figure.paste(fit_image(resolve_image(root, str(path)), tile_size), (x, y + label_height))

    figure.save(output)
    print(f"Qualitative figure: {output}")


if __name__ == "__main__":
    main()
