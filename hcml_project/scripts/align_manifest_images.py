"""Align/crop images listed in an image manifest.

This script supports two modes:

1. center_crop
   Simple local smoke-test mode. It does not need extra ML dependencies.

2. mtcnn
   Proper face-landmark alignment mode for Kaggle/GPU runs. It uses MTCNN
   landmarks and an ArcFace-style 112x112 crop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ARCFACE_REFERENCE_POINTS = [
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create aligned face crops from an image manifest."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_images.csv"),
        help="Image manifest created by build_image_manifest.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hcml_project/outputs/aligned_smoke"),
        help="Directory for aligned image crops.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_aligned.csv"),
        help="CSV report with alignment status and output paths.",
    )
    parser.add_argument(
        "--detector",
        choices=["center_crop", "mtcnn"],
        default="center_crop",
        help="Use center crop for local smoke tests or MTCNN for real alignment.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Output crop size. Use 112 for ArcFace/ElasticFace embeddings.",
    )
    return parser.parse_args()


def read_manifest(manifest_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    with manifest_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"Manifest CSV has no header: {manifest_csv}")

    return rows, fieldnames


def center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    from PIL import Image

    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped = image.crop((left, top, left + crop_size, top + crop_size))
    return cropped.resize((image_size, image_size), Image.Resampling.BILINEAR)


def arcface_align_rgb(image_rgb: np.ndarray, landmarks: np.ndarray, image_size: int) -> Image.Image:
    import numpy as np
    import cv2
    from PIL import Image

    if image_size != 112:
        raise ValueError("ArcFace landmark alignment currently expects --image-size 112")

    landmarks = landmarks.astype(np.float32)
    reference_points = np.asarray(ARCFACE_REFERENCE_POINTS, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks,
        reference_points,
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError("Could not estimate landmark transform")

    aligned_rgb = cv2.warpAffine(
        image_rgb,
        matrix,
        (image_size, image_size),
        borderValue=0.0,
    )
    return Image.fromarray(aligned_rgb)


def build_mtcnn() -> Any:
    try:
        import torch
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError(
            "MTCNN alignment requires torch and facenet-pytorch. "
            "Use --detector center_crop locally, or install dependencies on Kaggle."
        ) from exc

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return MTCNN(select_largest=True, post_process=False, device=device)


def align_with_mtcnn(image: Image.Image, mtcnn: Any, image_size: int) -> Image.Image:
    import numpy as np

    image_rgb = np.asarray(image.convert("RGB"))
    _, _, landmarks = mtcnn.detect(image, landmarks=True)
    if landmarks is None or len(landmarks) == 0:
        raise ValueError("No face landmarks detected")

    return arcface_align_rgb(image_rgb, np.asarray(landmarks[0]), image_size)


def safe_output_name(row: dict[str, str]) -> str:
    image_index = int(row["image_index"])
    source_stem = Path(row["image_path"]).stem
    image_type = row["image_type"]
    return f"{image_index:06d}_{image_type}_{source_stem}.png"


def align_rows(
    rows: list[dict[str, str]],
    output_dir: Path,
    detector: str,
    image_size: int,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mtcnn = build_mtcnn() if detector == "mtcnn" else None

    report_rows: list[dict[str, str]] = []
    for row in rows:
        output_path = output_dir / safe_output_name(row)
        report_row = dict(row)
        report_row["aligned_path"] = output_path.as_posix()
        report_row["alignment_method"] = detector

        try:
            from PIL import Image

            image = Image.open(row["image_path"]).convert("RGB")
            if detector == "center_crop":
                aligned = center_crop_resize(image, image_size)
            else:
                aligned = align_with_mtcnn(image, mtcnn, image_size)

            aligned.save(output_path)
            report_row["alignment_status"] = "success"
            report_row["alignment_error"] = ""
        except Exception as exc:  # Keep going and report failures for later review.
            report_row["alignment_status"] = "failed"
            report_row["alignment_error"] = str(exc)

        report_rows.append(report_row)

    return report_rows


def write_report(report_rows: list[dict[str, str]], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(report_rows[0].keys()) if report_rows else []
    with report_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    args = parse_args()
    rows, _ = read_manifest(args.manifest_csv)
    report_rows = align_rows(
        rows=rows,
        output_dir=args.output_dir,
        detector=args.detector,
        image_size=args.image_size,
    )
    write_report(report_rows, args.report_csv)

    success_count = sum(row["alignment_status"] == "success" for row in report_rows)
    failed_count = len(report_rows) - success_count

    print(f"Manifest CSV: {args.manifest_csv}")
    print(f"Detector: {args.detector}")
    print(f"Images processed: {len(report_rows)}")
    print(f"Successful alignments: {success_count}")
    print(f"Failed alignments: {failed_count}")
    print(f"Aligned image directory: {args.output_dir}")
    print(f"Alignment report CSV: {args.report_csv}")


if __name__ == "__main__":
    main()
