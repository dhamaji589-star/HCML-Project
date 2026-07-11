"""Extract identity embeddings from aligned face crops.

This script reads the alignment report, loads a face-recognition model, and
saves one embedding per successfully aligned image.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ARCHITECTURES = {"iresnet18", "iresnet34", "iresnet50", "iresnet100"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 512-d identity embeddings from aligned face crops."
    )
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_aligned.csv"),
        help="Alignment report CSV created by align_manifest_images.py.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path(""),
        help="Path to face-recognition model weights, e.g. ElasticCos.pth.",
    )
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default="iresnet100",
        help="Backbone architecture matching the weights file.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("hcml_project/embeddings/mad22_opencv_smoke_embeddings.npz"),
        help="Compressed NumPy file for extracted embeddings.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_embeddings.csv"),
        help="CSV report mapping images to embedding keys/status.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a report without loading weights or extracting embeddings.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def read_alignment_rows(alignment_csv: Path) -> list[dict[str, str]]:
    if not alignment_csv.is_file():
        raise FileNotFoundError(f"Alignment CSV not found: {alignment_csv}")

    with alignment_csv.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def successful_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("alignment_status") == "success"]


def import_backbone(architecture: str):
    backbone_root = Path("NegFaceDiff/face_recognition_training").resolve()
    sys.path.insert(0, str(backbone_root))
    from backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100

    return {
        "iresnet18": iresnet18,
        "iresnet34": iresnet34,
        "iresnet50": iresnet50,
        "iresnet100": iresnet100,
    }[architecture]


def load_model(architecture: str, weights_path: Path, device: torch.device) -> torch.nn.Module:
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "Use --dry-run until you have ElasticCos.pth or compatible weights."
        )

    model_class = import_backbone(architecture)
    model = model_class(num_features=512, dropout=0.0)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (112, 112):
        image = image.resize((112, 112), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def extract_embeddings(
    rows: list[dict[str, str]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.stack(
            [load_image_tensor(Path(row["aligned_path"])) for row in batch_rows]
        ).to(device)

        with torch.no_grad():
            features = F.normalize(model(batch), dim=1)

        for row, feature in zip(batch_rows, features.cpu().numpy()):
            key = f"image_{int(row['image_index']):06d}"
            embeddings[key] = feature.astype(np.float32)

    return embeddings


def write_report(
    rows: list[dict[str, str]],
    report_csv: Path,
    extracted_keys: set[str],
    dry_run: bool,
) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_index",
        "image_path",
        "aligned_path",
        "image_type",
        "roles",
        "identity_ids",
        "embedding_key",
        "embedding_status",
    ]

    with report_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = f"image_{int(row['image_index']):06d}"
            if dry_run:
                status = "dry_run_ready"
            else:
                status = "success" if key in extracted_keys else "not_extracted"

            writer.writerow(
                {
                    "image_index": row["image_index"],
                    "image_path": row["image_path"],
                    "aligned_path": row["aligned_path"],
                    "image_type": row["image_type"],
                    "roles": row["roles"],
                    "identity_ids": row["identity_ids"],
                    "embedding_key": key,
                    "embedding_status": status,
                }
            )


def main() -> None:
    args = parse_args()
    rows = read_alignment_rows(args.alignment_csv)
    rows = successful_rows(rows)

    if not rows:
        raise RuntimeError("No successful aligned images found in alignment CSV.")

    if args.dry_run:
        write_report(rows, args.report_csv, extracted_keys=set(), dry_run=True)
        print(f"Alignment CSV: {args.alignment_csv}")
        print(f"Successful aligned images: {len(rows)}")
        print("Dry run: embeddings were not extracted.")
        print(f"Embedding report CSV: {args.report_csv}")
        return

    device = choose_device(args.device)
    model = load_model(args.architecture, args.weights_path, device)
    embeddings = extract_embeddings(rows, model, device, args.batch_size)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **embeddings)
    write_report(rows, args.report_csv, set(embeddings), dry_run=False)

    print(f"Alignment CSV: {args.alignment_csv}")
    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Embeddings extracted: {len(embeddings)}")
    print(f"Embedding NPZ: {args.output_npz}")
    print(f"Embedding report CSV: {args.report_csv}")


if __name__ == "__main__":
    main()
