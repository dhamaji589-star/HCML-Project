"""Evaluate generated hidden-identity recovery images.

For every generated image, this script computes:

- similarity to the hidden identity
- similarity to the known identity
- margin = hidden similarity - known similarity

Positive margin means the generated image is closer to the hidden identity.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ARCHITECTURES = {"iresnet18", "iresnet34", "iresnet50", "iresnet100"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated NegFaceDiff/AdaptDiff outputs."
    )
    parser.add_argument(
        "--sampling-report-csv",
        type=Path,
        default=Path("hcml_project/outputs/generated_smoke_elasticface_arc/sampling_report.csv"),
        help="CSV written by sample_paired_diffusion.py.",
    )
    parser.add_argument(
        "--contexts-npz",
        type=Path,
        default=Path(
            "hcml_project/sampling_inputs/mad22_opencv_smoke_elasticface_arc/paired_contexts.npz"
        ),
        help="Paired context NPZ from prepare_sampling_inputs.py.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path(
            "hcml_project/sampling_inputs/mad22_opencv_smoke_elasticface_arc/paired_contexts_manifest.csv"
        ),
        help="Paired context manifest from prepare_sampling_inputs.py.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path(
            "hcml_project/model_assets/elasticface/ElasticFaceArc_295672backbone.pth"
        ),
        help="Face-recognition model weights for generated-image evaluation.",
    )
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default="iresnet100",
        help="Face-recognition backbone architecture.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generated-image embedding batch size.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "hcml_project/metadata/mad22_opencv_smoke_generated_recovery_eval.csv"
        ),
        help="Evaluation CSV to write.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_contexts(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Contexts NPZ not found: {path}")
    data = np.load(path)
    return {key: data[key].astype(np.float32) for key in data.files}


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


def load_state_dict(weights_path: Path, device: torch.device):
    if weights_path.suffix.lower() != ".zip":
        return torch.load(weights_path, map_location=device)

    with zipfile.ZipFile(weights_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".pth", ".pt", ".ckpt"))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one model file inside {weights_path}, found {len(candidates)}."
            )
        return torch.load(io.BytesIO(archive.read(candidates[0])), map_location=device)


def load_model(
    architecture: str, weights_path: Path, device: torch.device
) -> torch.nn.Module:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model_class = import_backbone(architecture)
    model = model_class(num_features=512, dropout=0.0)
    model.load_state_dict(load_state_dict(weights_path, device))
    model.to(device)
    model.eval()
    return model


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (112, 112):
        image = image.resize((112, 112), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1)


def embed_generated_images(
    rows: list[dict[str, str]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.stack(
            [load_image_tensor(Path(row["output_path"])) for row in batch_rows]
        ).to(device)
        with torch.no_grad():
            features = F.normalize(model(batch), dim=1)
        for row, feature in zip(batch_rows, features.cpu().numpy()):
            key = f"{row['setting']}::{row['trial_id']}"
            embeddings[key] = feature.astype(np.float32)
    return embeddings


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_manifest_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["trial_id"]: row for row in rows}


def evaluate(
    sampling_rows: list[dict[str, str]],
    manifest_lookup: dict[str, dict[str, str]],
    contexts: dict[str, np.ndarray],
    generated_embeddings: dict[str, np.ndarray],
) -> list[dict[str, str]]:
    results = []
    for row in sampling_rows:
        if row.get("status") != "generated":
            continue

        trial_id = row["trial_id"]
        setting = row["setting"]
        manifest_row = manifest_lookup[trial_id]
        context_id = int(manifest_row["context_id"])

        generated = generated_embeddings[f"{setting}::{trial_id}"]
        known = contexts["negative_contexts"][context_id]
        hidden = contexts["hidden_contexts"][context_id]

        sim_known = cosine(generated, known)
        sim_hidden = cosine(generated, hidden)
        margin = sim_hidden - sim_known

        results.append(
            {
                "trial_id": trial_id,
                "setting": setting,
                "known_id": manifest_row["known_id"],
                "hidden_id": manifest_row["hidden_id"],
                "generated_path": row["output_path"],
                "sim_generated_known": f"{sim_known:.6f}",
                "sim_generated_hidden": f"{sim_hidden:.6f}",
                "margin_hidden_minus_known": f"{margin:.6f}",
                "closer_to_hidden": str(sim_hidden > sim_known),
            }
        )
    return results


def write_results(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "setting",
        "known_id",
        "hidden_id",
        "generated_path",
        "sim_generated_known",
        "sim_generated_hidden",
        "margin_hidden_minus_known",
        "closer_to_hidden",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_csv: Path) -> None:
    by_setting: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_setting[row["setting"]].append(row)

    print(f"Generated images evaluated: {len(rows)}")
    for setting, setting_rows in sorted(by_setting.items()):
        total = len(setting_rows)
        successes = sum(row["closer_to_hidden"] == "True" for row in setting_rows)
        margins = [float(row["margin_hidden_minus_known"]) for row in setting_rows]
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        print()
        print(f"{setting}")
        print(f"  Successes: {successes}/{total}")
        print(f"  Mean margin hidden-known: {mean_margin:.6f}")
    print()
    print(f"Output CSV: {output_csv}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    sampling_rows = [
        row for row in read_csv(args.sampling_report_csv) if row.get("status") == "generated"
    ]
    if not sampling_rows:
        raise RuntimeError("No generated rows found in sampling report.")

    manifest_rows = read_csv(args.manifest_csv)
    manifest_lookup = build_manifest_lookup(manifest_rows)
    contexts = load_contexts(args.contexts_npz)

    model = load_model(args.architecture, args.weights_path, device)
    generated_embeddings = embed_generated_images(
        sampling_rows, model, device, args.batch_size
    )

    results = evaluate(sampling_rows, manifest_lookup, contexts, generated_embeddings)
    write_results(results, args.output_csv)
    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print_summary(results, args.output_csv)


if __name__ == "__main__":
    main()
