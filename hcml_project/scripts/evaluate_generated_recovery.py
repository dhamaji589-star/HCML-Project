"""Evaluate generated hidden-identity recovery images.

For every generated image, this script computes:

- similarity to the hidden identity
- similarity to the known identity
- directional margin = hidden similarity - known similarity
- threshold-based hidden-identity recovery

The directional metric shows whether generation moved away from the known
identity and toward the hidden identity. The threshold metric is stricter:
cosine(generated, hidden) must be above a face-recognition threshold.
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
            "hcml_project/model_assets/elasticface/ElasticFaceCos_295672backbone.pth"
        ),
        help="Face-recognition model weights for final generated-image evaluation.",
    )
    parser.add_argument(
        "--evaluation-model-name",
        default="elasticface_cos",
        help="Short name written to logs/CSV for the final evaluation model.",
    )
    parser.add_argument(
        "--positive-pair-threshold",
        type=float,
        default=0.321,
        help=(
            "Cosine threshold for generated/hidden positive pairs. "
            "0.321 is the ElasticFaceCos mean-std genuine threshold reported for CASIA-WebFace."
        ),
    )
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=None,
        help=(
            "Optional alignment report. When provided, known/hidden/morph images "
            "are evaluated from their aligned crops instead of raw image paths."
        ),
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


def normalize_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


def build_aligned_path_lookup(alignment_csv: Path | None) -> dict[str, str]:
    if alignment_csv is None:
        return {}
    rows = read_csv(alignment_csv)
    lookup = {}
    for row in rows:
        if row.get("alignment_status") == "success":
            lookup[normalize_path(row["image_path"])] = row["aligned_path"]
    return lookup


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


def embed_reference_images(
    manifest_rows: list[dict[str, str]],
    aligned_path_lookup: dict[str, str],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    image_paths = []
    for row in manifest_rows:
        image_paths.extend([row["morph_path"], row["known_path"], row["hidden_path"]])

    unique_paths = sorted({normalize_path(path) for path in image_paths})
    embedding_rows = []
    for image_path in unique_paths:
        eval_path = aligned_path_lookup.get(image_path, image_path)
        embedding_rows.append({"path_key": image_path, "output_path": eval_path})

    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(embedding_rows), batch_size):
        batch_rows = embedding_rows[start : start + batch_size]
        batch = torch.stack(
            [load_image_tensor(Path(row["output_path"])) for row in batch_rows]
        ).to(device)
        with torch.no_grad():
            features = F.normalize(model(batch), dim=1)
        for row, feature in zip(batch_rows, features.cpu().numpy()):
            embeddings[row["path_key"]] = feature.astype(np.float32)
    return embeddings


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_manifest_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["trial_id"]: row for row in rows}


def evaluate(
    sampling_rows: list[dict[str, str]],
    manifest_lookup: dict[str, dict[str, str]],
    generated_embeddings: dict[str, np.ndarray],
    reference_embeddings: dict[str, np.ndarray],
    positive_pair_threshold: float,
    evaluation_model_name: str,
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
        known = reference_embeddings[normalize_path(manifest_row["known_path"])]
        hidden = reference_embeddings[normalize_path(manifest_row["hidden_path"])]
        morph = reference_embeddings[normalize_path(manifest_row["morph_path"])]

        sim_known = cosine(generated, known)
        sim_hidden = cosine(generated, hidden)
        sim_morph_hidden = cosine(morph, hidden)
        directional_margin = sim_hidden - sim_known
        generated_minus_morph_hidden = sim_hidden - sim_morph_hidden
        positive_pair = sim_hidden >= positive_pair_threshold

        results.append(
            {
                "trial_id": trial_id,
                "setting": setting,
                "known_id": manifest_row["known_id"],
                "hidden_id": manifest_row["hidden_id"],
                "generated_path": row["output_path"],
                "evaluation_model_name": evaluation_model_name,
                "positive_pair_threshold": f"{positive_pair_threshold:.6f}",
                "sim_generated_known": f"{sim_known:.6f}",
                "sim_generated_hidden": f"{sim_hidden:.6f}",
                "sim_morph_hidden": f"{sim_morph_hidden:.6f}",
                "margin_hidden_minus_known": f"{directional_margin:.6f}",
                "margin_generated_minus_morph_hidden": f"{generated_minus_morph_hidden:.6f}",
                "closer_to_hidden": str(sim_hidden > sim_known),
                "hidden_positive_pair": str(positive_pair),
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
        "evaluation_model_name",
        "positive_pair_threshold",
        "sim_generated_known",
        "sim_generated_hidden",
        "sim_morph_hidden",
        "margin_hidden_minus_known",
        "margin_generated_minus_morph_hidden",
        "closer_to_hidden",
        "hidden_positive_pair",
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
        directional_recoveries = sum(
            row["closer_to_hidden"] == "True" for row in setting_rows
        )
        threshold_recoveries = sum(
            row["hidden_positive_pair"] == "True" for row in setting_rows
        )
        hidden_sims = [float(row["sim_generated_hidden"]) for row in setting_rows]
        margins = [float(row["margin_hidden_minus_known"]) for row in setting_rows]
        morph_margins = [
            float(row["margin_generated_minus_morph_hidden"]) for row in setting_rows
        ]
        mean_hidden_sim = sum(hidden_sims) / len(hidden_sims) if hidden_sims else 0.0
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        mean_morph_margin = (
            sum(morph_margins) / len(morph_margins) if morph_margins else 0.0
        )
        print()
        print(f"{setting}")
        print(f"  Directional recoveries: {directional_recoveries}/{total}")
        print(f"  Threshold recoveries: {threshold_recoveries}/{total}")
        print(f"  Mean cos(generated, hidden): {mean_hidden_sim:.6f}")
        print(f"  Mean directional margin hidden-known: {mean_margin:.6f}")
        print(f"  Mean generated-hidden minus morph-hidden: {mean_morph_margin:.6f}")
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
    if args.contexts_npz.is_file():
        load_contexts(args.contexts_npz)

    model = load_model(args.architecture, args.weights_path, device)
    generated_embeddings = embed_generated_images(
        sampling_rows, model, device, args.batch_size
    )
    aligned_path_lookup = build_aligned_path_lookup(args.alignment_csv)
    reference_embeddings = embed_reference_images(
        manifest_rows, aligned_path_lookup, model, device, args.batch_size
    )

    results = evaluate(
        sampling_rows,
        manifest_lookup,
        generated_embeddings,
        reference_embeddings,
        args.positive_pair_threshold,
        args.evaluation_model_name,
    )
    write_results(results, args.output_csv)
    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Evaluation model: {args.evaluation_model_name}")
    print(f"Positive-pair threshold: {args.positive_pair_threshold:.6f}")
    if args.alignment_csv is not None:
        print(f"Reference alignment CSV: {args.alignment_csv}")
    print_summary(results, args.output_csv)


if __name__ == "__main__":
    main()
