"""Prepare paired positive/negative contexts for NegFaceDiff-style sampling.

For this project, every directed trial needs:

- positive context: embedding of the morph image M_AB
- negative context: embedding of the known identity A
- hidden context: embedding of the hidden target B, used only for evaluation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SETTINGS = [
    {
        "name": "negfacediff",
        "description": "Supervisor setting for NegFaceDiff",
        "weight": 0.5,
        "adapt": False,
    },
    {
        "name": "adaptdiff",
        "description": "Supervisor setting for AdaptDiff",
        "weight": 1.0,
        "adapt": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired sampling context files from MAD22 trial embeddings."
    )
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_trials.csv"),
        help="Directed trial CSV.",
    )
    parser.add_argument(
        "--embedding-report-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_embeddings.csv"),
        help="CSV that maps image paths to embedding keys.",
    )
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path("hcml_project/embeddings/mad22_opencv_smoke_embeddings.npz"),
        help="NPZ file containing image embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hcml_project/sampling_inputs/mad22_opencv_smoke"),
        help="Directory where sampling input files will be written.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding NPZ not found: {path}")

    data = np.load(path)
    return {key: data[key].astype(np.float32) for key in data.files}


def normalize_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


def build_image_lookup(report_rows: list[dict[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in report_rows:
        if row["embedding_status"] == "success":
            lookup[normalize_path(row["image_path"])] = row["embedding_key"]
    return lookup


def read_embedding_model_name(report_rows: list[dict[str, str]]) -> str:
    names = {
        row.get("embedding_model_name", "unknown")
        for row in report_rows
        if row["embedding_status"] == "success"
    }
    if len(names) != 1:
        raise ValueError(
            "Expected exactly one embedding model name in the embedding report, "
            f"found: {sorted(names)}"
        )
    return names.pop()


def require_embedding(
    image_lookup: dict[str, str],
    embeddings: dict[str, np.ndarray],
    image_path: str,
) -> tuple[str, np.ndarray]:
    normalized_path = normalize_path(image_path)
    if normalized_path not in image_lookup:
        raise KeyError(f"No embedding key found for image path: {image_path}")

    key = image_lookup[normalized_path]
    if key not in embeddings:
        raise KeyError(f"Embedding key {key} is missing from the NPZ file.")

    return key, embeddings[key]


def build_context_rows(
    trial_rows: list[dict[str, str]],
    image_lookup: dict[str, str],
    embeddings: dict[str, np.ndarray],
    embedding_model_name: str,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    manifest_rows: list[dict[str, str]] = []
    positive_contexts = []
    negative_contexts = []
    hidden_contexts = []

    for context_id, row in enumerate(trial_rows):
        positive_key, positive = require_embedding(
            image_lookup, embeddings, row["morph_path"]
        )
        negative_key, negative = require_embedding(
            image_lookup, embeddings, row["known_path"]
        )
        hidden_key, hidden = require_embedding(
            image_lookup, embeddings, row["hidden_path"]
        )

        positive_contexts.append(positive)
        negative_contexts.append(negative)
        hidden_contexts.append(hidden)

        manifest_rows.append(
            {
                "context_id": str(context_id),
                "trial_id": row["trial_id"],
                "morph_method": row["morph_method"],
                "direction": row["direction"],
                "known_id": row["known_id"],
                "hidden_id": row["hidden_id"],
                "morph_path": row["morph_path"],
                "known_path": row["known_path"],
                "hidden_path": row["hidden_path"],
                "positive_embedding_key": positive_key,
                "negative_embedding_key": negative_key,
                "hidden_embedding_key": hidden_key,
                "embedding_model_name": embedding_model_name,
            }
        )

    return (
        manifest_rows,
        np.stack(positive_contexts).astype(np.float32),
        np.stack(negative_contexts).astype(np.float32),
        np.stack(hidden_contexts).astype(np.float32),
    )


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "context_id",
        "trial_id",
        "morph_method",
        "direction",
        "known_id",
        "hidden_id",
        "morph_path",
        "known_path",
        "hidden_path",
        "positive_embedding_key",
        "negative_embedding_key",
        "hidden_embedding_key",
        "embedding_model_name",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_settings(output_dir: Path, embedding_model_name: str) -> None:
    for setting in SETTINGS:
        path = output_dir / f"{setting['name']}_sampling_settings.yaml"
        text = "\n".join(
            [
                f"name: {setting['name']}",
                f"description: {setting['description']}",
                f"weight: {setting['weight']}",
                f"adapt: {str(setting['adapt']).lower()}",
                "reverse_adapt: false",
                "positive_context: morph_embedding",
                "negative_context: known_identity_embedding",
                "hidden_context: hidden_identity_embedding_for_evaluation_only",
                f"embedding_model_name: {embedding_model_name}",
                "",
            ]
        )
        path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    trial_rows = read_csv(args.trials_csv)
    report_rows = read_csv(args.embedding_report_csv)
    embeddings = load_embeddings(args.embedding_npz)
    image_lookup = build_image_lookup(report_rows)
    embedding_model_name = read_embedding_model_name(report_rows)

    manifest_rows, positive_contexts, negative_contexts, hidden_contexts = (
        build_context_rows(trial_rows, image_lookup, embeddings, embedding_model_name)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts_npz = args.output_dir / "paired_contexts.npz"
    manifest_csv = args.output_dir / "paired_contexts_manifest.csv"

    np.savez_compressed(
        contexts_npz,
        positive_contexts=positive_contexts,
        negative_contexts=negative_contexts,
        hidden_contexts=hidden_contexts,
    )
    write_manifest(manifest_rows, manifest_csv)
    write_settings(args.output_dir, embedding_model_name)

    print(f"Trials prepared: {len(manifest_rows)}")
    print(f"Context dimension: {positive_contexts.shape[1]}")
    print(f"Positive contexts shape: {positive_contexts.shape}")
    print(f"Negative contexts shape: {negative_contexts.shape}")
    print(f"Hidden contexts shape: {hidden_contexts.shape}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
