"""Evaluate morph/source similarities using extracted identity embeddings.

This is a pre-diffusion sanity check. It compares each morph embedding against
the known source identity and hidden source identity for every directed trial.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MAD22 directed trials using extracted embeddings."
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
        "--output-csv",
        type=Path,
        default=Path("hcml_project/metadata/mad22_opencv_smoke_baseline_eval.csv"),
        help="Per-trial evaluation CSV to write.",
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


def build_image_lookup(report_rows: list[dict[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in report_rows:
        if row["embedding_status"] == "success":
            lookup[normalize_path(row["image_path"])] = row["embedding_key"]
    return lookup


def build_gallery(
    report_rows: list[dict[str, str]],
    image_lookup: dict[str, str],
    embeddings: dict[str, np.ndarray],
) -> list[tuple[str, str, np.ndarray]]:
    gallery: list[tuple[str, str, np.ndarray]] = []
    for row in report_rows:
        if row["image_type"] != "bona_fide" or row["embedding_status"] != "success":
            continue

        identity_id = row["identity_ids"]
        image_path = normalize_path(row["image_path"])
        key = image_lookup[image_path]
        gallery.append((identity_id, image_path, embeddings[key]))
    return gallery


def normalize_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def hidden_rank(query: np.ndarray, hidden_id: str, gallery: list[tuple[str, str, np.ndarray]]) -> int:
    ranked = sorted(
        gallery,
        key=lambda item: cosine(query, item[2]),
        reverse=True,
    )
    for rank, (identity_id, _image_path, _embedding) in enumerate(ranked, start=1):
        if identity_id == hidden_id:
            return rank
    return 0


def evaluate(
    trial_rows: list[dict[str, str]],
    image_lookup: dict[str, str],
    embeddings: dict[str, np.ndarray],
    gallery: list[tuple[str, str, np.ndarray]],
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    for row in trial_rows:
        morph_key = image_lookup[normalize_path(row["morph_path"])]
        known_key = image_lookup[normalize_path(row["known_path"])]
        hidden_key = image_lookup[normalize_path(row["hidden_path"])]

        morph_embedding = embeddings[morph_key]
        known_embedding = embeddings[known_key]
        hidden_embedding = embeddings[hidden_key]

        sim_known = cosine(morph_embedding, known_embedding)
        sim_hidden = cosine(morph_embedding, hidden_embedding)
        margin_hidden_minus_known = sim_hidden - sim_known
        rank = hidden_rank(morph_embedding, row["hidden_id"], gallery)

        results.append(
            {
                "trial_id": row["trial_id"],
                "morph_method": row["morph_method"],
                "direction": row["direction"],
                "known_id": row["known_id"],
                "hidden_id": row["hidden_id"],
                "sim_morph_known": f"{sim_known:.6f}",
                "sim_morph_hidden": f"{sim_hidden:.6f}",
                "margin_hidden_minus_known": f"{margin_hidden_minus_known:.6f}",
                "closer_to_hidden": str(sim_hidden > sim_known),
                "hidden_rank_in_gallery": str(rank),
            }
        )

    return results


def write_results(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "morph_method",
        "direction",
        "known_id",
        "hidden_id",
        "sim_morph_known",
        "sim_morph_hidden",
        "margin_hidden_minus_known",
        "closer_to_hidden",
        "hidden_rank_in_gallery",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], gallery_size: int, output_csv: Path) -> None:
    total = len(rows)
    closer_to_hidden = sum(row["closer_to_hidden"] == "True" for row in rows)
    ranks = [int(row["hidden_rank_in_gallery"]) for row in rows]
    valid_ranks = [rank for rank in ranks if rank > 0]
    top1 = sum(rank == 1 for rank in valid_ranks)
    top5 = sum(rank <= 5 for rank in valid_ranks)
    mean_rank = sum(valid_ranks) / len(valid_ranks) if valid_ranks else 0.0

    print(f"Trials evaluated: {total}")
    print(f"Gallery identities: {gallery_size}")
    print(f"Closer to hidden: {closer_to_hidden}/{total}")
    print(f"Hidden retrieval top-1: {top1}/{total}")
    print(f"Hidden retrieval top-5: {top5}/{total}")
    print(f"Mean hidden rank: {mean_rank:.2f}")
    print(f"Output CSV: {output_csv}")


def main() -> None:
    args = parse_args()

    trial_rows = read_csv(args.trials_csv)
    report_rows = read_csv(args.embedding_report_csv)
    embeddings = load_embeddings(args.embedding_npz)
    image_lookup = build_image_lookup(report_rows)
    gallery = build_gallery(report_rows, image_lookup, embeddings)

    results = evaluate(trial_rows, image_lookup, embeddings, gallery)
    write_results(results, args.output_csv)
    print_summary(results, len(gallery), args.output_csv)


if __name__ == "__main__":
    main()
