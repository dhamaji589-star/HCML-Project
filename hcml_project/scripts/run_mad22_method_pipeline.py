"""Run one MAD22 morphing-method experiment with consistent file names.

This script is a thin orchestrator around the smaller project scripts. It keeps
the per-method outputs separate, which is important because MAD22 morphing
methods should not be merged when reporting results.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


METHODS = ["OpenCV", "FaceMorpher", "MIPGAN_I", "MIPGAN_II", "Webmorph"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing, generation, and evaluation for one MAD22 method."
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
        help="Morphing method folder under MAD22/original_sorted.",
    )
    parser.add_argument(
        "--num-morphs",
        type=int,
        default=10,
        help="Number of unique morph images to evaluate. Each gives two directed trials.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for deterministic non-overlapping subsets.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional output tag, e.g. subset2 or subset3.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "generate", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="How to choose the subset of morphs.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=7,
        help="Seed used only when --selection random.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Seed used for morph noising during diffusion sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for generation: cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Device for ElasticFaceArc embedding/evaluation.",
    )
    parser.add_argument(
        "--detector",
        choices=["center_crop", "mtcnn"],
        default="center_crop",
        help="Alignment detector for ElasticFaceArc embeddings.",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=200,
        help="DDIM steps for sampling.",
    )
    parser.add_argument(
        "--resize-filter",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="lanczos",
        help="Resize filter for morph images before autoencoder encoding.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("hcml_project/model_assets/elasticface/ElasticFaceArc_295672backbone.pth"),
        help="ElasticFaceArc weights path used for identity conditioning.",
    )
    parser.add_argument(
        "--evaluation-weights-path",
        type=Path,
        default=Path("hcml_project/model_assets/elasticface/ElasticFaceCos_295672backbone.pth"),
        help="ElasticFaceCos weights path used for final FR evaluation.",
    )
    parser.add_argument(
        "--positive-pair-threshold",
        type=float,
        default=0.321,
        help="Cosine threshold used for generated/hidden positive-pair recovery.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return sum(1 for _row in csv.DictReader(csv_file))


def run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def paths_for(method: str, output_tag: str) -> dict[str, Path]:
    slug = method_slug(method)
    prefix = f"mad22_{slug}_{output_tag}" if output_tag else f"mad22_{slug}_subset"
    return {
        "trials": Path(f"hcml_project/metadata/mad22_{slug}_trials.csv"),
        "skipped": Path(f"hcml_project/metadata/mad22_{slug}_skipped.csv"),
        "subset_trials": Path(f"hcml_project/metadata/{prefix}_trials.csv"),
        "images": Path(f"hcml_project/metadata/{prefix}_images.csv"),
        "aligned": Path(f"hcml_project/metadata/{prefix}_aligned.csv"),
        "embeddings": Path(f"hcml_project/embeddings/{prefix}_elasticface_arc.npz"),
        "embedding_report": Path(f"hcml_project/metadata/{prefix}_elasticface_arc_embeddings.csv"),
        "sampling_inputs": Path(f"hcml_project/sampling_inputs/{prefix}_elasticface_arc"),
        "outputs": Path(f"hcml_project/outputs/generated_{prefix}_elasticface_arc"),
        "eval": Path(f"hcml_project/metadata/{prefix}_generated_recovery_eval.csv"),
        "aligned_dir": Path(f"hcml_project/outputs/aligned_{prefix}"),
    }


def prepare_commands(args: argparse.Namespace, paths: dict[str, Path]) -> list[list[str]]:
    return [
        [
            sys.executable,
            "hcml_project/scripts/build_mad22_metadata.py",
            "--method",
            args.method,
            "--output-csv",
            str(paths["trials"]),
            "--skipped-csv",
            str(paths["skipped"]),
        ],
        [
            sys.executable,
            "hcml_project/scripts/create_smoke_subset.py",
            "--input-csv",
            str(paths["trials"]),
            "--output-csv",
            str(paths["subset_trials"]),
            "--num-morphs",
            str(args.num_morphs),
            "--start-index",
            str(args.start_index),
            "--selection",
            args.selection,
            "--seed",
            str(args.subset_seed),
        ],
        [
            sys.executable,
            "hcml_project/scripts/build_image_manifest.py",
            "--trials-csv",
            str(paths["subset_trials"]),
            "--output-csv",
            str(paths["images"]),
        ],
        [
            sys.executable,
            "hcml_project/scripts/align_manifest_images.py",
            "--manifest-csv",
            str(paths["images"]),
            "--output-dir",
            str(paths["aligned_dir"]),
            "--report-csv",
            str(paths["aligned"]),
            "--detector",
            args.detector,
        ],
        [
            sys.executable,
            "hcml_project/scripts/extract_identity_embeddings.py",
            "--alignment-csv",
            str(paths["aligned"]),
            "--weights-path",
            str(args.weights_path),
            "--embedding-model-name",
            "elasticface_arc",
            "--architecture",
            "iresnet100",
            "--output-npz",
            str(paths["embeddings"]),
            "--report-csv",
            str(paths["embedding_report"]),
            "--device",
            args.embedding_device,
        ],
        [
            sys.executable,
            "hcml_project/scripts/prepare_sampling_inputs.py",
            "--trials-csv",
            str(paths["subset_trials"]),
            "--embedding-report-csv",
            str(paths["embedding_report"]),
            "--embedding-npz",
            str(paths["embeddings"]),
            "--output-dir",
            str(paths["sampling_inputs"]),
        ],
    ]


def generate_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    manifest = paths["sampling_inputs"] / "paired_contexts_manifest.csv"
    max_trials = args.num_morphs * 2 if args.dry_run else count_csv_rows(manifest)
    return [
        sys.executable,
        "hcml_project/scripts/sample_paired_diffusion.py",
        "--contexts-npz",
        str(paths["sampling_inputs"] / "paired_contexts.npz"),
        "--manifest-csv",
        str(manifest),
        "--output-dir",
        str(paths["outputs"]),
        "--max-trials",
        str(max_trials),
        "--setting",
        "both",
        "--ddim-steps",
        str(args.ddim_steps),
        "--resize-filter",
        args.resize_filter,
        "--seed",
        str(args.sampling_seed),
        "--device",
        args.device,
    ]


def evaluate_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "hcml_project/scripts/evaluate_generated_recovery.py",
        "--sampling-report-csv",
        str(paths["outputs"] / "sampling_report.csv"),
        "--contexts-npz",
        str(paths["sampling_inputs"] / "paired_contexts.npz"),
        "--manifest-csv",
        str(paths["sampling_inputs"] / "paired_contexts_manifest.csv"),
        "--weights-path",
        str(args.evaluation_weights_path),
        "--evaluation-model-name",
        "elasticface_cos",
        "--positive-pair-threshold",
        str(args.positive_pair_threshold),
        "--alignment-csv",
        str(paths["aligned"]),
        "--architecture",
        "iresnet100",
        "--output-csv",
        str(paths["eval"]),
        "--device",
        args.embedding_device,
    ]


def main() -> None:
    args = parse_args()
    paths = paths_for(args.method, args.output_tag)

    print(f"Method: {args.method}")
    print(f"Subset morphs: {args.num_morphs}")
    print(f"Start index: {args.start_index}")
    if args.output_tag:
        print(f"Output tag: {args.output_tag}")
    print(f"Output root: {paths['outputs']}")

    if args.stage in {"prepare", "all"}:
        for command in prepare_commands(args, paths):
            run(command, args.dry_run)

    if args.stage in {"generate", "all"}:
        run(generate_command(args, paths), args.dry_run)

    if args.stage in {"evaluate", "all"}:
        run(evaluate_command(args, paths), args.dry_run)


if __name__ == "__main__":
    main()
