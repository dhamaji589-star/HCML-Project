"""Split the CompVis FFHQ256 LDM checkpoint into NegFaceDiff autoencoder files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ENCODER_PREFIXES = (
    "first_stage_model.encoder.",
    "first_stage_model.quant_conv.",
)

DECODER_PREFIXES = (
    "first_stage_model.decoder.",
    "first_stage_model.quantize.",
    "first_stage_model.post_quant_conv.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create first_stage_encoder/decoder state dicts from FFHQ LDM model.ckpt."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("hcml_project/model_assets/ffhq_ldm/model.ckpt"),
        help="CompVis FFHQ256 latent diffusion checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("NegFaceDiff/models/autoencoder"),
        help="Directory where NegFaceDiff expects autoencoder files.",
    )
    return parser.parse_args()


def strip_first_stage_prefix(key: str) -> str:
    return key.removeprefix("first_stage_model.")


def select_keys(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    selected = {}
    for key, value in state_dict.items():
        if key.startswith(prefixes):
            selected[strip_first_stage_prefix(key)] = value
    return selected


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    encoder_state = select_keys(state_dict, ENCODER_PREFIXES)
    decoder_state = select_keys(state_dict, DECODER_PREFIXES)

    if not encoder_state:
        raise RuntimeError("No encoder keys were found in the checkpoint.")
    if not decoder_state:
        raise RuntimeError("No decoder keys were found in the checkpoint.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_dir / "first_stage_encoder_state_dict.pt"
    decoder_path = args.output_dir / "first_stage_decoder_state_dict.pt"

    torch.save(encoder_state, encoder_path)
    torch.save(decoder_state, decoder_path)

    print(f"Encoder keys: {len(encoder_state)}")
    print(f"Decoder keys: {len(decoder_state)}")
    print(f"Encoder output: {encoder_path}")
    print(f"Decoder output: {decoder_path}")


if __name__ == "__main__":
    main()
