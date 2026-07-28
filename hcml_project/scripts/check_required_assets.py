"""Check whether the project model assets needed for final experiments exist."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RequiredAsset:
    label: str
    path: Path
    purpose: str
    alternate_paths: list[Path] = field(default_factory=list)


REQUIRED_ASSETS = [
    RequiredAsset(
        label="DM_CASIA config",
        path=Path("DM_CASIA_cpd25/.hydra/config.yaml"),
        purpose="Needed to instantiate the pretrained CASIA diffusion model.",
        alternate_paths=[
            Path("DM_CASIA_cpd25-20260709T144335Z-2-001.zip"),
        ],
    ),
    RequiredAsset(
        label="DM_CASIA EMA checkpoint",
        path=Path("DM_CASIA_cpd25/checkpoints/ema_averaged_model.ckpt"),
        purpose="Main diffusion model weights used for sampling.",
        alternate_paths=[
            Path("DM_CASIA_cpd25-20260709T132139Z-2-002.zip"),
        ],
    ),
    RequiredAsset(
        label="Latent autoencoder config",
        path=Path("NegFaceDiff/models/autoencoder/first_stage_config.yaml"),
        purpose="Architecture config for the latent autoencoder.",
    ),
    RequiredAsset(
        label="Latent autoencoder encoder",
        path=Path("NegFaceDiff/models/autoencoder/first_stage_encoder_state_dict.pt"),
        purpose="Encodes morph images into latent space.",
    ),
    RequiredAsset(
        label="Latent autoencoder decoder",
        path=Path("NegFaceDiff/models/autoencoder/first_stage_decoder_state_dict.pt"),
        purpose="Decodes generated latents back into face images.",
    ),
    RequiredAsset(
        label="ElasticFaceArc weights",
        path=Path("hcml_project/model_assets/elasticface/ElasticFaceArc_295672backbone.pth"),
        purpose="Final identity context model requested by the supervisor.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check project assets required for NegFaceDiff/AdaptDiff experiments."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero code if any required asset is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = []

    print("Required asset check")
    print("--------------------")

    for asset in REQUIRED_ASSETS:
        alternate_present = [path for path in asset.alternate_paths if path.is_file()]
        if asset.path.is_file():
            status = "OK"
        elif alternate_present:
            status = "ZIP"
        else:
            status = "MISSING"

        print(f"{status:7} {asset.label}")
        print(f"        path: {asset.path}")
        print(f"        why:  {asset.purpose}")
        if status == "ZIP":
            print("        note: downloaded zip exists, but final extracted path is missing")
            for path in alternate_present:
                print(f"              zip: {path}")
        if status != "OK":
            missing.append(asset.label)

    print()
    if missing:
        print(f"Missing assets: {len(missing)}")
        for label in missing:
            print(f"- {label}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("All required assets are present.")


if __name__ == "__main__":
    main()
