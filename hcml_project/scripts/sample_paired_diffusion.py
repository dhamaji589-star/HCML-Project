"""Run paired NegFaceDiff/AdaptDiff sampling for MAD22 recovery trials.

This script follows the project-specific setup:

- starting noise comes from the morph image, not from random noise
- positive context is the morph identity embedding
- negative context is the known/live identity embedding
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm


NEG_FACE_DIFF_ROOT = Path("NegFaceDiff").resolve()
sys.path.insert(0, str(NEG_FACE_DIFF_ROOT))

from models.autoencoder.vqgan import VQDecoderInterface, VQEncoderInterface
from utils.helpers import denormalize_to_zero_to_one


SETTINGS = {
    "negfacediff": {"weight": 0.5, "adapt": False},
    "adaptdiff": {"weight": 1.0, "adapt": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample recovered faces from paired MAD22 contexts."
    )
    parser.add_argument(
        "--contexts-npz",
        type=Path,
        default=Path(
            "hcml_project/sampling_inputs/mad22_opencv_smoke_elasticface_arc/paired_contexts.npz"
        ),
        help="NPZ file from prepare_sampling_inputs.py.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path(
            "hcml_project/sampling_inputs/mad22_opencv_smoke_elasticface_arc/paired_contexts_manifest.csv"
        ),
        help="Manifest CSV from prepare_sampling_inputs.py.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("DM_CASIA_cpd25"),
        help="Extracted DM_CASIA_cpd25 checkpoint directory.",
    )
    parser.add_argument(
        "--autoencoder-dir",
        type=Path,
        default=Path("NegFaceDiff/models/autoencoder"),
        help="Directory containing first_stage autoencoder files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hcml_project/outputs/generated_smoke_elasticface_arc"),
        help="Directory where generated images and reports are saved.",
    )
    parser.add_argument(
        "--setting",
        choices=["negfacediff", "adaptdiff", "both"],
        default="both",
        help="Which supervisor setting to run.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=1,
        help="Maximum number of trials to run. Keep this small for smoke tests.",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=200,
        help="DDIM reverse sampling steps. Supervisor recommended 200.",
    )
    parser.add_argument(
        "--noise-timestep",
        type=int,
        default=None,
        help=(
            "Forward diffusion timestep used to turn the morph latent into noise. "
            "Default uses the final training timestep, T-1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the forward-noising epsilon.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs and create morph-derived noisy latents without running the diffusion model.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_contexts(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Contexts NPZ not found: {path}")
    data = np.load(path)
    return {key: data[key].astype(np.float32) for key in data.files}


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def load_autoencoder(autoencoder_dir: Path, device: torch.device):
    config = autoencoder_dir / "first_stage_config.yaml"
    encoder_weights = autoencoder_dir / "first_stage_encoder_state_dict.pt"
    decoder_weights = autoencoder_dir / "first_stage_decoder_state_dict.pt"

    latent_encoder = VQEncoderInterface(
        first_stage_config_path=str(config),
        encoder_state_dict_path=str(encoder_weights),
    ).to(device)
    latent_decoder = VQDecoderInterface(
        first_stage_config_path=str(config),
        decoder_state_dict_path=str(decoder_weights),
    ).to(device)

    latent_encoder.eval()
    latent_decoder.eval()
    return latent_encoder, latent_decoder


def load_diffusion_model(checkpoint_dir: Path, device: torch.device):
    train_cfg = OmegaConf.load(checkpoint_dir / ".hydra" / "config.yaml")
    diffusion_model = instantiate(train_cfg.diffusion).to(device)
    checkpoint_path = checkpoint_dir / "checkpoints" / "ema_averaged_model.ckpt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    diffusion_model.load_state_dict(state_dict)
    diffusion_model.eval()
    return diffusion_model, train_cfg


def morph_to_noisy_latent(
    morph_path: Path,
    image_size: int,
    latent_encoder: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    device: torch.device,
    generator: torch.Generator,
    noise_timestep: int | None,
) -> torch.Tensor:
    image = load_image_tensor(morph_path, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = latent_encoder(image)

    # Closed-form q(x_T | x_0). This is equivalent to applying the forward
    # Markovian noising process for all T diffusion steps.
    final_t = diffusion_model.T - 1 if noise_timestep is None else noise_timestep
    if final_t < 0 or final_t >= diffusion_model.T:
        raise ValueError(
            f"--noise-timestep must be between 0 and {diffusion_model.T - 1}; "
            f"got {final_t}"
        )

    epsilon = torch.randn(
        latent.shape,
        generator=generator,
        device=device,
        dtype=latent.dtype,
    )
    return (
        diffusion_model.sqrt_alpha_bars[final_t] * latent
        + diffusion_model.sqrt_one_minus_alpha_bars[final_t] * epsilon
    )


def sample_ddim_from_noisy_latent(
    diffusion_model: torch.nn.Module,
    x_t: torch.Tensor,
    context: torch.Tensor,
    negative_context: torch.Tensor,
    weight: float,
    adapt: bool,
    ddim_steps: int,
    start_timestep: int,
) -> torch.Tensor:
    n_samples = x_t.shape[0]
    device = x_t.device
    final_weight = weight
    timesteps = np.linspace(0, start_timestep, ddim_steps, dtype=np.int64)
    timesteps = np.unique(timesteps)

    with torch.no_grad():
        reversed_timesteps = list(reversed(timesteps.tolist()))
        for step_index, i in enumerate(
            tqdm(reversed_timesteps, total=len(reversed_timesteps), desc="DDIM sampling")
        ):
            t = torch.full((n_samples,), i, dtype=torch.long, device=device)
            eps_pos = diffusion_model.eps_model(x_t, t, context)
            eps_neg = diffusion_model.eps_model(x_t, t, negative_context)

            if adapt:
                final_weight = weight * (1 - i / diffusion_model.T)

            model_output = (1 + final_weight) * eps_pos - final_weight * eps_neg
            if step_index + 1 < len(reversed_timesteps):
                prev_timestep = reversed_timesteps[step_index + 1]
            else:
                prev_timestep = -1

            alpha_prod_t = diffusion_model.alpha_bars[i]
            if prev_timestep >= 0:
                alpha_prod_t_prev = diffusion_model.alpha_bars[prev_timestep].to(device)
            else:
                alpha_prod_t_prev = torch.tensor(1.0, device=device)

            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            pred_original_sample = (
                x_t - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
            variance = (beta_prod_t_prev / beta_prod_t) * (
                1 - alpha_prod_t / alpha_prod_t_prev
            )
            std_dev_t = torch.tensor(0.0, device=device) * variance.sqrt()
            model_output = (
                x_t - alpha_prod_t.sqrt() * pred_original_sample
            ) / beta_prod_t.sqrt()
            pred_sample_direction = (
                1 - alpha_prod_t_prev - std_dev_t**2
            ).sqrt() * model_output
            x_t = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

    return x_t


def selected_settings(setting_arg: str) -> list[str]:
    if setting_arg == "both":
        return ["negfacediff", "adaptdiff"]
    return [setting_arg]


def write_report(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    report_path = output_dir / "sampling_report.csv"
    fieldnames = list(rows[0])
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    manifest = read_manifest(args.manifest_csv)[: args.max_trials]
    contexts = load_contexts(args.contexts_npz)
    latent_encoder, latent_decoder = load_autoencoder(args.autoencoder_dir, device)

    train_cfg = OmegaConf.load(args.checkpoint_dir / ".hydra" / "config.yaml")
    image_size = int(train_cfg.constants.image_size)

    diffusion_model = None
    if not args.prepare_only:
        diffusion_model, train_cfg = load_diffusion_model(args.checkpoint_dir, device)

    report_rows: list[dict[str, str]] = []
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    for row in manifest:
        context_id = int(row["context_id"])
        trial_id = row["trial_id"]
        positive = torch.from_numpy(
            contexts["positive_contexts"][context_id]
        ).unsqueeze(0).to(device)
        negative = torch.from_numpy(
            contexts["negative_contexts"][context_id]
        ).unsqueeze(0).to(device)

        if diffusion_model is None:
            # Instantiate only the schedule/model object needed for q(x_T|x_0).
            diffusion_model = instantiate(train_cfg.diffusion).to(device)
            diffusion_model.eval()

        noise_timestep = (
            diffusion_model.T - 1 if args.noise_timestep is None else args.noise_timestep
        )

        noisy_latent = morph_to_noisy_latent(
            morph_path=Path(row["morph_path"]),
            image_size=image_size,
            latent_encoder=latent_encoder,
            diffusion_model=diffusion_model,
            device=device,
            generator=generator,
            noise_timestep=noise_timestep,
        )

        noisy_path = output_root / "noisy_latents" / f"{trial_id}.pt"
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(noisy_latent.cpu(), noisy_path)

        if args.prepare_only:
            report_rows.append(
                {
                    "trial_id": trial_id,
                    "setting": "prepare_only",
                    "output_path": "",
                    "noisy_latent_path": str(noisy_path),
                    "noise_timestep": str(noise_timestep),
                    "status": "prepared_noisy_latent",
                }
            )
            continue

        for setting_name in selected_settings(args.setting):
            setting = SETTINGS[setting_name]
            sampled_latent = sample_ddim_from_noisy_latent(
                diffusion_model=diffusion_model,
                x_t=noisy_latent.clone(),
                context=positive,
                negative_context=negative,
                weight=setting["weight"],
                adapt=setting["adapt"],
                ddim_steps=args.ddim_steps,
                start_timestep=noise_timestep,
            )
            with torch.no_grad():
                image = latent_decoder(sampled_latent).cpu()
            image = denormalize_to_zero_to_one(image)

            output_path = output_root / setting_name / f"{trial_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(image, output_path)

            report_rows.append(
                {
                    "trial_id": trial_id,
                    "setting": setting_name,
                    "output_path": str(output_path),
                    "noisy_latent_path": str(noisy_path),
                    "noise_timestep": str(noise_timestep),
                    "status": "generated",
                }
            )

    write_report(report_rows, output_root)
    print(f"Device: {device}")
    print(f"Trials processed: {len(manifest)}")
    if report_rows:
        print(f"Noise timestep: {report_rows[0]['noise_timestep']}")
    print(f"Prepare only: {args.prepare_only}")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
