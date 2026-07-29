# HCML MAD22 Hidden Identity Recovery

This repository contains the code used for an HCML project on hidden-identity
recovery from face morphing attacks. Given a morphed face image and one known
contributor, the pipeline uses identity-conditioned diffusion sampling to
generate a candidate image that moves toward the other, hidden contributor.

The implementation builds on the provided NegFaceDiff/AdaptDiff code and adds
project-specific wrappers for MAD22 metadata preparation, face alignment,
ElasticFaceArc embedding extraction, paired diffusion sampling, and recovery
evaluation.

## Repository Contents

```text
NegFaceDiff/              Original NegFaceDiff/AdaptDiff source code used by the project
hcml_project/
  README.md               Detailed project workflow and command reference
  requirements_kaggle.txt Minimal Kaggle dependencies
  metadata/               Lightweight metadata produced during setup
  scripts/                Project-specific pipeline scripts
```

Large datasets, pretrained weights, embeddings, generated images, and Kaggle
result archives are intentionally not included in GitHub.

## External Assets Required

The following assets must be downloaded separately before running the full
pipeline:

1. MAD22/SYN-MAD-2022 dataset.
2. `DM_CASIA_cpd25` diffusion checkpoint from the supervisor-provided shared
   folder.
3. FFHQ latent autoencoder weights from the Latent Diffusion release, converted
   with `hcml_project/scripts/convert_ffhq_autoencoder.py`.
4. ElasticFaceArc pretrained weights from the ElasticFace repository.
5. ElasticFaceCos pretrained weights from the ElasticFace repository for final
   threshold-based face-recognition evaluation.

Expected local paths are documented in `hcml_project/README.md`. They are also
checked by:

```bash
python hcml_project/scripts/check_required_assets.py --strict
```

## Main Experiment Flow

1. Build directed MAD22 recovery trials.
2. Build a unique image manifest.
3. Align/crop all required faces.
4. Extract ElasticFaceArc identity embeddings for generation contexts.
5. Prepare positive, negative, and hidden identity contexts.
6. Generate images with NegFaceDiff and AdaptDiff settings.
7. Evaluate generated images with ElasticFaceCos using directional and
   threshold-based metrics.

The two compared sampling settings are:

```text
NegFaceDiff: adapt=false, weight=0.5
AdaptDiff:   adapt=true,  weight=1.0
```

Each MAD22 morphing method is evaluated independently, as required for the
project report.

The final metric follows the supervisor feedback: a generated/hidden pair is
counted as recovered only when the ElasticFaceCos cosine similarity is above
the threshold used for positive pairs:

```text
cosine(generated, hidden identity) >= 0.321
```

The older directional metric,
`cosine(generated, hidden) > cosine(generated, known)`, is still reported as
supporting evidence that sampling moved away from the known identity.

## Running on Kaggle

Upload this repository as a Kaggle dataset together with separate private Kaggle
datasets for MAD22 and the model assets. Then copy the repository to
`/kaggle/working` and follow the commands in `hcml_project/README.md`.

The helper below runs one morphing subset end-to-end:

```bash
python hcml_project/scripts/run_mad22_method_pipeline.py \
  --method OpenCV \
  --num-morphs 10 \
  --stage all \
  --device cuda \
  --embedding-device cpu
```

Supported methods:

```text
OpenCV
FaceMorpher
MIPGAN_I
MIPGAN_II
Webmorph
```

## Notes

The report is submitted separately by email. This GitHub repository is intended
to provide the reproducible code and setup instructions only.
