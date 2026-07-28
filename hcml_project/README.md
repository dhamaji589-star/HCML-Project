# HCML MAD22 Hidden Identity Recovery Code

This folder contains project-specific wrapper code for the HCML practical
project. It is intentionally separate from the original `NegFaceDiff`,
`SYN-MAD-2022`, and `SMDD` repositories.

## Current implemented step

### 1. Build MAD22 OpenCV metadata

Script:

```bash
python hcml_project/scripts/build_mad22_metadata.py
```

Input:

```text
MAD22/original_sorted/OpenCV
MAD22/original_sorted/BonaFide
```

Output:

```text
hcml_project/metadata/mad22_opencv_trials.csv
hcml_project/metadata/mad22_opencv_skipped.csv
```

The trial CSV contains directed hidden-identity recovery rows.

For a morph named:

```text
001_08-vs-010_08.jpg
```

the script writes:

```text
M_AB + 001_08 -> recover 010_08
M_AB + 010_08 -> recover 001_08
```

This lets us test both possible directions for every usable morph.

Current local output:

```text
usable OpenCV morphs: 604
directed trials:      1208
skipped morphs:       137
```

Skipped morphs are written separately because some morph filenames refer to
source bona fide images that are not present in the local extracted MAD22
subset.

### 2. Create a smoke-test subset

Script:

```bash
python hcml_project/scripts/create_smoke_subset.py
```

Input:

```text
hcml_project/metadata/mad22_opencv_trials.csv
```

Output:

```text
hcml_project/metadata/mad22_opencv_smoke_trials.csv
```

Default behavior:

```text
selected morphs:             10
directed smoke-test trials:  20
directions per morph:        A_to_B and B_to_A
```

This small CSV lets us test the rest of the pipeline quickly before running
alignment, embedding extraction, diffusion sampling, and evaluation on the full
1208 directed trials.

Useful options:

```bash
python hcml_project/scripts/create_smoke_subset.py --num-morphs 20
python hcml_project/scripts/create_smoke_subset.py --selection random --seed 42
```

### 3. Build a unique image manifest

Script:

```bash
python hcml_project/scripts/build_image_manifest.py
```

Input:

```text
hcml_project/metadata/mad22_opencv_smoke_trials.csv
```

Output:

```text
hcml_project/metadata/mad22_opencv_smoke_images.csv
```

Why this is useful:

The trial CSV repeats images. For example, the same bona fide image can appear
as the known identity in one row and as the hidden identity in another row. The
image manifest stores every unique image only once, so later steps can align
and embed it once and reuse the result.

Current smoke manifest:

```text
unique images:       22
unique morph images: 10
unique bona fide:    12
```

Every bona fide source image has role `hidden|known` in the smoke manifest
because each selected morph is evaluated in both directions.

### 4. Align/crop manifest images

Script:

```bash
python hcml_project/scripts/align_manifest_images.py --detector center_crop
```

Input:

```text
hcml_project/metadata/mad22_opencv_smoke_images.csv
```

Output:

```text
hcml_project/outputs/aligned_smoke/
hcml_project/metadata/mad22_opencv_smoke_aligned.csv
```

What this step does:

The image manifest contains one row per unique image. This script creates one
aligned crop per row and writes a report with:

```text
original image path
aligned image path
alignment method
success/failure status
error message, if any
```

Why alignment matters:

Face-recognition models expect faces to be in a consistent position. A simple
example:

```text
bad crop:   eyes appear in different places for every image
good crop:  eyes, nose, and mouth are placed near fixed coordinates
```

If faces are not aligned, identity embeddings become noisy and cosine
similarity becomes less reliable.

Two modes are available:

```bash
# Simple smoke-test mode, useful for checking file paths and reports.
python hcml_project/scripts/align_manifest_images.py --detector center_crop

# Proper landmark-based mode for Kaggle/GPU experiments.
python hcml_project/scripts/align_manifest_images.py --detector mtcnn
```

Local dependency status:

The local Python environment now has the required basic image packages
installed, and the center-crop smoke alignment succeeds:

```text
images processed:       22
successful alignments:  22
failed alignments:       0
```

For Kaggle and full experiments, make sure these packages are available:

```text
pillow
numpy
opencv-python
torch
facenet-pytorch
```

For the actual experiment, use `--detector mtcnn`, because it estimates facial
landmarks and creates ArcFace-style 112x112 crops.

### 5. Extract identity embeddings

Script:

```bash
python hcml_project/scripts/extract_identity_embeddings.py --dry-run
```

Input:

```text
hcml_project/metadata/mad22_opencv_smoke_aligned.csv
```

Dry-run output:

```text
hcml_project/metadata/mad22_opencv_smoke_embeddings.csv
```

What an identity embedding is:

An identity embedding is a vector representation of a face. In our case it
should be 512 numbers. Two images of the same person should have similar
vectors. Two different people should have less similar vectors.

Simple example:

```text
image of A -> face recognition model -> e_A
image of B -> face recognition model -> e_B

cosine(e_A, e_B) high  = identities look similar to the model
cosine(e_A, e_B) low   = identities look different to the model
```

Why this step matters:

For generation:

```text
positive context = embedding(morph image M_AB)
negative context = embedding(known identity A)
```

For evaluation:

```text
generated output G -> embedding e_G
hidden target B    -> embedding e_B
known source A     -> embedding e_A

success if cosine(e_G, e_B) > cosine(e_G, e_A)
```

Correct model choice after supervisor clarification:

```text
Identity contexts for generation: ElasticFaceArc
Optional final cosine evaluation: ElasticFaceCos or another FR model
```

The temporary `FRmodel_FarNeg_CASIA` and `FRmodel_FarNegAdaptive_CASIA` files
were useful for local smoke testing, but they are not the final identity
context models for this project.

Download the ElasticFace-Arc pretrained checkpoint from the official
ElasticFace repository and keep it outside git, for example:

```text
hcml_project/model_assets/elasticface/ElasticFaceArc_295672backbone.pth
```

Final smoke extraction command that has been run locally:

```bash
python hcml_project/scripts/extract_identity_embeddings.py \
  --weights-path hcml_project/model_assets/elasticface/ElasticFaceArc_295672backbone.pth \
  --architecture iresnet100 \
  --embedding-model-name elasticface_arc \
  --output-npz hcml_project/embeddings/mad22_opencv_smoke_elasticface_arc.npz \
  --report-csv hcml_project/metadata/mad22_opencv_smoke_elasticface_arc_embeddings.csv
```

Useful local dry run before the weights are available:

```bash
python hcml_project/scripts/extract_identity_embeddings.py \
  --dry-run \
  --embedding-model-name elasticface_arc \
  --report-csv hcml_project/metadata/mad22_opencv_smoke_elasticface_arc_embeddings.csv
```

The script saves real embeddings to an NPZ file. Every row in the report also
stores the embedding model name so we can avoid mixing temporary smoke-test
features with final ElasticFaceArc contexts.

Current ElasticFaceArc smoke result:

```text
Embeddings extracted: 22
Embedding model: elasticface_arc
```

### 6. Check required model assets

Script:

```bash
python hcml_project/scripts/check_required_assets.py
```

This checks whether the final-experiment assets are present:

```text
DM_CASIA diffusion checkpoint
latent autoencoder config
latent autoencoder encoder weights
latent autoencoder decoder weights
ElasticFaceArc weights
```

Strict mode, useful in Kaggle before running expensive jobs:

```bash
python hcml_project/scripts/check_required_assets.py --strict
```

The FFHQ LDM download provides a single `model.ckpt`. Convert it into the two
NegFaceDiff autoencoder files with:

```bash
python hcml_project/scripts/convert_ffhq_autoencoder.py
```

This writes:

```text
NegFaceDiff/models/autoencoder/first_stage_encoder_state_dict.pt
NegFaceDiff/models/autoencoder/first_stage_decoder_state_dict.pt
```

Current local asset status:

```text
All required assets are present.
```

### 7. Evaluate the embedding sanity check

Script:

```bash
python hcml_project/scripts/evaluate_embedding_baseline.py
```

Input:

```text
hcml_project/metadata/mad22_opencv_smoke_trials.csv
hcml_project/metadata/mad22_opencv_smoke_elasticface_arc_embeddings.csv
hcml_project/embeddings/mad22_opencv_smoke_elasticface_arc.npz
```

Output:

```text
hcml_project/metadata/mad22_opencv_smoke_elasticface_arc_sanity_eval.csv
```

What this step does:

For every directed trial, it compares:

```text
cosine(embedding(morph), embedding(known identity A))
cosine(embedding(morph), embedding(hidden identity B))
```

Simple example:

```text
sim_morph_known  = 0.70
sim_morph_hidden = 0.82

margin_hidden_minus_known = 0.82 - 0.70 = 0.12
```

In this example, the morph is closer to the hidden identity according to the
face-recognition model.

Why this step matters:

This is not the final project result because we have not generated recovered
faces yet. It is a sanity check that confirms:

```text
trial metadata -> aligned images -> embeddings -> similarity evaluation
```

are connected correctly.

Current smoke-test result with ElasticFaceArc:

```text
Trials evaluated:       20
Gallery identities:     12
Closer to hidden:       10/20
Hidden retrieval top-1: 10/20
Hidden retrieval top-5: 18/20
Mean hidden rank:       2.50
```

The `10/20` top-1 result is reasonable for this smoke setup. Each morph is
used twice:

```text
M_AB + A -> hidden B
M_AB + B -> hidden A
```

If the raw morph embedding is closer to identity A, then the `hidden A`
direction succeeds and the `hidden B` direction fails. Diffusion recovery is
the later step that should use negative guidance to push the generated image
away from the known identity and toward the hidden one.

This is only a sanity check for our metadata/alignment/embedding plumbing. It
is not the diffusion reconstruction baseline Eduarda mentioned.

### 8. Prepare paired sampling inputs

Script:

```bash
python hcml_project/scripts/prepare_sampling_inputs.py
```

Input:

```text
hcml_project/metadata/mad22_opencv_smoke_trials.csv
hcml_project/metadata/mad22_opencv_smoke_elasticface_arc_embeddings.csv
hcml_project/embeddings/mad22_opencv_smoke_elasticface_arc.npz
```

Output:

```text
hcml_project/sampling_inputs/mad22_opencv_smoke_elasticface_arc/
```

Files created:

```text
paired_contexts.npz
paired_contexts_manifest.csv
negfacediff_sampling_settings.yaml
adaptdiff_sampling_settings.yaml
```

What this step does:

For every directed trial, it prepares the three vectors we need:

```text
positive context = embedding(morph image M_AB)
negative context = embedding(known identity A)
hidden context   = embedding(hidden identity B)
```

The hidden context is saved for evaluation only. During generation, the model
should use the morph embedding as the positive context and the known identity
embedding as the negative context.

Example:

```text
trial: OpenCV_001_08_vs_010_08_A_to_B
known identity:  001_08
hidden identity: 010_08

positive context -> embedding of 001_08-vs-010_08.jpg
negative context -> embedding of 001_08.jpg
hidden context   -> embedding of 010_08.jpg
```

Current smoke-test result:

```text
Trials prepared: 20
Context dimension: 512
Positive contexts shape: (20, 512)
Negative contexts shape: (20, 512)
Hidden contexts shape:   (20, 512)
Embedding model: elasticface_arc
```

The two project settings from the supervisor are also written:

```text
NegFaceDiff: weight = 0.5, adapt = false
AdaptDiff:   weight = 1.0, adapt = true
```

### 9. Run paired diffusion sampling

Script:

```bash
python hcml_project/scripts/sample_paired_diffusion.py
```

What this step does:

For each trial, it follows the supervisor-corrected generation flow:

```text
morph image
  -> latent autoencoder encoder
  -> 1000-step forward noising process
  -> morph-derived noisy latent
  -> DDIM reverse sampling
  -> latent autoencoder decoder
  -> generated recovered face
```

The script uses:

```text
positive context = ElasticFaceArc embedding of morph image
negative context = ElasticFaceArc embedding of known identity
```

Smoke validation without reverse diffusion:

```bash
python hcml_project/scripts/sample_paired_diffusion.py \
  --prepare-only \
  --max-trials 1 \
  --device cpu
```

Tiny local code-path test:

```bash
python hcml_project/scripts/sample_paired_diffusion.py \
  --max-trials 1 \
  --setting negfacediff \
  --ddim-steps 1 \
  --device cpu
```

This one-step command only checks that loading, noising, sampling, decoding,
and image saving work. It is not a valid experiment result.

Real smoke experiment on Kaggle/GPU:

```bash
python hcml_project/scripts/sample_paired_diffusion.py \
  --max-trials 20 \
  --setting both \
  --ddim-steps 200 \
  --device cuda
```

With `--ddim-steps 200`, the default sampler now follows the original
NegFaceDiff DDIM loop: `skip = 1000 // 200 = 5`, so the reverse process starts
at timestep `995` and then uses `990, 985, ... 0`.

Outputs:

```text
hcml_project/outputs/generated_smoke_elasticface_arc/noisy_latents/
hcml_project/outputs/generated_smoke_elasticface_arc/negfacediff/
hcml_project/outputs/generated_smoke_elasticface_arc/adaptdiff/
hcml_project/outputs/generated_smoke_elasticface_arc/sampling_report.csv
```

Image-quality diagnostic:

```bash
python hcml_project/scripts/sample_paired_diffusion.py \
  --max-trials 4 \
  --setting both \
  --ddim-steps 200 \
  --noise-timestep 500 \
  --ddim-timestep-mode linspace \
  --output-dir hcml_project/outputs/generated_smoke_elasticface_arc_t500 \
  --device cuda
```

The default behavior uses the original NegFaceDiff-style DDIM schedule. The
`--noise-timestep` option is only for controlled debugging/ablation runs. A
smaller value keeps more information from the morph latent before reverse
sampling, so it can help us check whether the very blurry outputs are caused by
starting from almost pure noise.

Autoencoder reconstruction diagnostic:

```bash
python hcml_project/scripts/sample_paired_diffusion.py \
  --prepare-only \
  --max-trials 4 \
  --resize-filter lanczos \
  --save-morph-reconstructions \
  --output-dir hcml_project/outputs/reconstruction_check_lanczos \
  --device cuda
```

This saves images in:

```text
hcml_project/outputs/reconstruction_check_lanczos/morph_reconstructions/
```

Use this before another full generation run. If these reconstructions are
already blurry or distorted, then the autoencoder/latent representation is a
major image-quality bottleneck. If they look good, then the quality loss is
mostly happening during the diffusion reverse sampling.

## Next planned steps

### 10. Evaluate generated recovery images

Script:

```bash
python hcml_project/scripts/evaluate_generated_recovery.py
```

What this step computes:

```text
sim_generated_hidden = cosine(embedding(generated), embedding(hidden identity))
sim_generated_known  = cosine(embedding(generated), embedding(known identity))
margin               = sim_generated_hidden - sim_generated_known
```

Success condition:

```text
sim_generated_hidden > sim_generated_known
```

Kaggle command after smoke generation:

```bash
python hcml_project/scripts/evaluate_generated_recovery.py \
  --device cuda
```

To evaluate a diagnostic output directory, point the evaluator to that
directory's report:

```bash
python hcml_project/scripts/evaluate_generated_recovery.py \
  --sampling-report-csv hcml_project/outputs/generated_smoke_elasticface_arc_t500/sampling_report.csv \
  --output-csv hcml_project/metadata/mad22_opencv_smoke_generated_recovery_eval_t500.csv \
  --device cuda
```

Output:

```text
hcml_project/metadata/mad22_opencv_smoke_generated_recovery_eval.csv
```

This gives separate summaries for:

```text
NegFaceDiff
AdaptDiff
```

### 11. Run Equal-Size Subsets For Each MAD22 Morphing Method

The supervisor requested that morphing methods are evaluated independently. Use
the helper below to run the same number of morphs per method, with separate
outputs and metrics for each method:

```bash
python hcml_project/scripts/run_mad22_method_pipeline.py \
  --method OpenCV \
  --num-morphs 10 \
  --stage all \
  --device cuda \
  --embedding-device cpu
```

Repeat by changing `--method`:

```text
OpenCV
FaceMorpher
MIPGAN_I
MIPGAN_II
Webmorph
```

Each method writes separate files, for example:

```text
hcml_project/outputs/generated_mad22_opencv_subset_elasticface_arc/
hcml_project/metadata/mad22_opencv_subset_generated_recovery_eval.csv
```

Important: do not merge the CSVs into one overall metric. In the report, use one
row per morphing method and keep qualitative examples method-specific.

## Next planned steps

1. Run generated-image recovery evaluation on Kaggle.
2. Debug output quality using the evaluation numbers and qualitative examples.
3. Scale each MAD22 morphing subset independently.
