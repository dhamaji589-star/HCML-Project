# HCML MAD22 Pipeline

This folder contains the project-specific code for hidden-identity recovery from
MAD22 face morphing attacks. The scripts wrap the original NegFaceDiff/AdaptDiff
implementation and provide a reproducible pipeline for:

1. Building directed recovery trials from MAD22 filenames.
2. Aligning images.
3. Extracting ElasticFaceArc identity embeddings for conditioning.
4. Preparing paired diffusion contexts.
5. Sampling with NegFaceDiff and AdaptDiff settings.
6. Evaluating hidden-identity recovery with ElasticFaceCos.

## Required Assets

The repository does not include datasets or model weights. Place them locally or
in Kaggle input datasets using the following structure:

```text
MAD22/
  original_sorted/
    BonaFide/
    OpenCV/
    FaceMorpher/
    MIPGAN-I/
    MIPGAN-II/
    WebMorph/

DM_CASIA_cpd25/
  .hydra/config.yaml
  checkpoints/ema_averaged_model.ckpt

NegFaceDiff/models/autoencoder/
  first_stage_config.yaml
  first_stage_encoder_state_dict.pt
  first_stage_decoder_state_dict.pt

hcml_project/model_assets/elasticface/
  ElasticFaceArc_295672backbone.pth
  ElasticFaceCos_295672backbone.pth
```

Check the expected files before running expensive jobs:

```bash
python hcml_project/scripts/check_required_assets.py --strict
```

## Dependencies

On Kaggle, install the lightweight requirements:

```bash
pip install -r hcml_project/requirements_kaggle.txt
```

The full sampling step should be run with GPU enabled.

## Morphing Methods

The supported MAD22 morphing subsets are:

```text
OpenCV
FaceMorpher
MIPGAN_I
MIPGAN_II
Webmorph
```

Each method should be evaluated independently. Do not merge all methods into one
single aggregate metric.

## End-to-End Method Runner

The simplest way to run one method is:

```bash
python hcml_project/scripts/run_mad22_method_pipeline.py \
  --method OpenCV \
  --num-morphs 10 \
  --stage all \
  --device cuda \
  --embedding-device cpu
```

Use `--start-index` to run the next non-overlapping subset:

```bash
python hcml_project/scripts/run_mad22_method_pipeline.py \
  --method OpenCV \
  --num-morphs 10 \
  --start-index 10 \
  --output-tag subset2 \
  --stage all \
  --device cuda \
  --embedding-device cpu
```

Important sampling settings used in the project:

```text
NegFaceDiff: adapt=false, weight=0.5
AdaptDiff:   adapt=true,  weight=1.0
DDIM steps:  200
```

The final face-recognition evaluation uses ElasticFaceCos with threshold
`0.321`. This threshold is used to decide whether the generated image and hidden
identity image form a positive pair:

```text
cosine(generated, hidden identity) >= 0.321
```

The directional metric is still reported as supporting evidence:

```text
cosine(generated, hidden identity) > cosine(generated, known identity)
```

In short: the threshold metric is the stricter recovery metric, while the
directional metric only shows movement away from the known contributor.

## Manual Pipeline

The end-to-end runner calls the same scripts listed below. They can also be run
manually when debugging.

### 1. Build Directed Trials

```bash
python hcml_project/scripts/build_mad22_metadata.py \
  --method OpenCV
```

This creates directed trials. For a morph from identities `A` and `B`, two rows
are produced:

```text
morph + known A -> recover hidden B
morph + known B -> recover hidden A
```

### 2. Create a Small Subset

```bash
python hcml_project/scripts/create_smoke_subset.py \
  --num-morphs 10
```

### 3. Build Image Manifest

```bash
python hcml_project/scripts/build_image_manifest.py
```

The manifest stores each unique image once, so repeated identities are aligned
and embedded only once.

### 4. Align Images

```bash
python hcml_project/scripts/align_manifest_images.py --detector mtcnn
```

For quick file-path smoke tests, `--detector center_crop` can be used, but MTCNN
alignment is preferred for actual experiments.

### 5. Extract ElasticFaceArc Embeddings

```bash
python hcml_project/scripts/extract_identity_embeddings.py \
  --weights-path hcml_project/model_assets/elasticface/ElasticFaceArc_295672backbone.pth \
  --architecture iresnet100 \
  --embedding-model-name elasticface_arc \
  --device cuda
```

The generated 512-dimensional embeddings are used as identity contexts.

### 6. Prepare Sampling Inputs

```bash
python hcml_project/scripts/prepare_sampling_inputs.py
```

For each directed trial, this prepares:

```text
positive context = embedding(morph image)
negative context = embedding(known identity)
hidden context   = embedding(hidden identity, evaluation only)
```

### 7. Generate Images

```bash
python hcml_project/scripts/sample_paired_diffusion.py \
  --setting both \
  --ddim-steps 200 \
  --device cuda
```

The morph image is encoded into latent space, noised with the diffusion forward
process, sampled back using DDIM, and decoded into generated face images.

### 8. Evaluate Generated Recovery

```bash
python hcml_project/scripts/evaluate_generated_recovery.py \
  --weights-path hcml_project/model_assets/elasticface/ElasticFaceCos_295672backbone.pth \
  --evaluation-model-name elasticface_cos \
  --positive-pair-threshold 0.321 \
  --device cuda
```

The evaluator embeds generated, hidden, known, and morph images with
ElasticFaceCos. It reports:

```text
cosine(generated, hidden identity)
cosine(generated, known identity)
cosine(morph, hidden identity)
```

The main threshold-recovery metric is:

```text
cosine(generated, hidden) >= 0.321
```

The supporting directional metric is:

```text
cosine(generated, hidden) > cosine(generated, known)
```

The reported directional margin is:

```text
cosine(generated, hidden) - cosine(generated, known)
```

## Report Plot Helpers

After collecting method-specific evaluation CSV files, use:

```bash
python hcml_project/scripts/summarize_method_results.py
python hcml_project/scripts/make_report_plots.py
python hcml_project/scripts/make_best_qualitative_figure.py
```

These scripts create summary tables and qualitative figures for the report. The
generated figures are not committed to GitHub by default.
