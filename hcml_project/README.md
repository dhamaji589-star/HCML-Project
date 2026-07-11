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

## Next planned steps

1. Add identity embedding extraction.
2. Add paired NegFaceDiff / AdaptDiff sampling.
3. Add similarity and retrieval evaluation.
