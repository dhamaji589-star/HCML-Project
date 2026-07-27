# HCML Project Conceptual Notes

These notes explain the project in simple language, with the technical terms
you can use in the report or presentation.

## 1. Main Project Goal

The project is about hidden identity recovery from morphed face images.

A morph image is created from two people:

```text
Person A + Person B -> morphed image M_AB
```

In our task, one identity is known:

```text
known identity = A
```

The goal is to recover the other identity:

```text
hidden identity = B
```

Simple example:

```text
input morph: Alice + Bob
known identity: Alice
target: recover Bob
```

Presentation sentence:

> Given a morph image and one known contributing identity, the goal is to
> generate a face image that is closer to the hidden contributing identity.

## 2. Dataset: MAD22

MAD22 contains:

```text
1. Bona fide images: real/genuine source face images
2. Morphed images: images created by combining two identities
```

Example morph filename:

```text
001_08-vs-010_08.jpg
```

This means the morph was made from:

```text
001_08 and 010_08
```

For each morph, we create two directed recovery trials:

```text
Trial 1: morph + 001_08 -> recover 010_08
Trial 2: morph + 010_08 -> recover 001_08
```

Technical term:

```text
directed hidden-identity recovery trials
```

Why two trials?

The task is directional. If A is known, B is hidden. If B is known, A is
hidden.

## 3. Metadata CSV

The metadata CSV turns image filenames into clear machine-readable tasks.

Each row stores:

```text
trial_id
morph_path
known_path
hidden_path
known_id
hidden_id
```

Technical term:

```text
trial metadata
```

Presentation sentence:

> I converted the dataset filenames into directed metadata rows, where each
> row specifies the morph image, the known identity, and the hidden target
> identity.

## 4. Smoke Subset

The full OpenCV subset has:

```text
604 usable morphs
1208 directed trials
```

Before running everything, we use a small smoke subset:

```text
10 morphs
20 directed trials
```

Technical term:

```text
smoke test
```

Why?

It lets us quickly check whether paths, preprocessing, embeddings, model
loading, and generation work before spending GPU time on the full dataset.

Smoke-test results are not final research results. They are for debugging.

## 5. Image Manifest

The trial CSV repeats images. The same image can appear in many trials.

An image manifest stores each unique image once.

Technical term:

```text
unique image manifest
```

Difference:

```text
metadata CSV = what recovery tasks to solve
image manifest = what unique image files need processing
```

Why?

We align and embed each image only once, then reuse it in all relevant trials.

## 6. Face Alignment

Before extracting identity embeddings, face images should be aligned.

Alignment means:

```text
detect/crop face
place face consistently
resize image
```

For ElasticFaceArc embeddings, we use:

```text
112 x 112 face crops
```

Technical terms:

```text
face alignment
ArcFace-style alignment
face preprocessing
```

Important distinction:

```text
112x112 aligned images -> used for face-recognition embeddings
128x128 images -> used by the diffusion autoencoder
```

## 7. ElasticFaceArc

ElasticFaceArc is the face-recognition model we use for identity contexts.

Simple function:

```text
face image -> ElasticFaceArc -> 512-dimensional identity embedding
```

Technical terms:

```text
face-recognition embedding extractor
identity encoder
feature extractor
```

We use ElasticFaceArc to compute:

```text
positive context = embedding(morph image)
negative context = embedding(known identity image)
hidden context = embedding(hidden identity image, for evaluation)
```

Important correction from the supervisor:

```text
Final identity contexts must use ElasticFaceArc.
FRmodel_FarNeg_CASIA and FRmodel_FarNegAdaptive_CASIA are not used for final conditioning.
ElasticFaceCos is optional for final cosine-similarity evaluation.
```

## 8. Identity Embeddings

An identity embedding is a vector of numbers representing a face identity.

In our project:

```text
embedding size = 512 numbers
```

If two images are of the same person, their embeddings should be similar.

Technical term:

```text
512-dimensional identity embedding
```

Similarity is usually measured with:

```text
cosine similarity
```

## 9. Autoencoder

The autoencoder works with images, not identity embeddings.

It has two parts:

```text
encoder: image -> latent
decoder: latent -> image
```

Technical terms:

```text
latent autoencoder
VQGAN autoencoder
first-stage encoder/decoder
```

In our project:

```text
morph image -> autoencoder encoder -> morph latent
generated latent -> autoencoder decoder -> generated face image
```

Important:

The autoencoder does not create identity embeddings. ElasticFaceArc creates
identity embeddings.

## 10. Latent Space

Latent space is a compressed representation space.

Instead of generating directly in pixel space, the diffusion model works in
latent space.

Simple idea:

```text
image space = visible image pixels
latent space = compressed hidden representation
```

Why use latent space?

It is more efficient and is the setup used by the pretrained diffusion model.

## 11. Forward Diffusion / Markovian Noising

Before sampling, the morph image is encoded into latent space and noise is
added.

Flow:

```text
morph image -> morph latent -> noisy morph latent
```

Technical terms:

```text
forward diffusion process
Markovian noising chain
q(x_t | x_0)
```

Supervisor instruction:

```text
Use a total of 1000 noising steps to obtain the starting noise.
Use DDIM with 200 steps during sampling.
```

## 12. DM_CASIA Diffusion Model

DM_CASIA is the pretrained identity-conditioned diffusion model.

Simple function:

```text
noisy latent + identity condition -> denoised/generated latent
```

Technical terms:

```text
conditional denoising diffusion model
latent diffusion model
identity-conditioned diffusion model
```

We do not train DM_CASIA. We load pretrained weights and use it for sampling.

## 13. NegFaceDiff

NegFaceDiff is a sampling/guidance method, not a newly trained model in this
project.

It uses:

```text
positive context = morph embedding
negative context = known identity embedding
weight = 0.5
adapt = false
```

Simple idea:

```text
move toward the morph identity
move away from the known identity
hopefully reveal the hidden identity
```

Technical term:

```text
negative identity guidance
fixed negative guidance
```

## 14. AdaptDiff

AdaptDiff uses the same general idea as NegFaceDiff, but the guidance strength
changes over denoising time.

It uses:

```text
positive context = morph embedding
negative context = known identity embedding
weight = 1.0
adapt = true
```

Technical terms:

```text
adaptive negative guidance
adaptive guidance schedule
```

## 15. Full Correct Pipeline

The full pipeline has two parallel branches.

Identity branch:

```text
morph image -> ElasticFaceArc -> positive identity embedding
known image -> ElasticFaceArc -> negative identity embedding
hidden image -> ElasticFaceArc -> hidden identity embedding for evaluation
```

Image/latent branch:

```text
morph image
  -> autoencoder encoder
  -> morph latent
  -> forward diffusion/noising
  -> noisy morph latent
```

Sampling branch:

```text
noisy morph latent
+ positive identity embedding
+ negative identity embedding
  -> DM_CASIA with NegFaceDiff or AdaptDiff guidance
  -> generated latent
```

Decoding branch:

```text
generated latent -> autoencoder decoder -> generated face image
```

Evaluation branch:

```text
generated face image -> face-recognition model -> generated embedding
compare generated embedding with hidden and known embeddings
```

## 16. What We Are Training

In this project, we are not training the main models.

We use pretrained:

```text
ElasticFaceArc
latent autoencoder
DM_CASIA diffusion model
```

NegFaceDiff and AdaptDiff are sampling strategies applied to the pretrained
diffusion model.

Technical term:

```text
inference / sampling with pretrained models
```

Not training means we are not doing:

```text
loss.backward()
optimizer.step()
```

Presentation sentence:

> I did not train a new model from scratch. I used pretrained components and
> evaluated negative identity guidance strategies for hidden identity recovery.

## 17. Current Face Recognition And Evaluation Setup

Current identity-context model:

```text
ElasticFaceArc with iresnet100 backbone
```

Used for:

```text
positive context
negative context
hidden context
smoke sanity checks
```

Current sanity evaluation:

```text
cosine(embedding(morph), embedding(known))
cosine(embedding(morph), embedding(hidden))
```

This is only a sanity check, not the final diffusion evaluation.

Final generated-image evaluation should compute:

```text
embedding(generated image)
embedding(hidden identity)
embedding(known identity)
```

Then:

```text
sim_to_hidden = cosine(generated, hidden)
sim_to_known = cosine(generated, known)
margin = sim_to_hidden - sim_to_known
```

Success condition:

```text
sim_to_hidden > sim_to_known
```

Optional:

ElasticFaceCos or another face-recognition model can be used as an additional
evaluation model, but ElasticFaceArc is the required model for identity
contexts.

## 18. Current Project Status

Completed:

```text
MAD22 metadata creation
smoke subset creation
image manifest creation
face alignment
ElasticFaceArc embedding extraction
paired sampling input preparation
model asset setup
paired diffusion sampler
20-trial smoke generation on Kaggle GPU
```

Current observation:

```text
generated images are blurry
NegFaceDiff and AdaptDiff outputs look very similar
generated faces do not clearly match hidden identities yet
```

Next step:

```text
debug output quality
then compute generated-image evaluation metrics
```

