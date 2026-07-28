# Report Ideas and Argument Flow

## Central Question

The report should answer one clear question:

Can negative identity guidance in a pretrained diffusion model recover the hidden contributor of a face morph when the other contributor is already known?

## Main Story

1. A morph image contains identity information from two contributors.
2. If one contributor is known, the useful target is the other contributor.
3. The pipeline creates directed recovery tasks: known A -> recover B and known B -> recover A.
4. ElasticFaceArc supplies identity vectors for the morph, known identity, and hidden identity.
5. The diffusion model starts from the noised morph latent and is guided by the morph context while being pushed away from the known identity.
6. NegFaceDiff uses fixed negative guidance; AdaptDiff uses adaptive negative guidance.
7. Evaluation asks whether the generated image is closer to the hidden identity than to the known identity in embedding space.

## Report Tone

The report should not claim that the generated images are visually perfect. The honest and stronger argument is:

- Visual quality is limited.
- The embedding metric still shows a recovery signal.
- AdaptDiff consistently improves over NegFaceDiff across all morphing methods.

## Important Results To Emphasize

- AdaptDiff has a higher success percentage for every morphing subset.
- AdaptDiff also has a higher mean hidden-minus-known cosine margin for every subset.
- MIPGAN-I is the easiest subset in this evaluation.
- OpenCV is the hardest subset.
- Results must remain separated by morphing method because each subset behaves differently.

## Best Visuals

Use two figures:

1. Success-rate bar plot using percentages only.
2. Best qualitative examples selected by positive AdaptDiff margin, showing:
   - known identity
   - morph image
   - hidden identity
   - NegFaceDiff output
   - AdaptDiff output

## Limitations To State Clearly

- No model was trained or fine-tuned in this project.
- The generated images have low visual fidelity.
- ElasticFaceArc was used for both conditioning and evaluation.
- The conclusion is about identity recovery in embedding space, not photorealistic reconstruction.
