# Hidden Identity Recovery from Face Morphing Attacks using NegFaceDiff and AdaptDiff

**Student:** Vishu  
**Project:** HCML MAD22 Project  

## Abstract

Face morphing attacks combine two identities into one facial image, creating a sample that can be accepted as both contributors by a face recognition system. This project evaluates whether a pretrained identity-conditioned diffusion model can recover the hidden contributor when the morph image and one known contributor are given. I used NegFaceDiff and AdaptDiff as two negative-guidance sampling strategies on the MAD22 morphing subsets: OpenCV, FaceMorpher, MIPGAN-I, MIPGAN-II, and WebMorph. Identity contexts and evaluation embeddings were extracted with ElasticFaceArc. Across 40 morph images per method, corresponding to 80 directed recovery trials per method, AdaptDiff consistently outperformed NegFaceDiff. The generated images were visually blurry, but the embedding-based recovery metric showed that AdaptDiff more often produced samples closer to the hidden identity than to the known identity.

## 1. Introduction

Face morphing attacks are a security risk for biometric systems because a single morphed image can contain identity information from two people. If such an image is used in an identity document, both contributing subjects may be able to match against it. The task in this project is not morphing attack detection, but **hidden identity recovery**: given a morph image and one known contributing identity, generate an image that is closer to the other, hidden identity.

For a morph image created from identities A and B, two directed tasks can be formed. In the first task, A is known and B is hidden; in the second task, B is known and A is hidden. This directional setup is important because the model is guided away from the known identity while using the morph image as the positive identity condition.

## 2. Method

The pipeline uses pretrained models and performs inference only. No model was trained from scratch. First, dataset filenames were converted into trial metadata containing the morph image, known identity image, and hidden identity image. Each morph therefore creates two directed recovery trials.

Identity embeddings were extracted using ElasticFaceArc with an iresnet100 backbone. The morph embedding was used as the positive context, the known identity embedding was used as the negative context, and the hidden identity embedding was kept only for evaluation. Each identity context is a 512-dimensional embedding.

The image generation branch used the pretrained DM_CASIA latent diffusion model from the NegFaceDiff repository. The morph image was encoded into latent space using the pretrained first-stage autoencoder. Noise was then added using the forward diffusion process with a 1000-step schedule, matching the setting used for the provided model. Sampling was performed with DDIM using 200 reverse steps.

Two sampling settings were evaluated:

| Setting | Adapt | Negative weight | Description |
|---|---:|---:|---|
| NegFaceDiff | false | 0.5 | Fixed negative identity guidance |
| AdaptDiff | true | 1.0 | Adaptive negative identity guidance |

In simple terms, both settings try to preserve information from the morph while moving away from the known identity. The difference is that NegFaceDiff uses a fixed negative guidance strength, while AdaptDiff changes the guidance behavior during the denoising process.

## 3. Experiments

Experiments were performed on the MAD22 dataset. The evaluated morphing methods were OpenCV, FaceMorpher, MIPGAN-I, MIPGAN-II, and WebMorph. Following the project instruction, each morphing method was evaluated independently; results were not merged into one global metric.

For each method, I evaluated 40 morph images. Since each morph produces two directed recovery trials, this gives 80 trials per method. For every trial, both NegFaceDiff and AdaptDiff generated one recovered image. The generated image was then embedded using ElasticFaceArc and compared against the hidden and known identity embeddings using cosine similarity.

The success condition was:

```text
cosine(generated, hidden) > cosine(generated, known)
```

I also computed the margin:

```text
margin = cosine(generated, hidden) - cosine(generated, known)
```

A positive margin means the generated image is closer to the hidden identity than to the known identity in the face-recognition embedding space.

## 4. Results

![Success rate by morphing method](../../results/success_rate_by_method.png)

**Table 1: Hidden identity recovery results.** Each method uses 40 morph images and 80 directed trials. The margin is the mean value of cosine(generated, hidden) minus cosine(generated, known).

| Morphing method | NegFaceDiff success | NegFaceDiff margin | AdaptDiff success | AdaptDiff margin |
|---|---:|---:|---:|---:|
| OpenCV | 48/80 (60.0%) | 0.069954 | 50/80 (62.5%) | 0.090291 |
| FaceMorpher | 60/80 (75.0%) | 0.045358 | 66/80 (82.5%) | 0.059136 |
| MIPGAN-I | 64/80 (80.0%) | 0.063586 | 71/80 (88.8%) | 0.084853 |
| MIPGAN-II | 58/80 (72.5%) | 0.060798 | 64/80 (80.0%) | 0.078214 |
| WebMorph | 57/80 (71.2%) | 0.063933 | 60/80 (75.0%) | 0.087545 |

AdaptDiff achieved higher success than NegFaceDiff for every morphing method. The largest gain was observed for MIPGAN-I, where AdaptDiff improved from 64/80 to 71/80 successful directed trials. AdaptDiff also produced higher mean margins in all five groups, suggesting that its successful generations were not only more frequent but also more separated from the known identity in embedding space.

MIPGAN-I was the easiest subset for both methods, while OpenCV was the hardest. This suggests that different morph generation algorithms preserve and distribute identity information differently. Since each subset was evaluated separately, this method-dependent behavior is visible instead of being hidden by an aggregated metric.

## 5. Qualitative Analysis

![Qualitative examples](../../results/qualitative_examples.png)

The generated samples often have low visual fidelity: faces are blurry, sometimes color-shifted, and not always convincing to a human observer. However, the quantitative identity metric can still show a positive recovery signal because it measures similarity in a face-recognition embedding space rather than human-perceived image quality.

This difference is important. A generated image may not look photorealistic, but it can still contain identity cues that ElasticFaceArc maps closer to the hidden contributor than to the known contributor. For the report, this means the results should be interpreted as **identity recovery in embedding space**, not as high-quality face reconstruction.

## 6. Discussion

The main result is that adaptive negative guidance is more effective than fixed negative guidance in this experimental setting. AdaptDiff likely benefits from not applying the same negative pressure throughout the whole denoising process. Early denoising stages control broad image structure, while later stages refine identity and appearance. A fixed negative weight can restrict the generation process too strongly, while adaptive guidance can balance exploration and identity separation more flexibly.

The main limitation is visual quality. The project used pretrained model weights and did not fine-tune the diffusion model or autoencoder on MAD22. The image resolution and latent autoencoder also limit fine details. Another limitation is that ElasticFaceArc was used both for identity conditioning and for evaluation, which is consistent with the project instructions but may favor identity information captured by that specific model. A future extension would evaluate the generated images with an additional independent face-recognition model.

## 7. Conclusion

This project implemented an end-to-end hidden identity recovery pipeline for MAD22 morphing attacks using NegFaceDiff and AdaptDiff. The pipeline prepares directed trials, extracts ElasticFaceArc identity contexts, generates recovered images with a pretrained latent diffusion model, and evaluates whether the generated image is closer to the hidden or known identity. Across all evaluated morphing methods, AdaptDiff outperformed NegFaceDiff in both success rate and mean identity margin. The results support the usefulness of adaptive negative guidance, while the qualitative outputs show that identity recovery and visual realism remain separate challenges.

## References

[1] E. Caldeira, T. Chettaoui, N. Damer, and F. Boutros, "AdaptDiff: Adaptive Guidance in Diffusion Models for Diverse and Identity-Consistent Face Synthesis," 2026. Code: https://github.com/EduardaCaldeira/NegFaceDiff/

[2] M. Huber et al., "SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data," IJCB, 2022. Dataset/code: https://github.com/marcohuber/SYN-MAD-2022

[3] F. Boutros, N. Damer, F. Kirchbuchner, and A. Kuijper, "ElasticFace: Elastic Margin Loss for Deep Face Recognition," CVPR Workshops, 2022.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," CVPR, 2019.

[5] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS, 2020.

[6] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," ICLR, 2021.

[7] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR, 2022.
