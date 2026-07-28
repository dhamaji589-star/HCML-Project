# Hidden Identity Recovery from Face Morphing Attacks using NegFaceDiff and AdaptDiff

**Student:** Vishu  
**Project:** HCML MAD22 Project

## Abstract

Face morphing attacks combine identity information from two subjects into one facial image. This project studies a hidden-identity recovery setting: given a morph image and one known contributor, the goal is to generate an image that is closer to the other contributor. The implemented pipeline uses pretrained components from NegFaceDiff/AdaptDiff, a latent diffusion model trained on CASIA, a pretrained latent autoencoder, and ElasticFaceArc identity embeddings. Five MAD22 morphing subsets were evaluated independently: OpenCV, FaceMorpher, MIPGAN-I, MIPGAN-II, and WebMorph. The experiments show a consistent pattern: AdaptDiff gives higher hidden-identity recovery success than NegFaceDiff for every morphing method. The visual outputs remain blurry, so the main conclusion is about identity recovery in embedding space rather than photorealistic face reconstruction.

## 1. Introduction

Face morphing is a biometric attack in which two face identities are blended into one image. Such an image can be problematic because it may retain enough information from both contributors to match against either person. In this project, the question is not whether a morph can be detected, but whether one can recover information about the unknown contributor when the other contributor is already known.

For a morph created from identities A and B, the recovery task is directional. If A is given as the known identity, B is the hidden target; if B is given, A becomes the hidden target. Therefore, each morph image is converted into two directed recovery trials. This design makes the problem explicit: the model should use the morph as a source of mixed identity information, while suppressing the known identity so that the generated sample moves toward the hidden one.

## 2. Method

The project uses pretrained models and performs inference only. No diffusion model, autoencoder, or face-recognition model was trained during this work.

The first part of the pipeline prepares metadata. Each trial stores the path to the morph image, the known identity image, and the hidden identity image. The identity branch then uses ElasticFaceArc with an iresnet100 backbone to extract 512-dimensional embeddings. The morph embedding is used as the positive identity context, the known identity embedding is used as the negative context, and the hidden identity embedding is reserved for evaluation.

The image branch works in latent space. The morph image is encoded by the pretrained first-stage autoencoder. Forward diffusion noise is added to the morph latent using the 1000-step schedule expected by the provided DM_CASIA model. During sampling, DDIM is used with 200 denoising steps. The generated latent is then decoded back into an image by the autoencoder decoder.

Two guidance settings are compared:

| Setting | Adapt flag | Negative weight | Interpretation |
|---|---:|---:|---|
| NegFaceDiff | false | 0.5 | Fixed negative identity guidance |
| AdaptDiff | true | 1.0 | Adaptive negative identity guidance |

Both settings use the same broad idea: keep useful identity information from the morph while pushing the generation away from the known contributor. The difference is that NegFaceDiff applies a fixed negative guidance strength, whereas AdaptDiff changes the negative guidance behavior during the denoising process.

## 3. Experimental Protocol

Experiments were performed on five MAD22 morphing subsets: OpenCV, FaceMorpher, MIPGAN-I, MIPGAN-II, and WebMorph. Each subset was evaluated separately because morphing algorithms can preserve identity information in different ways. Keeping the subsets separate also follows the project instruction not to merge all morphing methods into a single aggregate metric.

For each directed trial, both NegFaceDiff and AdaptDiff generated one candidate hidden-identity image. The generated image was embedded using ElasticFaceArc and compared to the known and hidden identity embeddings using cosine similarity. A generation is counted as successful when:

```text
cosine(generated, hidden) > cosine(generated, known)
```

The mean margin is computed as:

```text
cosine(generated, hidden) - cosine(generated, known)
```

A larger positive margin means that, according to the face-recognition embedding space, the generated sample is more clearly shifted toward the hidden contributor and away from the known contributor.

## 4. Results

![Success rate by morphing method](../../results/success_rate_by_method.png)

**Table 1: Hidden-identity recovery results.** Success is reported as a percentage. The margin is the mean hidden-minus-known cosine similarity difference.

| Morphing method | NegFaceDiff success | NegFaceDiff margin | AdaptDiff success | AdaptDiff margin |
|---|---:|---:|---:|---:|
| OpenCV | 60.0% | 0.069954 | 62.5% | 0.090291 |
| FaceMorpher | 75.0% | 0.045358 | 82.5% | 0.059136 |
| MIPGAN-I | 80.0% | 0.063586 | 88.8% | 0.084853 |
| MIPGAN-II | 72.5% | 0.060798 | 80.0% | 0.078214 |
| WebMorph | 71.2% | 0.063933 | 75.0% | 0.087545 |

AdaptDiff improves over NegFaceDiff for all five morphing methods. This happens not only in the success percentage, but also in the mean margin. The margin result is important because it shows that AdaptDiff is not merely changing a few borderline cases; on average, its generated embeddings are more strongly separated from the known identity and closer to the hidden one.

The easiest subset in this evaluation is MIPGAN-I, where both methods achieve their strongest recovery rates. OpenCV is the most difficult subset. This variation supports the decision to report each morphing method independently: different morphing procedures create different recovery conditions.

## 5. Qualitative Analysis

![Best qualitative examples](../../results/qualitative_best_examples.png)

The qualitative examples show the known identity, input morph, hidden identity, and the two generated outputs. These examples were selected from successful AdaptDiff cases with strong positive margins, so they represent clearer cases according to the embedding metric.

A visual limitation is still obvious. The generated samples are often blurry, color-shifted, and less realistic than the original MAD22 images. This means the method should not be described as producing clean photo-quality reconstructions. Instead, the stronger and more accurate interpretation is that the generated images contain identity cues that ElasticFaceArc places closer to the hidden contributor than to the known contributor.

This distinction matters for the conclusion. A human observer may not always see a convincing recovered face, but the embedding-space comparison can still indicate that some hidden identity information has been recovered. Therefore, the project demonstrates an identity-recovery signal, not a complete visual reconstruction solution.

## 6. Discussion

The consistent advantage of AdaptDiff suggests that adaptive negative guidance is better suited to this hidden-identity recovery task than fixed negative guidance. One possible explanation is that diffusion sampling changes character over time. Earlier denoising steps influence broad structure, while later steps refine details and identity-related features. If the negative condition is fixed throughout the whole process, the model may be constrained too rigidly. Adaptive guidance can allow broader exploration early and stronger identity separation later.

There are also important limitations. First, all main components are pretrained; the project does not fine-tune the diffusion model for MAD22. Second, ElasticFaceArc is used both for conditioning and evaluation. This is consistent with the project setup, but an additional independent face-recognition model would provide a stronger external check. Third, visual quality remains weak, which limits the practical interpretability of individual generated images.

Despite these limitations, the pipeline is complete and reproducible: metadata preparation, alignment, identity embedding extraction, latent diffusion sampling, image decoding, and embedding-based evaluation are all implemented as separate steps.

## 7. Conclusion

This project implemented and evaluated a hidden-identity recovery pipeline for MAD22 face morphing attacks. The results show that AdaptDiff performs better than NegFaceDiff across all evaluated morphing methods. The main evidence is the consistent increase in success percentage and mean hidden-minus-known identity margin. The generated images are not visually high quality, but they often move in the correct identity direction in ElasticFaceArc embedding space. Overall, adaptive negative guidance appears more effective than fixed negative guidance for this recovery setting.

## References

[1] E. Caldeira, T. Chettaoui, N. Damer, and F. Boutros, "AdaptDiff: Adaptive Guidance in Diffusion Models for Diverse and Identity-Consistent Face Synthesis," 2026. Code: https://github.com/EduardaCaldeira/NegFaceDiff/

[2] M. Huber et al., "SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data," IJCB, 2022. Dataset/code: https://github.com/marcohuber/SYN-MAD-2022

[3] F. Boutros, N. Damer, F. Kirchbuchner, and A. Kuijper, "ElasticFace: Elastic Margin Loss for Deep Face Recognition," CVPR Workshops, 2022.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," CVPR, 2019.

[5] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS, 2020.

[6] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," ICLR, 2021.

[7] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR, 2022.
