# Facial Emotion Recognition - SOTA deep learning models and benchmark datasets

_Generated from 50 structured JSON entries in `results/`._

**Total: 50** (26 models, 24 datasets)

---

## Table of Contents

### Models (26)

1. [Ada-DF (Adaptive Dual-branch Fusion / Adaptive Label Distribution Learning)](#ada-df-adaptive-dual-branch-fusion-adaptive-label-distribution-learning-1) — release_year: 2023 | subcategory: static-image
2. [AffectGPT](#affectgpt-2) — release_year: 2024 | subcategory: mllm
3. [APViT (Attentive Patches Vision Transformer)](#apvit-attentive-patches-vision-transformer-3) — release_year: 2022 | subcategory: static-image | accuracy_rafdb: 91.98% Overall accuracy on RAF-DB basic 7-class subset
4. [CLIPER (CLIP-based Facial Expression Recognition with Parameter-Efficient Prompt Tuning)](#cliper-clip-based-facial-expression-recognition-with-parameter-efficient-prompt-tuning-4) — release_year: 2023 | subcategory: vlm
5. [DAN (Distract Your Attention Network)](#dan-distract-your-attention-network-5) — release_year: 2021 | subcategory: static-image | accuracy_rafdb: 89.70% (Overall accuracy on basic 7-class subset)
6. [DDAMFN (Dual-Direction Attention Mixed Feature Network)](#ddamfn-dual-direction-attention-mixed-feature-network-6) — release_year: 2023 | subcategory: static-image | accuracy_rafdb: 91.35% (overall accuracy on RAF-DB basic 7-class subset)
7. [DFER-CLIP](#dfer-clip-7) — release_year: 2023 | subcategory: dynamic-video | accuracy_rafdb: not reported (DFER-CLIP targets dynamic FER benchmarks, n...
8. [EAC (Erasing Attention Consistency)](#eac-erasing-attention-consistency-8) — release_year: 2022 | subcategory: static-image | accuracy_rafdb: 89.99% Overall on RAF-DB basic subset, ResNet-18 (clean-l... | param_count: ~11.7 (ResNet-18) / ~25.6 (ResNet-50)
9. [EfficientFace](#efficientface-9) — release_year: 2021 | subcategory: static-image | accuracy_rafdb: 88.36% (overall accuracy on RAF-DB basic 7-class subset) | param_count: 1.28M
10. [EmoNeXt](#emonext-10) — release_year: 2023 | subcategory: static-image | param_count: Approximately 28M (Tiny), 50M (Small), 89M (Base), 198M (...
11. [Emotion-LLaMA](#emotion-llama-11) — release_year: 2024 | subcategory: mllm
12. [Face2Exp](#face2exp-12) — release_year: 2022 | subcategory: static-image
13. [FaRL (Facial Representation Learning)](#farl-facial-representation-learning-13) — release_year: 2022 | subcategory: foundation-model
14. [FineCLIPER](#finecliper-14) — release_year: 2024 | subcategory: dynamic-video | accuracy_rafdb: not reported
15. [HiCMAE (Hierarchical Contrastive Masked Autoencoder)](#hicmae-hierarchical-contrastive-masked-autoencoder-15) — release_year: 2024 | subcategory: dynamic-video
16. [LA-Net (Landmark-Aware Network)](#la-net-landmark-aware-network-16) — release_year: 2023 | subcategory: static-image | accuracy_rafdb: 91.56% (Overall accuracy on RAF-DB basic 7-class subset) | param_count: ~12M (ResNet-18 backbone plus auxiliary landmark detectio...
17. [MA-Net (Multi-scale Attention Network)](#ma-net-multi-scale-attention-network-17) — release_year: 2021 | subcategory: static-image | accuracy_rafdb: 88.40% (Overall accuracy on basic 7-class subset)
18. [MAE-DFER](#mae-dfer-18) — release_year: 2023 | subcategory: dynamic-video
19. [POSTER++ (POSTER V2)](#poster-poster-v2-19) — release_year: 2023 | subcategory: static-image | accuracy_rafdb: 92.21% (Overall accuracy on RAF-DB basic 7-class) | param_count: 43.7M
20. [RAN (Region Attention Network)](#ran-region-attention-network-20) — release_year: 2020 | subcategory: static-image | accuracy_rafdb: 86.90% (Overall accuracy on RAF-DB basic subset, ResNet-1...
21. [S2D (Static-to-Dynamic)](#s2d-static-to-dynamic-21) — release_year: 2024 | subcategory: dynamic-video
22. [SCN (Self-Cure Network)](#scn-self-cure-network-22) — release_year: 2020 | subcategory: static-image | accuracy_rafdb: 88.14% (Overall accuracy on RAF-DB basic 7-class with Res... | param_count: ~11.2M (ResNet-18 backbone; SCN modules add a small FC im...
23. [SVFAP (Self-supervised Video Facial Affect Perceiver)](#svfap-self-supervised-video-facial-affect-perceiver-23) — release_year: 2024 | subcategory: dynamic-video
24. [SwinFace](#swinface-24) — release_year: 2023 | subcategory: static-image
25. [TransFER](#transfer-25) — release_year: 2021 | subcategory: static-image | accuracy_rafdb: 90.91% Overall accuracy on RAF-DB basic 7-class subset
26. [VTFF (Visual Transformers with Feature Fusion)](#vtff-visual-transformers-with-feature-fusion-26) — release_year: 2021 | subcategory: static-image | accuracy_rafdb: 88.14% (Overall accuracy on RAF-DB basic 7-class)

### Datasets (24)

27. [AFEW (Acted Facial Expressions in the Wild)](#afew-acted-facial-expressions-in-the-wild-27) — release_year: 2011 | subcategory: dynamic-video | num_images: AFEW 7.0: ~1809 video clips total (Train ~773 / Val ~383 ... | in_the_wild_or_lab: in-the-wild (movie/TV clips)
28. [Aff-Wild2](#aff-wild2-28) — release_year: 2019 | subcategory: dynamic-video | num_images: ~564 in-the-wild YouTube videos comprising approximately ... | in_the_wild_or_lab: in-the-wild
29. [AffectNet](#affectnet-29) — release_year: 2017 | subcategory: static-image | num_images: ~1,000,000 images crawled (≈1M total); ~440,000 (often ci... | in_the_wild_or_lab: in-the-wild
30. [BP4D / BP4D+ (Binghamton-Pittsburgh 4D Spontaneous Expression Database)](#bp4d-bp4d-binghamton-pittsburgh-4d-spontaneous-expression-database-30) — release_year: BP4D: 2013-2014; BP4D+: 2016 | subcategory: au | num_images: BP4D: ~368,000 frames across 328 sequences (41 subjects x... | in_the_wild_or_lab: lab-controlled
31. [BU-3DFE / BU-4DFE](#bu-3dfe-bu-4dfe-31) — release_year: BU-3DFE: 2006; BU-4DFE: 2008 | subcategory: 3d / 4d (dynamic 3d video) | num_images: BU-3DFE: 2,500 static 3D face scans (100 subjects x 6 exp... | in_the_wild_or_lab: lab-controlled
32. [C-EXPR-DB (Compound Expression Database)](#c-expr-db-compound-expression-database-32) — release_year: 2023 | subcategory: dynamic-video | num_images: ~400 videos / ~200,000 annotated frames | in_the_wild_or_lab: in-the-wild (YouTube / web video sources)
33. [CAER-S (Context-Aware Emotion Recognition - Static)](#caer-s-context-aware-emotion-recognition---static-33) — release_year: 2019 | subcategory: static-image (context-aware FER; derived from CAER dynami... | num_images: 70000 | in_the_wild_or_lab: in-the-wild (frames sampled from 79 TV shows)
34. [CAS(ME)^3](#casme3-34) — release_year: 2022 | subcategory: micro-expression | num_images: Approximately 1,109 micro-expression samples and 3,490 ma... | in_the_wild_or_lab: lab-controlled
35. [CK+ (Extended Cohn-Kanade Dataset)](#ck-extended-cohn-kanade-dataset-35) — release_year: 2010 | subcategory: static-image | num_images: 593 video sequences (123 subjects). 327 sequences are emo... | in_the_wild_or_lab: lab-controlled
36. [DFEW (Dynamic Facial Expression in the Wild)](#dfew-dynamic-facial-expression-in-the-wild-36) — release_year: 2020 | subcategory: dynamic-video | num_images: 16,372 video clips selected from over 1,500 movies | in_the_wild_or_lab: in-the-wild
37. [DISFA / DISFA+ (Denver Intensity of Spontaneous Facial Action Database)](#disfa-disfa-denver-intensity-of-spontaneous-facial-action-database-37) — release_year: DISFA: 2013; DISFA+: 2016 | subcategory: au | num_images: DISFA: ~130,000 frames (27 videos, ~4,845 frames per vide... | in_the_wild_or_lab: lab-controlled
38. [EmoSet](#emoset-38) — release_year: 2023 | subcategory: affective-image (visual emotion analysis, broader than face) | num_images: 3.3M images in total (EmoSet-3.3M); 118,102 images carefu... | in_the_wild_or_lab: in-the-wild (collected from social media platforms and st...
39. [EMOTIC (Emotion Recognition in Context)](#emotic-emotion-recognition-in-context-39) — release_year: 2017 | subcategory: context-emotion (static-image, person-in-scene) | num_images: 23,571 images containing 34,320 annotated person instances | in_the_wild_or_lab: in-the-wild
40. [EmotioNet](#emotionet-40) — release_year: 2016 | subcategory: static-image | num_images: approximately 1,000,000 (1M) facial images; ~25,000 manua... | in_the_wild_or_lab: in-the-wild (web-crawled images)
41. [ExpW (Expression in-the-Wild)](#expw-expression-in-the-wild-41) — release_year: 2018 | subcategory: static-image | num_images: 91,793 face images | in_the_wild_or_lab: in-the-wild (web-crawled images)
42. [FER-2013](#fer-2013-42) — release_year: 2013 | subcategory: static-image | num_images: 35887 | in_the_wild_or_lab: in-the-wild
43. [FERV39k](#ferv39k-43) — release_year: 2022 | subcategory: dynamic-video | num_images: 38935 | in_the_wild_or_lab: in-the-wild
44. [JAFFE (Japanese Female Facial Expression)](#jaffe-japanese-female-facial-expression-44) — release_year: 1998 | subcategory: static-image | num_images: 213 | in_the_wild_or_lab: lab-controlled
45. [MAFW (Multi-modal Affective Facial expressions in the Wild)](#mafw-multi-modal-affective-facial-expressions-in-the-wild-45) — release_year: 2022 | subcategory: dynamic-video | num_images: 10045 | in_the_wild_or_lab: in-the-wild
46. [MEAD (Multi-view Emotional Audio-visual Dataset)](#mead-multi-view-emotional-audio-visual-dataset-46) — release_year: 2020 | subcategory: audio-visual | in_the_wild_or_lab: lab-controlled
47. [MMI Facial Expression Database](#mmi-facial-expression-database-47) — release_year: 2005 | subcategory: static-image | in_the_wild_or_lab: lab-controlled
48. [Oulu-CASIA NIR&VIS Facial Expression Database](#oulu-casia-nirvis-facial-expression-database-48) — release_year: 2011 | subcategory: static-image | in_the_wild_or_lab: lab-controlled
49. [RAF-DB (Real-world Affective Faces Database)](#raf-db-real-world-affective-faces-database-49) — release_year: 2017 | subcategory: static-image | num_images: 29672 | in_the_wild_or_lab: in-the-wild
50. [SFEW 2.0 (Static Facial Expressions in the Wild 2.0)](#sfew-20-static-facial-expressions-in-the-wild-20-50) — release_year: 2015 | subcategory: static-image | num_images: 1766 | in_the_wild_or_lab: in-the-wild (frames extracted from feature-length movies,...

---

## Models

### <a id="ada-df-adaptive-dual-branch-fusion-adaptive-label-distribution-learning-1"></a>1. Ada-DF (Adaptive Dual-branch Fusion / Adaptive Label Distribution Learning)

**Common**

- **name**: Ada-DF (Adaptive Dual-branch Fusion / Adaptive Label Distribution Learning)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2023

**Model**

- **architecture_family**: CNN (dual-branch)
- **loss_function**:
  > Composite loss combining Cross-Entropy on the auxiliary branch and KL-divergence label distribution loss on the target branch, with adaptive importance weighting between the two branches to combat label ambiguity
- **training_data_used**: RAF-DB, AffectNet, SFEW 2.0
- **input_modality**: RGB image (aligned face, typically 224x224)
- **dynamic_fer_metrics**: N/A (static-image model)
- **deployment_target**: server-GPU (lightweight ResNet-18 also feasible for edge)

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- authors_or_creators
- backbone
- code_link
- cross_dataset_generalization
- explainability
- hardware_requirements
- inference_fps_or_latency
- paper_link
- param_count
- pretrained_weights_available
- pretraining_corpus
- pretraining_strategy
- publication_venue
- robustness

---

### <a id="affectgpt-2"></a>2. AffectGPT

**Common**

- **name**: AffectGPT
- **type**: model
- **subcategory**: mllm
- **release_year**: 2024
- **authors_or_creators**:
  > Zheng Lian, Licai Sun, Haiyang Sun, Kang Chen, Zhuofan Wen, Hao Gu, Bin Liu, Jianhua Tao; Institute of Automation, Chinese Academy of Sciences (CASIA) and Tsinghua University
- **paper_link**: https://arxiv.org/abs/2306.15401
- **publication_venue**:
  > arXiv preprint (with extension under review / ICML 2024 workshop versions); EMER framework introduced in this preprint line

**Model**

- **architecture_family**: MLLM
- **accuracy_fer2013**:
  > Not reported - model is evaluated on multimodal emotion reasoning (EMER) rather than static FER benchmarks
- **code_link**: https://github.com/zeroQiaoba/AffectGPT
- **loss_function**: Autoregressive language-modeling cross-entropy loss for instruction tuning on EMER explanation data
- **training_data_used**:
  > EMER (Explainable Multimodal Emotion Reasoning) dataset built on MER2023 with manually annotated multimodal clues and natural-language emotion rationales; supplementary multimodal emotion datasets
- **pretraining_strategy**:
  > VLM-pretrain - leverages pretrained vision/audio encoders and an instruction-tuned LLM, then fine-tuned on EMER explanation data
- **input_modality**: audio-visual + text (multimodal: video frames, audio, subtitle/transcript text)
- **abaw_metrics**: Not the target benchmark - not reported
- **cross_dataset_generalization**:
  > Demonstrates generalization across MER2023 and EMER; framework designed for open-vocabulary explainable emotion recognition
- **explainability**:
  > MLLM rationale support - core contribution is generating natural-language explanations / reasoning chains identifying multimodal emotional cues alongside predicted emotion labels (Explainable Multimodal Emotion Reasoning, EMER)
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_rafdb
- backbone
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- pretraining_corpus
- robustness

---

### <a id="apvit-attentive-patches-vision-transformer-3"></a>3. APViT (Attentive Patches Vision Transformer)

**Common**

- **name**: APViT (Attentive Patches Vision Transformer)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2022
- **authors_or_creators**:
  > Fanglei Xue, Qiangchang Wang, Zichang Tan, Zhongsong Ma, Guodong Guo (Institute of Deep Learning, Baidu Research)
- **paper_link**: https://arxiv.org/abs/2212.05463
- **publication_venue**: IEEE Transactions on Affective Computing (TAC), 2022

**Model**

- **architecture_family**: hybrid
- **backbone**:
  > IR-50 (CNN stem) + Vision Transformer with Attentive Patch Selection (APS) and Attentive Pooling (AP)
- **accuracy_affectnet**: 66.91% Overall (WAR) on AffectNet-7 class setting
- **accuracy_rafdb**: 91.98% Overall accuracy on RAF-DB basic 7-class subset
- **code_link**: https://github.com/youqingxiaozhua/APViT
- **pretrained_weights_available**:
  > yes (released via the official GitHub repository, including IR-50 face-recognition pretrained weights and APViT FER checkpoints)
- **loss_function**:
  > Softmax cross-entropy classification loss; attentive patch/pooling modules trained jointly end-to-end (no auxiliary loss reported as primary; follow-up to TransFER-style attention regularization)
- **training_data_used**: RAF-DB, AffectNet (7-class), FERPlus; IR-50 backbone pretrained on MS-Celeb-1M
- **pretraining_strategy**: face-recognition-pretrain (IR-50 backbone pretrained for face recognition on MS-Celeb-1M)
- **pretraining_corpus**: MS-Celeb-1M
- **input_modality**: RGB image
- **dynamic_fer_metrics**: not applicable (static-image model)
- **abaw_metrics**: not reported
- **explainability**:
  > Attention/patch-importance visualizations supported (APS scores and AP weights provide interpretable evidence of which patches drive the prediction)
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- robustness

---

### <a id="cliper-clip-based-facial-expression-recognition-with-parameter-efficient-prompt-tuning-4"></a>4. CLIPER (CLIP-based Facial Expression Recognition with Parameter-Efficient Prompt Tuning)

**Common**

- **name**: CLIPER (CLIP-based Facial Expression Recognition with Parameter-Efficient Prompt Tuning)
- **type**: model
- **subcategory**: vlm
- **release_year**: 2023
- **authors_or_creators**:
  > Hanting Li, Hongjing Niu, Zhaoqing Zhu, Feng Zhao (University of Science and Technology of China, USTC)
- **paper_link**: https://arxiv.org/abs/2303.00193

**Model**

- **architecture_family**: VLM
- **backbone**: CLIP ViT-B/16 image encoder + CLIP Transformer text encoder (frozen, with learnable prompts)
- **loss_function**:
  > Cross-entropy classification loss over similarity scores between image features and class-specific text-prompt embeddings; two-stage training with multi-modal prompt learning (coarse-grained then fine-grained class-aware prompts)
- **training_data_used**: RAF-DB, AffectNet (7- and 8-class), FERPlus
- **pretraining_strategy**:
  > VLM-pretrain (initialized from CLIP pretrained on 400M image-text pairs); FER adaptation uses parameter-efficient prompt tuning rather than full fine-tuning
- **pretraining_corpus**: WIT (WebImageText, ~400M image-text pairs collected by OpenAI for CLIP)
- **input_modality**: RGB image + text-conditioned class prompts
- **dynamic_fer_metrics**: not applicable (static-image model)
- **abaw_metrics**: not reported
- **explainability**:
  > Class-aware text prompts provide semantic interpretability of the decision (which textual concept best matches the image); CLIP attention maps can be visualized
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- code_link
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- publication_venue
- robustness

---

### <a id="dan-distract-your-attention-network-5"></a>5. DAN (Distract Your Attention Network)

**Common**

- **name**: DAN (Distract Your Attention Network)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2021
- **authors_or_creators**:
  > Zhengyao Wen, Wenzhong Lin, Tao Wang, Ge Xu (South China University of Technology / affiliated institutions)
- **paper_link**: https://arxiv.org/abs/2109.07270
- **publication_venue**:
  > arXiv preprint (2021); later journal version: Biomimetics 2023, 8(2), 199 (MDPI), DOI 10.3390/biomimetics8020199

**Model**

- **architecture_family**: CNN
- **backbone**: ResNet-18
- **accuracy_affectnet**: 65.69% on AffectNet-7 (Overall/WAR); 62.09% on AffectNet-8 (Overall/WAR)
- **accuracy_rafdb**: 89.70% (Overall accuracy on basic 7-class subset)
- **code_link**: https://github.com/yaoing/DAN
- **pretrained_weights_available**:
  > yes - pretrained checkpoints on RAF-DB and AffectNet are released via the official GitHub repository (Google Drive links in README)
- **loss_function**:
  > Composite loss = Cross-Entropy (classification) + Affinity Loss (FCN, large-margin class clustering) + Partition Loss (AFN, encourages attention heads to attend to distinct regions)
- **training_data_used**: Trained/evaluated on RAF-DB, AffectNet (7 and 8 class), and SFEW 2.0
- **pretraining_strategy**: face-recognition-pretrain (backbone initialized from ResNet-18 pretrained on MS-Celeb-1M)
- **pretraining_corpus**: MS-Celeb-1M
- **input_modality**: RGB image (aligned/cropped face, typically 224x224)
- **dynamic_fer_metrics**: N/A (static-image model)
- **explainability**:
  > Attention map visualization (multi-head attention heatmaps over facial regions) supported natively by the MAN/AFN modules
- **deployment_target**: server-GPU (also feasible on edge/mobile due to lightweight ResNet-18 backbone)

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- robustness

---

### <a id="ddamfn-dual-direction-attention-mixed-feature-network-6"></a>6. DDAMFN (Dual-Direction Attention Mixed Feature Network)

**Common**

- **name**: DDAMFN (Dual-Direction Attention Mixed Feature Network)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2023
- **authors_or_creators**:
  > Saining Zhang, Yuhang Zhang, Ye Liu, Heng Su (and collaborators); MDPI Electronics 2023. Affiliations include Beijing institutions (Beijing Jiaotong University / Beijing University of Posts and Telecommunications collaborators).
- **paper_link**: https://www.mdpi.com/2079-9292/12/17/3595
- **publication_venue**: MDPI Electronics, Vol. 12, Issue 17, 3595 (2023)

**Model**

- **architecture_family**: CNN
- **backbone**:
  > MixedFeatureNet (MFN) — a MobileFaceNet-inspired lightweight backbone with mixed depthwise/pointwise feature blocks
- **accuracy_affectnet**: AffectNet-7: 67.03% (overall accuracy); AffectNet-8: 64.25% (overall accuracy)
- **accuracy_rafdb**: 91.35% (overall accuracy on RAF-DB basic 7-class subset)
- **code_link**: https://github.com/SainingZhang/DDAMFN
- **pretrained_weights_available**:
  > yes — pretrained weights for RAF-DB / AffectNet / FERPlus released via the official GitHub repository
- **loss_function**:
  > Combination of Cross-Entropy classification loss with an Attention Loss (att_loss) that encourages diversity/orthogonality among the multi-head dual-direction attention maps
- **training_data_used**: RAF-DB, AffectNet (7-class and 8-class), FERPlus — trained separately per benchmark
- **pretraining_strategy**:
  > face-recognition-pretrain — MFN backbone is pretrained on a large face recognition corpus (MS-Celeb-1M / refined face dataset) before fine-tuning on FER datasets
- **input_modality**: RGB image (aligned face crop, typically 112x112)
- **dynamic_fer_metrics**: Not applicable — DDAMFN is a static-image FER model and does not report DFEW/FERV39k/MAFW results
- **abaw_metrics**: Not reported in canonical paper
- **explainability**:
  > Yes — the dual-direction attention maps (horizontal and vertical attention) can be visualized as attention heatmaps, providing interpretability
- **deployment_target**: edge / mobile — designed as a lightweight FER model suitable for resource-constrained deployment

**Uncertain (skipped) fields**

- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretraining_corpus
- robustness

---

### <a id="dfer-clip-7"></a>7. DFER-CLIP

**Common**

- **name**: DFER-CLIP
- **type**: model
- **subcategory**: dynamic-video
- **release_year**: 2023
- **authors_or_creators**: Zengqun Zhao and Ioannis Patras, Queen Mary University of London (QMUL)
- **paper_link**: https://arxiv.org/abs/2308.13382
- **publication_venue**: BMVC 2023 (British Machine Vision Conference)

**Model**

- **architecture_family**: VLM
- **backbone**: CLIP ViT-B/32 (image encoder) with Transformer-based temporal module; CLIP text encoder
- **accuracy_rafdb**: not reported (DFER-CLIP targets dynamic FER benchmarks, not RAF-DB)
- **code_link**: https://github.com/zengqunzhao/DFER-CLIP
- **pretrained_weights_available**:
  > yes - released via the official GitHub repository (CLIP-ViT-B/32 based checkpoints for DFEW, FERV39k, MAFW)
- **loss_function**:
  > Cross-entropy classification loss combined with CLIP-style image-text contrastive alignment between the temporal visual feature and learnable class-name text prompts
- **training_data_used**:
  > DFEW, FERV39k, MAFW (each benchmark trained on its own training split using its standard protocol, e.g., 5-fold cross-validation on DFEW)
- **pretraining_strategy**: VLM-pretrain (initialized from CLIP pretrained weights); fine-tuned on dynamic FER datasets
- **pretraining_corpus**:
  > OpenAI CLIP WIT (400M image-text pairs) used for CLIP pretraining; downstream FER training on DFEW/FERV39k/MAFW
- **input_modality**: video clip (RGB frames) with text-conditioned class descriptors (learnable text prompts)
- **abaw_metrics**: not reported
- **explainability**:
  > Attention maps from the temporal Transformer can be visualized; learnable text prompts provide class-level semantic interpretability
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- cross_dataset_generalization
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count
- robustness

---

### <a id="eac-erasing-attention-consistency-8"></a>8. EAC (Erasing Attention Consistency)

**Common**

- **name**: EAC (Erasing Attention Consistency)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2022
- **authors_or_creators**:
  > Yuhang Zhang, Chengrui Wang, Xu Ling, Weihong Deng - Beijing University of Posts and Telecommunications (BUPT), Pattern Recognition and Intelligent System Laboratory
- **paper_link**: https://arxiv.org/abs/2207.10299
- **publication_venue**: ECCV 2022 (European Conference on Computer Vision)

**Model**

- **architecture_family**: CNN
- **backbone**: ResNet-18 (primary); ResNet-50 also reported
- **param_count**: ~11.7 (ResNet-18) / ~25.6 (ResNet-50)
- **accuracy_rafdb**:
  > 89.99% Overall on RAF-DB basic subset, ResNet-18 (clean-label setting); strong results also under 10/20/30% synthetic label-noise injections
- **code_link**: https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency
- **loss_function**:
  > Combination of (1) standard cross-entropy classification loss, (2) flip attention consistency loss (KL/L2 between attention maps of an image and its horizontal flip), and (3) a CAM-erasing branch that randomly erases the most-attended region and enforces label consistency on the erased image to prevent overfitting to noisy labels
- **training_data_used**:
  > RAF-DB, AffectNet (7-class and 8-class), FERPlus; evaluated under both clean and synthetic noisy-label (10%/20%/30%) protocols
- **pretraining_strategy**:
  > supervised-pretrain - backbone initialized from MS-Celeb-1M face-recognition-pretrained weights (following the standard FER pipeline), with ImageNet weights also reported
- **pretraining_corpus**: MS-Celeb-1M (face recognition) and/or ImageNet
- **input_modality**: RGB image (plus its horizontal flip used during training for the attention-consistency branch)
- **dynamic_fer_metrics**: N/A - static-image model
- **abaw_metrics**: N/A
- **robustness**:
  > Strong noisy-label robustness: outperforms prior noisy-label FER methods (SCN, RUL, DMUE) on RAF-DB, AffectNet, FERPlus under 10%/20%/30% synthetic label noise. Also robust on the real noisy WebEmotion / asymmetric-noise settings reported in the paper
- **explainability**:
  > Built around Class Activation Maps (CAMs) and attention visualization - the method explicitly produces and constrains attention maps, making attention behavior directly inspectable
- **deployment_target**:
  > server-GPU (also suitable for edge given the lightweight ResNet-18 backbone and zero inference-time overhead from the consistency mechanism)

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- pretrained_weights_available

---

### <a id="efficientface-9"></a>9. EfficientFace

**Common**

- **name**: EfficientFace
- **type**: model
- **subcategory**: static-image
- **release_year**: 2021
- **authors_or_creators**:
  > Zengqun Zhao, Qingshan Liu, Feng Zhou; Nanjing University of Information Science and Technology (and collaborators)
- **paper_link**: https://arxiv.org/abs/2103.06401
- **publication_venue**: AAAI 2021 (Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35)

**Model**

- **architecture_family**: CNN
- **backbone**:
  > Modified ShuffleNetV2 (1.0x) augmented with a Local-Feature Extractor (LFE) and Channel-Spatial Modulator (CSM)
- **param_count**: 1.28M
- **accuracy_affectnet**: AffectNet-7: 63.70% (overall accuracy / WAR); AffectNet-8: 59.89% (overall accuracy / WAR)
- **accuracy_rafdb**: 88.36% (overall accuracy on RAF-DB basic 7-class subset)
- **code_link**: https://github.com/zengqunzhao/EfficientFace
- **pretrained_weights_available**:
  > yes — pretrained weights provided via the official GitHub repository (RAF-DB, AffectNet, CAER-S checkpoints)
- **hardware_requirements**:
  > Training: single GPU (e.g., NVIDIA GTX 1080 Ti / RTX 2080) sufficient; Inference: capable of running on mobile / CPU due to small parameter count and low FLOPs
- **loss_function**:
  > Cross-entropy classification loss combined with a label-distribution-aware loss; the paper introduces a label distribution learning component to handle ambiguous expression labels
- **training_data_used**: RAF-DB, AffectNet (7-class and 8-class), CAER-S — evaluated separately per benchmark
- **pretraining_strategy**:
  > face-recognition-pretrain — backbone pretrained on MS-Celeb-1M for face recognition before FER fine-tuning
- **pretraining_corpus**: MS-Celeb-1M (face recognition pretraining)
- **input_modality**: RGB image (aligned face crop, 224x224)
- **dynamic_fer_metrics**:
  > Not applicable for the original paper — EfficientFace is a static-image FER model. (Some follow-up works have extended/applied EfficientFace features to dynamic FER benchmarks like DFEW.)
- **abaw_metrics**: Not reported in canonical paper
- **explainability**: Supports attention/feature visualization via the Channel-Spatial Modulator; Grad-CAM compatible
- **deployment_target**: mobile / edge — explicitly targeted at mobile deployment with ~1.28M parameters

**Uncertain (skipped) fields**

- accuracy_fer2013
- cross_dataset_generalization
- inference_fps_or_latency
- robustness

---

### <a id="emonext-10"></a>10. EmoNeXt

**Common**

- **name**: EmoNeXt
- **type**: model
- **subcategory**: static-image
- **release_year**: 2023
- **authors_or_creators**:
  > Yassine El Boudouri and Amine Bouridane. Centre for Data Analytics and Cybersecurity (CDAC), University of Sharjah, UAE.
- **paper_link**: https://arxiv.org/abs/2310.06868
- **publication_venue**: IEEE 25th International Workshop on Multimedia Signal Processing (MMSP), 2023

**Model**

- **architecture_family**: CNN (ConvNeXt-based with Spatial Transformer Network and Squeeze-and-Excitation blocks)
- **backbone**:
  > ConvNeXt (variants: Tiny / Small / Base / Large) augmented with a Spatial Transformer Network (STN) front-end and SE attention blocks
- **param_count**:
  > Approximately 28M (Tiny), 50M (Small), 89M (Base), 198M (Large) — based on ConvNeXt variants with added STN/SE modules
- **accuracy_fer2013**: 76.12% (EmoNeXt-Large) on FER2013 test set; smaller variants ~74-75%
- **code_link**: https://github.com/yelboudouri/EmoNeXt
- **hardware_requirements**: Trained on NVIDIA GPUs; ConvNeXt-Large variant requires substantial GPU memory (>=16GB for training)
- **loss_function**:
  > Cross-entropy loss combined with a self-attention regularization term (encouraging attention diversity in the STN module)
- **training_data_used**: FER2013 (primary benchmark); ConvNeXt backbones initialized from ImageNet-pretrained weights
- **pretraining_strategy**: supervised-pretrain (ConvNeXt backbone pretrained on ImageNet-1k/22k)
- **pretraining_corpus**: ImageNet (ConvNeXt backbone)
- **input_modality**: RGB image
- **dynamic_fer_metrics**: not applicable — EmoNeXt is a static-image FER model
- **abaw_metrics**: not reported
- **explainability**:
  > Spatial Transformer Network produces interpretable spatial attention/transformation; attention maps can be visualized
- **deployment_target**:
  > server-GPU (especially Base/Large variants); Tiny variant could be deployed on edge with optimization

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_rafdb
- cross_dataset_generalization
- inference_fps_or_latency
- pretrained_weights_available
- robustness

---

### <a id="emotion-llama-11"></a>11. Emotion-LLaMA

**Common**

- **name**: Emotion-LLaMA
- **type**: model
- **subcategory**: mllm
- **release_year**: 2024
- **authors_or_creators**:
  > Zebang Cheng, Zhi-Qi Cheng, Jun-Yan He, Kai Wang, Yuxiang Lin, Zheng Lian, Xiaojiang Peng, Alexander Hauptmann; Shenzhen Technology University, Carnegie Mellon University, Institute of Automation Chinese Academy of Sciences
- **paper_link**: https://arxiv.org/abs/2406.11161
- **publication_venue**: NeurIPS 2024

**Model**

- **architecture_family**: MLLM
- **backbone**:
  > LLaMA-2-7B (LLM); HuBERT-large (audio encoder); EVA (visual frame encoder); MAE and VideoMAE (facial expression encoders)
- **code_link**: https://github.com/ZebangCheng/Emotion-LLaMA
- **pretrained_weights_available**: yes - available via the official GitHub repository and Hugging Face
- **loss_function**: Cross-entropy / autoregressive language modeling loss for instruction tuning over the MERR dataset
- **training_data_used**:
  > MERR (Multimodal Emotion Recognition and Reasoning) instruction-tuning dataset (~28K coarse-grained + ~4.6K fine-grained samples) constructed from MER2023 and other multimodal emotion sources
- **pretraining_strategy**:
  > VLM-pretrain followed by instruction tuning; uses pretrained HuBERT, EVA, MAE, VideoMAE encoders aligned with LLaMA-2 via projection layers, then instruction-tuned on MERR
- **pretraining_corpus**:
  > LLaMA-2 pretraining corpus (LLM); HuBERT pretraining (LibriSpeech); EVA pretraining (LAION/ImageNet); MAE/VideoMAE on face / video data
- **input_modality**: audio-visual (audio waveform + video frames + facial expression frames) with text instructions
- **abaw_metrics**: Not the primary benchmark; reported MER2024 challenge metrics instead
- **cross_dataset_generalization**:
  > Strong zero-shot generalization across MER2023, MER2024-NOISE, MER2024-OV, DFEW; SOTA on MER2024-NOISE and MER2024-OV tracks
- **robustness**: SOTA on MER2024-NOISE track which evaluates robustness to noisy / corrupted audio-visual input
- **explainability**:
  > MLLM rationale support - generates natural-language emotional reasoning and descriptions of visual/audio cues alongside predicted emotion labels
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count

---

### <a id="face2exp-12"></a>12. Face2Exp

**Common**

- **name**: Face2Exp
- **type**: model
- **subcategory**: static-image
- **release_year**: 2022
- **authors_or_creators**:
  > Dan Zeng, Zhiyuan Lin, Xiao Yan, Yuting Liu, Fei Wang, Bo Tang. Lead authors associated with Wuhan University and collaborators.
- **paper_link**:
  > https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_Face2Exp_Combating_Data_Biases_for_Facial_Expression_Recognition_CVPR_2022_paper.pdf
- **publication_venue**: CVPR 2022

**Model**

- **architecture_family**: CNN
- **backbone**: ResNet-18 / ResNet-50 (face-recognition pretrained)
- **loss_function**:
  > Cross-entropy combined with meta-learning (Meta-Face2Exp) objective; uses a base network and an adaptation network with knowledge distillation/balancing terms to combat class imbalance.
- **training_data_used**:
  > RAF-DB and AffectNet for FER training; large-scale face recognition dataset (e.g., MS-Celeb-1M) for face-recognition pretraining.
- **pretraining_strategy**:
  > face-recognition-pretrain combined with meta-learning fine-tuning; uses knowledge from a face-recognition pretrained model to debias the FER classifier.
- **input_modality**: RGB image
- **dynamic_fer_metrics**: N/A (static-image model)
- **abaw_metrics**: N/A
- **robustness**:
  > Designed to be robust to class imbalance and distribution bias; specifically targets the long-tailed/imbalanced label distributions in AffectNet and RAF-DB. Occlusion/pose robustness not specifically reported.
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- code_link
- cross_dataset_generalization
- explainability
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- pretraining_corpus

---

### <a id="farl-facial-representation-learning-13"></a>13. FaRL (Facial Representation Learning)

**Common**

- **name**: FaRL (Facial Representation Learning)
- **type**: model
- **subcategory**: foundation-model
- **release_year**: 2022
- **authors_or_creators**:
  > Yinglin Zheng, Hao Yang, Ting Zhang, Jianmin Bao, Dongdong Chen, Yangyu Huang, Lu Yuan, Dong Chen, Ming Zeng, Fang Wen. Microsoft Research Asia (with Xiamen University collaboration).
- **paper_link**: https://arxiv.org/abs/2112.03109
- **publication_venue**: CVPR 2022

**Model**

- **architecture_family**: VLM (Vision-Language pretraining; CLIP-style hybrid)
- **backbone**: ViT-B/16 (image encoder) + Transformer text encoder (CLIP-style); also released ViT-L/14 variants.
- **code_link**: https://github.com/FacePerceiver/FaRL
- **pretrained_weights_available**: yes - pretrained weights released on the FaRL GitHub repository (multiple epochs/variants).
- **hardware_requirements**:
  > Pretraining required multi-GPU clusters (large-batch contrastive training on 20M pairs). Inference: single modern GPU sufficient for ViT-B/16 backbone.
- **loss_function**:
  > Image-text contrastive loss (CLIP-style InfoNCE) combined with masked image modeling (MIM) objective for low-level facial representation learning.
- **training_data_used**:
  > LAION-FACE (20M face image-text pairs filtered from LAION-400M). Downstream evaluation on face parsing (LaPa, CelebAMask-HQ), alignment (300W, WFLW), AU detection (BP4D), and expression recognition tasks.
- **pretraining_strategy**:
  > VLM-pretrain (vision-language contrastive) combined with self-supervised masked image modeling on facial images.
- **pretraining_corpus**: LAION-FACE 20M (subset of LAION-400M filtered for facial images with text captions).
- **input_modality**: RGB image (with text-conditioned pretraining)
- **dynamic_fer_metrics**: N/A (foundation model; not benchmarked on dynamic FER in the original paper)
- **cross_dataset_generalization**:
  > Strong transfer reported across many face tasks (parsing, alignment, AU detection); generalizes well as a face foundation model. Specific FER cross-dataset numbers not reported in original paper.
- **robustness**:
  > Pretraining on diverse in-the-wild LAION-FACE images provides robustness to varied lighting, pose, and occlusion; downstream task performance demonstrates strong generalization. No specific Occlusion-RAF-DB / Pose-RAF-DB numbers reported.
- **explainability**: Attention maps from ViT can be visualized; no MLLM-style rationale support.
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- inference_fps_or_latency
- param_count

---

### <a id="finecliper-14"></a>14. FineCLIPER

**Common**

- **name**: FineCLIPER
- **type**: model
- **subcategory**: dynamic-video
- **release_year**: 2024
- **paper_link**: https://arxiv.org/abs/2407.02157
- **publication_venue**: ACM Multimedia (ACM MM) 2024

**Model**

- **architecture_family**: VLM
- **backbone**:
  > CLIP ViT-B/16 image encoder (with adapters/parameter-efficient tuning) + CLIP text encoder; uses a face-aware module and temporal modeling on top
- **accuracy_fer2013**: not reported
- **accuracy_affectnet**: not reported (FineCLIPER targets dynamic FER benchmarks)
- **accuracy_rafdb**: not reported
- **loss_function**:
  > Multi-granularity cross-modal contrastive (image-text) alignment loss combined with cross-entropy classification; aligns text and dynamic facial features at multiple granularities (class-level, description-level, fine-grained AU/landmark-level)
- **training_data_used**: DFEW, FERV39k, MAFW (each trained with the standard benchmark protocol)
- **pretraining_strategy**:
  > VLM-pretrain (CLIP initialization) followed by parameter-efficient fine-tuning with multi-modal multi-granularity supervision
- **pretraining_corpus**: OpenAI CLIP WIT (400M image-text pairs); fine-tuned on DFEW/FERV39k/MAFW
- **input_modality**:
  > video clip (RGB frames) plus text descriptors at multiple granularities (coarse class names and fine-grained AU/landmark/face-region descriptions); optionally augmented with face-parsing/landmark cues
- **abaw_metrics**: not reported
- **explainability**:
  > Multi-granularity text alignment (including AU/landmark-level cues) provides semantic interpretability; attention/CAM visualizations possible via the CLIP image encoder
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- authors_or_creators
- code_link
- cross_dataset_generalization
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- robustness

---

### <a id="hicmae-hierarchical-contrastive-masked-autoencoder-15"></a>15. HiCMAE (Hierarchical Contrastive Masked Autoencoder)

**Common**

- **name**: HiCMAE (Hierarchical Contrastive Masked Autoencoder)
- **type**: model
- **subcategory**: dynamic-video
- **release_year**: 2024
- **authors_or_creators**:
  > Licai Sun, Zheng Lian, Bin Liu, Jianhua Tao (Institute of Automation, Chinese Academy of Sciences; Tsinghua University)
- **paper_link**: https://arxiv.org/abs/2401.05698

**Model**

- **architecture_family**: self-supervised
- **backbone**:
  > Hierarchical dual-branch Transformer (Video ViT for visual stream + Audio ViT for audio stream, with cross-modal fusion blocks)
- **code_link**: https://github.com/sunlicai/HiCMAE
- **pretrained_weights_available**: yes - released on the official GitHub repository
- **loss_function**:
  > Combined objective: (1) masked reconstruction loss (MSE) on visual and audio modalities, (2) intra-modal contrastive loss, and (3) cross-modal (inter-modal) contrastive loss applied at multiple hierarchical levels; cross-entropy for downstream classification
- **training_data_used**:
  > Pretraining: large-scale audio-visual face/voice corpus (VoxCeleb2 audio+visual, ~1M clips). Finetuning: MAFW, DFEW, MER2023, CREMA-D, RAVDESS, MSP-IMPROV, IEMOCAP and similar audio-visual emotion datasets
- **pretraining_strategy**:
  > self-supervised (masked autoencoding combined with hierarchical multi-level contrastive learning across audio and visual modalities)
- **pretraining_corpus**: VoxCeleb2 (audio-visual)
- **input_modality**: audio-visual (synchronized face video + speech audio / mel-spectrogram)
- **explainability**: Attention visualizations across audio and visual streams discussed in paper
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- cross_dataset_generalization
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count
- publication_venue
- robustness

---

### <a id="la-net-landmark-aware-network-16"></a>16. LA-Net (Landmark-Aware Network)

**Common**

- **name**: LA-Net (Landmark-Aware Network)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2023
- **authors_or_creators**: Zhiyu Wu, Jinshi Cui (Peking University, China)
- **paper_link**: https://arxiv.org/abs/2308.13234
- **publication_venue**: ICCV 2023 (IEEE/CVF International Conference on Computer Vision)

**Model**

- **architecture_family**: CNN (with auxiliary landmark detection branch)
- **backbone**: ResNet-18
- **accuracy_rafdb**: 91.56% (Overall accuracy on RAF-DB basic 7-class subset)
- **loss_function**:
  > Multi-task loss combining Cross-Entropy for expression classification, landmark detection regression loss (auxiliary task), and a landmark-guided attention regularization to combat label noise
- **training_data_used**:
  > RAF-DB, AffectNet, FERPlus (with auxiliary landmark supervision; landmarks obtained from face alignment toolkits)
- **pretraining_strategy**: supervised-pretrain (ImageNet) for ResNet-18 backbone; landmark branch jointly trained
- **input_modality**:
  > RGB image with auxiliary facial landmark supervision (RGB+landmarks during training; RGB-only at inference)
- **dynamic_fer_metrics**: N/A (static-image model)
- **robustness**:
  > Explicitly designed to combat label noise via landmark-guided attention; reports robustness to synthetic label noise (e.g. 10%/20%/30% flip noise on RAF-DB) and to occlusion/pose variants
- **explainability**:
  > Attention maps (landmark-guided attention) provide interpretability of where the model focuses on the face
- **deployment_target**: server-GPU (lightweight ResNet-18 also feasible for edge/mobile inference)

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_affectnet
- accuracy_fer2013
- code_link
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- pretraining_corpus

---

### <a id="ma-net-multi-scale-attention-network-17"></a>17. MA-Net (Multi-scale Attention Network)

**Common**

- **name**: MA-Net (Multi-scale Attention Network)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2021
- **authors_or_creators**: Zengqun Zhao, Qingshan Liu, Shanmin Wang (Nanjing University of Information Science and Technology)
- **paper_link**: https://doi.org/10.1109/TIP.2021.3093397
- **publication_venue**: IEEE Transactions on Image Processing (TIP), vol. 30, pp. 6544-6556, 2021

**Model**

- **architecture_family**: CNN
- **backbone**:
  > ResNet-based feature pre-extractor (modified ResNet-18 / ResNet variant) with multi-scale module and local attention module
- **accuracy_affectnet**: 64.53% on AffectNet-7 (Overall/WAR); 60.29% on AffectNet-8 (Overall/WAR)
- **accuracy_rafdb**: 88.40% (Overall accuracy on basic 7-class subset)
- **code_link**: https://github.com/zengqunzhao/MA-Net
- **pretrained_weights_available**: yes - pretrained models and training logs released on the official GitHub repository
- **loss_function**: Cross-entropy classification loss (standard softmax CE)
- **training_data_used**: Trained/evaluated on RAF-DB, AffectNet (7 and 8 class), CAER-S, and SFEW 2.0
- **input_modality**: RGB image (aligned/cropped face, typically 224x224)
- **dynamic_fer_metrics**: N/A (static-image model)
- **explainability**: Attention map visualization via local attention module (visualized in the paper)
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretraining_corpus
- pretraining_strategy
- robustness

---

### <a id="mae-dfer-18"></a>18. MAE-DFER

**Common**

- **name**: MAE-DFER
- **type**: model
- **subcategory**: dynamic-video
- **release_year**: 2023
- **authors_or_creators**:
  > Licai Sun, Zheng Lian, Bin Liu, Jianhua Tao. Institute of Automation, Chinese Academy of Sciences (CASIA) and University of Chinese Academy of Sciences.
- **paper_link**: https://arxiv.org/abs/2307.02227
- **publication_venue**: ACM Multimedia (ACM MM) 2023

**Model**

- **architecture_family**: self-supervised
- **backbone**: ViT-B/16 (with Local-Global Interaction Transformer / LGI-Former decoder design)
- **code_link**: https://github.com/sunlicai/MAE-DFER
- **pretrained_weights_available**:
  > yes - VoxCeleb2 self-supervised pretrained weights and DFEW/FERV39k/MAFW fine-tuned checkpoints released on the GitHub repo
- **loss_function**:
  > Self-supervised pretraining: pixel reconstruction loss (MSE) on masked patches (MAE-style) plus an auxiliary contrastive/feature reconstruction objective from a teacher (VideoMAE-style). Fine-tuning: cross-entropy.
- **training_data_used**:
  > Pretraining: VoxCeleb2 (~1M+ unlabeled face video clips). Fine-tuning: DFEW, FERV39k, MAFW (downstream dynamic FER benchmarks).
- **pretraining_strategy**:
  > self-supervised (Masked Autoencoder, video MAE variant) with a Local-Global Interaction (LGI-Former) decoder for efficient spatiotemporal modeling
- **pretraining_corpus**: VoxCeleb2 (large-scale unlabeled talking-face video corpus)
- **input_modality**: video clip (RGB face frames; typically 16 frames at 224x224)
- **abaw_metrics**: not reported
- **robustness**:
  > not explicitly evaluated on Occlusion-RAF-DB / Pose-RAF-DB; self-supervised pretraining argued to improve robustness to limited labeled data
- **explainability**: attention map visualizations shown qualitatively in the paper
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- cross_dataset_generalization
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count

---

### <a id="poster-poster-v2-19"></a>19. POSTER++ (POSTER V2)

**Common**

- **name**: POSTER++ (POSTER V2)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2023
- **authors_or_creators**:
  > Jiawei Mao, Rui Xu, Xuesong Yin, Yuanqi Chang, Binling Nie, Aibin Huang. Hangzhou Dianzi University / Zhejiang University City College, China.
- **paper_link**: https://arxiv.org/abs/2301.12149
- **publication_venue**:
  > arXiv preprint (2023); follow-up to POSTER (ICCVW 2023). Published in Pattern Recognition (Elsevier) 2024.

**Model**

- **architecture_family**: hybrid (CNN + Transformer)
- **backbone**: IR-50 (image stream) + MobileFaceNet (landmark stream) + Pyramid Cross-Fusion Transformer
- **param_count**: 43.7M
- **accuracy_affectnet**: 63.77% on AffectNet-8 (Overall/WAR); 67.49% on AffectNet-7 (Overall/WAR)
- **accuracy_rafdb**: 92.21% (Overall accuracy on RAF-DB basic 7-class)
- **code_link**: https://github.com/Talented-Q/POSTER_V2
- **pretrained_weights_available**:
  > yes — pretrained weights for RAF-DB, AffectNet-7, and AffectNet-8 are released on the official GitHub repository
- **hardware_requirements**:
  > Training on NVIDIA GPUs (paper trains on a single GPU, e.g., RTX 3090 / V100); inference feasible on consumer GPUs
- **loss_function**: Cross-entropy classification loss
- **training_data_used**:
  > RAF-DB, AffectNet-7, AffectNet-8 (separate models trained per benchmark). IR-50 backbone pretrained on MS-Celeb-1M; MobileFaceNet pretrained on face recognition data.
- **pretraining_strategy**: face-recognition-pretrain (IR-50 and MobileFaceNet are pretrained on face recognition tasks)
- **pretraining_corpus**: MS-Celeb-1M (for IR-50 backbone); face recognition corpus for MobileFaceNet
- **input_modality**: RGB image (with implicit landmark features extracted via MobileFaceNet)
- **dynamic_fer_metrics**: not applicable — POSTER++ is a static-image FER model
- **abaw_metrics**: not reported in the original paper
- **explainability**:
  > Attention maps from the cross-fusion transformer can be visualized; Grad-CAM-style visualizations shown in the paper
- **deployment_target**: server-GPU (43.7M params, transformer-based)

**Uncertain (skipped) fields**

- accuracy_fer2013
- cross_dataset_generalization
- inference_fps_or_latency
- robustness

---

### <a id="ran-region-attention-network-20"></a>20. RAN (Region Attention Network)

**Common**

- **name**: RAN (Region Attention Network)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2020
- **authors_or_creators**:
  > Kai Wang, Xiaojiang Peng, Jianfei Yang, Debin Meng, Yu Qiao - Shenzhen Institutes of Advanced Technology (SIAT), Chinese Academy of Sciences
- **paper_link**: https://arxiv.org/abs/1905.04075
- **publication_venue**: IEEE Transactions on Image Processing (TIP) 2020

**Model**

- **architecture_family**: CNN
- **backbone**: ResNet-18 (also reports VGG-16 variants)
- **accuracy_rafdb**: 86.90% (Overall accuracy on RAF-DB basic subset, ResNet-18 backbone)
- **code_link**: https://github.com/kaiwang960112/Challenge-condition-FER-dataset
- **loss_function**:
  > Region Biased Loss (RB-Loss) ensuring the most discriminative region's attention weight exceeds other regions, combined with standard cross-entropy classification loss
- **training_data_used**:
  > RAF-DB, AffectNet, FERPlus, plus authors' own Occlusion-RAF-DB / Pose-RAF-DB / Occlusion-FERPlus / Pose-FERPlus / Occlusion-AffectNet test sets
- **input_modality**: RGB image (full face plus a fixed set of cropped sub-regions extracted from the face image)
- **dynamic_fer_metrics**: N/A - static-image model
- **abaw_metrics**: N/A
- **robustness**:
  > Designed specifically for occlusion- and pose-robust FER. Reports strong gains on the proposed Occlusion-RAF-DB, Pose-RAF-DB (30/45 deg), Occlusion-FERPlus, Pose-FERPlus, and Occlusion-AffectNet benchmarks; the region-attention mechanism downweights occluded regions
- **explainability**:
  > Per-region attention weights provide interpretable visualization showing which face regions the model relies on; compatible with Grad-CAM-style analysis
- **deployment_target**: server-GPU (also feasible on edge given the small ResNet-18 backbone)

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- pretraining_corpus
- pretraining_strategy

---

### <a id="s2d-static-to-dynamic-21"></a>21. S2D (Static-to-Dynamic)

**Common**

- **name**: S2D (Static-to-Dynamic)
- **type**: model
- **subcategory**: dynamic-video
- **release_year**: 2024
- **paper_link**: https://arxiv.org/abs/2312.05447
- **publication_venue**: International Journal of Computer Vision (IJCV) 2024

**Model**

- **architecture_family**: hybrid (ViT-based static FER backbone + temporal modeling adapters)
- **backbone**:
  > POSTER++ / POSTER-V2 (IR-50 + ViT image-feature backbone) extended with Temporal Modeling Adapters (TMAs)
- **code_link**: https://github.com/MSA-LMC/S2D
- **pretrained_weights_available**: yes - checkpoints for DFEW / FERV39k / MAFW released on the GitHub repo
- **training_data_used**:
  > Training: DFEW, FERV39k, MAFW for dynamic FER. Static-base initialized from POSTER++ pretrained on AffectNet/RAF-DB.
- **pretraining_strategy**:
  > Adapts a strong supervised static FER model (POSTER++/POSTER-V2) to video by inserting temporal modeling adapters and fine-tuning on dynamic FER datasets; static backbone weights largely frozen or partially tuned.
- **pretraining_corpus**: Static FER pretraining via POSTER++: AffectNet / RAF-DB (and ImageNet/MS-Celeb-1M for backbone init)
- **input_modality**: video clip (RGB face frames)
- **robustness**:
  > Not explicitly evaluated on Occlusion-RAF-DB / Pose-RAF-DB; inherits robustness properties of POSTER++
- **explainability**: Attention visualizations from the underlying POSTER++ landmark-aware cross-attention can be applied
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- authors_or_creators
- cross_dataset_generalization
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- loss_function
- param_count

---

### <a id="scn-self-cure-network-22"></a>22. SCN (Self-Cure Network)

**Common**

- **name**: SCN (Self-Cure Network)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2020
- **authors_or_creators**:
  > Kai Wang, Xiaojiang Peng, Jianfei Yang, Shijian Lu, Yu Qiao (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences; SIAT-SenseTime Joint Lab)
- **paper_link**: https://arxiv.org/abs/2002.10392
- **publication_venue**: CVPR 2020

**Model**

- **architecture_family**: CNN
- **backbone**: ResNet-18 (ImageNet pretrained); also reported with IR-50 / VGG variants in some experiments
- **param_count**: ~11.2M (ResNet-18 backbone; SCN modules add a small FC importance head)
- **accuracy_affectnet**: 60.23% on AffectNet-8 (8-class, Overall/WAR with ResNet-18 backbone)
- **accuracy_rafdb**: 88.14% (Overall accuracy on RAF-DB basic 7-class with ResNet-18)
- **code_link**: https://github.com/kaiwang960112/Self-Cure-Network
- **pretrained_weights_available**:
  > partial — backbone pretrained weights (ResNet-18 on MS-Celeb-1M) and training code available via the official GitHub repository
- **loss_function**:
  > Composite: standard cross-entropy classification loss + Rank Regularization (RR) loss enforcing high-importance vs low-importance margin + Relabeling consistency (uses logits-based reassignment for low-importance samples)
- **training_data_used**:
  > Trained on RAF-DB, AffectNet, and FERPlus separately; also evaluated with synthetic label noise (10/20/30%) injected into RAF-DB and FERPlus
- **pretraining_strategy**:
  > face-recognition-pretrain (backbone pretrained on MS-Celeb-1M) followed by supervised fine-tuning with the SCN noise-suppression objective
- **pretraining_corpus**: MS-Celeb-1M for face-recognition pretraining of the backbone; ImageNet for some ablations
- **input_modality**: RGB image (aligned face crop, typically 224x224)
- **dynamic_fer_metrics**: N/A (static-image model)
- **abaw_metrics**: N/A (not evaluated on ABAW challenges in the original paper)
- **robustness**:
  > Specifically designed for label-noise robustness: validated on RAF-DB and FERPlus with 10%, 20%, 30% synthetic noise, showing clear gains over baseline (e.g., on FERPlus with 30% noise, SCN substantially outperforms vanilla CE-trained ResNet-18); also reports improvements on the WebEmotion noisy web-collected dataset
- **explainability**:
  > Self-Attention Importance weights provide per-sample confidence scores indicating which training samples are likely uncertain/mislabeled; can be visualized as a soft importance ranking
- **deployment_target**: server-GPU (also feasible on edge given the lightweight ResNet-18 backbone)

**Uncertain (skipped) fields**

- accuracy_fer2013
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency

---

### <a id="svfap-self-supervised-video-facial-affect-perceiver-23"></a>23. SVFAP (Self-supervised Video Facial Affect Perceiver)

**Common**

- **name**: SVFAP (Self-supervised Video Facial Affect Perceiver)
- **type**: model
- **subcategory**: dynamic-video
- **release_year**: 2024
- **authors_or_creators**:
  > Licai Sun, Zheng Lian, Bin Liu, Jianhua Tao (Institute of Automation, Chinese Academy of Sciences; Tsinghua University)
- **paper_link**: https://arxiv.org/abs/2401.00416
- **publication_venue**: IEEE Transactions on Affective Computing (TAFFC), 2024

**Model**

- **architecture_family**: self-supervised
- **backbone**: ViT (Video Transformer with temporal-aware spatial-temporal encoder)
- **code_link**: https://github.com/sunlicai/SVFAP
- **pretrained_weights_available**: yes - released on the official GitHub repository
- **loss_function**:
  > Self-supervised reconstruction loss (MSE on masked spatio-temporal tubes) for pretraining; cross-entropy for downstream emotion classification finetuning
- **training_data_used**:
  > Pretraining: VoxCeleb2 (face videos, label-free). Finetuning: DFEW, FERV39k, MAFW, Aff-Wild2, CREMA-D, RAVDESS, and additional dynamic FER datasets
- **pretraining_strategy**:
  > self-supervised (temporal-aware masked autoencoding with high masking ratio on spatio-temporal tubes)
- **pretraining_corpus**: VoxCeleb2 (large-scale face video corpus)
- **input_modality**: video clip (RGB face video)
- **explainability**: Attention map visualizations of spatio-temporal regions discussed in paper
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- abaw_metrics
- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- cross_dataset_generalization
- dynamic_fer_metrics
- hardware_requirements
- inference_fps_or_latency
- param_count
- robustness

---

### <a id="swinface-24"></a>24. SwinFace

**Common**

- **name**: SwinFace
- **type**: model
- **subcategory**: static-image
- **release_year**: 2023
- **authors_or_creators**:
  > Lixiong Qin, Mei Wang, Chao Deng, Ke Wang, Xi Chen, Jiani Hu, Weihong Deng (Beijing University of Posts and Telecommunications; Du Xiaoman Financial)
- **paper_link**: https://arxiv.org/abs/2308.11509
- **publication_venue**: IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2024 (preprint Aug 2023)

**Model**

- **architecture_family**: ViT
- **backbone**: Swin Transformer (Swin-S / Swin-B variants) shared across multi-task heads
- **code_link**: https://github.com/lxq1000/SwinFace
- **pretrained_weights_available**: yes (released on the official GitHub repository, including Swin-based multi-task checkpoints)
- **loss_function**:
  > Multi-task combination: cross-entropy for face recognition (with margin-based ArcFace-style loss), cross-entropy for expression and attribute classification, regression (MSE/L1) for age estimation, with a Multi-Level Channel Attention (MLCA) module to mitigate task conflicts
- **pretraining_strategy**:
  > supervised-pretrain (Swin Transformer backbone initialized from ImageNet supervised weights, then jointly fine-tuned in a multi-task face analysis framework)
- **pretraining_corpus**: ImageNet-1k (Swin backbone); MS-Celeb-1M used during multi-task face joint training
- **input_modality**: RGB image (aligned face crop)
- **dynamic_fer_metrics**: not applicable (static-image model)
- **abaw_metrics**: not reported
- **explainability**:
  > Attention maps from the Swin Transformer and the Multi-Level Channel Attention (MLCA) module provide some interpretability of feature sharing across tasks
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- accuracy_rafdb
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- robustness
- training_data_used

---

### <a id="transfer-25"></a>25. TransFER

**Common**

- **name**: TransFER
- **type**: model
- **subcategory**: static-image
- **release_year**: 2021
- **authors_or_creators**:
  > Fanglei Xue, Qiangchang Wang, Guodong Guo (Institute of Deep Learning, Baidu Research; National Engineering Laboratory for Deep Learning Technology and Application, Beijing, China)
- **paper_link**: https://arxiv.org/abs/2108.11116
- **publication_venue**: ICCV 2021 (IEEE/CVF International Conference on Computer Vision)

**Model**

- **architecture_family**: hybrid
- **backbone**: IR-50 (Improved ResNet-50) + Transformer encoder
- **accuracy_affectnet**: 66.23% Overall (WAR) on AffectNet-7 class setting
- **accuracy_rafdb**: 90.91% Overall accuracy on RAF-DB basic 7-class subset
- **pretrained_weights_available**: no
- **loss_function**:
  > Standard softmax cross-entropy classification loss combined with Multi-Attention Dropping (MAD) and Multi-head Self-Attention Dropping (MSAD) regularization to enforce diverse attention
- **training_data_used**: RAF-DB, AffectNet, FERPlus (fine-tuning); IR-50 backbone pretrained on MS-Celeb-1M
- **pretraining_strategy**: face-recognition-pretrain (IR-50 backbone pretrained for face recognition on MS-Celeb-1M)
- **pretraining_corpus**: MS-Celeb-1M
- **input_modality**: RGB image
- **dynamic_fer_metrics**: not applicable (static-image model)
- **abaw_metrics**: not reported
- **explainability**: Attention map visualization supported (multi-head self-attention maps from the Transformer encoder)
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_fer2013
- code_link
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- robustness

---

### <a id="vtff-visual-transformers-with-feature-fusion-26"></a>26. VTFF (Visual Transformers with Feature Fusion)

**Common**

- **name**: VTFF (Visual Transformers with Feature Fusion)
- **type**: model
- **subcategory**: static-image
- **release_year**: 2021
- **authors_or_creators**: Fuyan Ma, Bin Sun, Shutao Li (Hunan University, College of Electrical and Information Engineering)
- **paper_link**: https://arxiv.org/abs/2103.16854
- **publication_venue**: IEEE Transactions on Affective Computing (2023); preprint arXiv 2021

**Model**

- **architecture_family**: hybrid (CNN + ViT)
- **backbone**:
  > Two-branch CNN (ResNet-18 / IR-50 stem producing LANet attentional feature maps) fused via a Visual Transformer encoder
- **accuracy_rafdb**: 88.14% (Overall accuracy on RAF-DB basic 7-class)
- **loss_function**: Standard softmax cross-entropy classification loss over expression categories
- **training_data_used**:
  > Trained and evaluated separately on RAF-DB, FERPlus, and AffectNet; CNN backbone initialized from MS-Celeb-1M / face-recognition pretraining
- **pretraining_strategy**:
  > face-recognition-pretrain (CNN backbone pretrained on large-scale face recognition data); transformer fusion module trained from scratch with end-to-end fine-tuning
- **input_modality**: RGB image (single static face crop, typically 112x112 or 224x224)
- **dynamic_fer_metrics**: N/A (static-image model)
- **abaw_metrics**: N/A (not evaluated on ABAW challenges)
- **explainability**:
  > Attention map visualization from the visual transformer; LANet attentional feature maps highlight discriminative facial regions
- **deployment_target**: server-GPU

**Uncertain (skipped) fields**

- accuracy_affectnet
- accuracy_fer2013
- code_link
- cross_dataset_generalization
- hardware_requirements
- inference_fps_or_latency
- param_count
- pretrained_weights_available
- pretraining_corpus
- robustness

---

## Datasets

### <a id="afew-acted-facial-expressions-in-the-wild-27"></a>27. AFEW (Acted Facial Expressions in the Wild)

**Common**

- **name**: AFEW (Acted Facial Expressions in the Wild)
- **type**: dataset
- **subcategory**: dynamic-video
- **authors_or_creators**:
  > Abhinav Dhall, Roland Goecke, Simon Lucey, Tamas Gedeon — Australian National University (ANU), University of Canberra, CMU
- **paper_link**: https://doi.org/10.1109/MMUL.2012.26
- **publication_venue**: IEEE MultiMedia, 2012 (initial dataset paper at FG 2011 workshop / BeFIT)

**Dataset**

- **num_images**: AFEW 7.0: ~1809 video clips total (Train ~773 / Val ~383 / Test ~653)
- **num_classes**: 7
- **label_taxonomy**: basic-7 (Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral)
- **in_the_wild_or_lab**: in-the-wild (movie/TV clips)
- **license**: research-only (EULA / End User License Agreement required; non-commercial academic use)
- **download_url**: https://cs.anu.edu.au/few/AFEW.html (also distributed via the EmotiW Challenge organizers)
- **known_issues**:
  > Although called 'in-the-wild', clips are sourced from acted movie/TV scenes, so expressions are professional-actor portrayals rather than spontaneous behavior. Class imbalance is notable (Happy/Neutral/Angry over-represented; Disgust under-represented). Demographic bias toward Western/Hollywood content (predominantly Caucasian actors, English-language). Test-set labels are not publicly released — evaluation historically required submission to EmotiW organizers, which limits reproducibility outside the challenge. Some clips contain multiple faces or occlusions, requiring face tracking. Clip lengths and aspect ratios vary widely. Lip-reading / contextual cues (audio, body) can leak label information.
- **modality**: audio-visual (video clips with audio track; commonly used video-only)
- **annotation_method**: expert (semi-automatic recommender system 'SAFE' followed by human expert verification)
- **train_val_test_split**:
  > AFEW 7.0 official subject-independent split: Train 773 clips / Val 383 clips / Test 653 clips. Splits are subject- and movie-independent to prevent identity leakage. Used as the dynamic-video track of EmotiW 2013-2019.
- **demographic_metadata**:
  > Predominantly Western Hollywood actors; skewed toward Caucasian subjects with smaller representation of other ethnicities. Both genders represented but male-skewed. Wide age range from children to elderly. No formal demographic balancing.

**Uncertain (skipped) fields**

- num_subjects
- release_year
- resolution_or_fps
- top_sota_model

---

### <a id="aff-wild2-28"></a>28. Aff-Wild2

**Common**

- **name**: Aff-Wild2
- **type**: dataset
- **subcategory**: dynamic-video
- **release_year**: 2019
- **authors_or_creators**: Dimitrios Kollias and Stefanos Zafeiriou, Imperial College London
- **paper_link**: https://arxiv.org/abs/1811.07770

**Dataset**

- **num_subjects**: 458
- **num_classes**: 8 (expression task: basic-6 + neutral + other); also 12 action units; and continuous valence-arousal
- **label_taxonomy**:
  > mixed: (1) valence-arousal continuous in [-1,1], (2) 8 expression categories (6 basic emotions + neutral + other), (3) 12 facial action units (AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU15, AU23, AU24, AU25, AU26)
- **in_the_wild_or_lab**: in-the-wild
- **license**:
  > research-only; access via End User License Agreement (EULA) request to authors at Imperial College London
- **download_url**: https://ibug.doc.ic.ac.uk/resources/aff-wild2/
- **known_issues**:
  > Significant class imbalance across emotion categories (neutral and happy dominate; disgust, fear, anger are minority classes). Demographic imbalance with limited age, gender, and ethnicity diversity (sourced from YouTube). Annotation noise inherent to in-the-wild video frames; some frames have ambiguous or transitional expressions. Many frames have only partial labels (not all three tasks annotated for every frame) which complicates multi-task learning. Privacy and consent concerns are mitigated only by EULA-gated access. Track-level subject identity labels were refined across ABAW iterations, so older publications may report slightly different subject counts. Distribution bound by YouTube terms; videos can become unavailable over time.
- **modality**: video
- **annotation_method**:
  > expert (multiple trained annotators per frame using a custom annotation tool; valence-arousal annotated continuously by 4 experts, expressions and AUs by 3+ experts with majority/agreement protocol)
- **resolution_or_fps**: Variable resolution (commonly up to 1080p, with many 720p and 480p clips); typical frame rate 30 fps
- **demographic_metadata**:
  > Diverse YouTube sources covering varied ages, genders, and ethnicities, but not demographically balanced; skewed toward adult speakers and Western media content. No formal demographic statistics released by the authors.

**Uncertain (skipped) fields**

- num_images
- publication_venue
- top_sota_model
- train_val_test_split

---

### <a id="affectnet-29"></a>29. AffectNet

**Common**

- **name**: AffectNet
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2017
- **authors_or_creators**:
  > Ali Mollahosseini, Behzad Hasani, Mohammad H. Mahoor. University of Denver (Department of Electrical and Computer Engineering).
- **paper_link**: https://arxiv.org/abs/1708.03985
- **publication_venue**: IEEE Transactions on Affective Computing (TAFFC), 2017/2019

**Dataset**

- **num_classes**: 8
- **label_taxonomy**:
  > mixed: basic-8 categorical (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt) plus continuous valence-arousal (each in [-1, 1]); additional labels for None/Uncertain/Non-Face are provided to flag unusable samples
- **in_the_wild_or_lab**: in-the-wild
- **license**:
  > Research-only, requires signing an End User License Agreement (EULA) with the University of Denver; redistribution is prohibited. Commercial use is not permitted.
- **download_url**: http://mohammadmahoor.com/affectnet/ (request form / EULA must be submitted to the authors)
- **known_issues**:
  > AffectNet has several well-documented and critical issues. (1) Severe class imbalance: 'Happy' dominates (~134,000 annotated training images), while 'Disgust' (~3,800) and 'Contempt' (~3,750) are roughly 30-40x smaller; this skew strongly biases models trained without resampling/weighting and inflates Overall Accuracy (WAR) relative to Mean-class accuracy (UAR). The standard validation set is therefore deliberately balanced (500 images per class for the 8-class set, 4,000 total; 3,500 for the 7-class subset) to enable fair UAR reporting. (2) Western/internet bias: images were collected by querying three search engines (Google, Bing, Yahoo) in English, Spanish, Portuguese, German, Arabic and Farsi using 1,250 emotion-related keywords, but the resulting distribution skews toward Western, adult, lighter-skinned subjects; demographic balance across age, gender, and ethnicity is not enforced or documented. (3) Annotator subjectivity: only one of 12 trained annotators labeled each image, so inter-annotator agreement is limited; the authors report ~60% agreement on a doubly-annotated subset and ~65% human accuracy on categorical labels, indicating substantial label noise especially among Fear/Surprise and Anger/Disgust pairs. (4) Many crawled images are non-faces, cartoons, or duplicates and are flagged 'None/Uncertain/Non-Face'; cleaning is required. (5) The official test set has never been publicly released — researchers report results on the validation set, which can lead to inadvertent test-set tuning. (6) Ethical/consent concerns: images were scraped from the web without subject consent, raising GDPR and ethical questions; access has tightened over time and the EULA explicitly forbids redistribution. (7) Reported demographic audits (e.g., FairFace and follow-up fairness papers) show under-representation of Black, East Asian, and elderly subjects, leading to disparate accuracy across demographic groups.
- **modality**: image
- **annotation_method**:
  > expert (12 trained human annotators provided categorical, valence, and arousal labels; each image annotated by a single annotator with quality-control double-annotation on a subset)
- **train_val_test_split**:
  > Training set: ~287,651 images (8-class manual annotations); often cited overall ~414K-440K manually annotated. Validation set: 4,000 images for 8-class (500 per class, balanced) or 3,500 for 7-class. Official test set was withheld and never publicly released. Splits are random, not subject-independent (no subject IDs).
- **demographic_metadata**:
  > No formal age/gender/ethnicity labels distributed with the dataset. Empirical fairness audits report a Western-centric, adult-skewed distribution with under-representation of Black, East Asian, and elderly subjects; gender distribution is roughly balanced but not officially certified.

**Uncertain (skipped) fields**

- num_images
- num_subjects
- resolution_or_fps
- top_sota_model

---

### <a id="bp4d-bp4d-binghamton-pittsburgh-4d-spontaneous-expression-database-30"></a>30. BP4D / BP4D+ (Binghamton-Pittsburgh 4D Spontaneous Expression Database)

**Common**

- **name**: BP4D / BP4D+ (Binghamton-Pittsburgh 4D Spontaneous Expression Database)
- **type**: dataset
- **subcategory**: au
- **release_year**: BP4D: 2013-2014; BP4D+: 2016
- **authors_or_creators**:
  > Xing Zhang, Lijun Yin, Jeffrey F. Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, Peng Liu (Binghamton University and University of Pittsburgh)
- **paper_link**:
  > BP4D: https://doi.org/10.1016/j.imavis.2014.06.002 ; BP4D+: https://openaccess.thecvf.com/content_cvpr_2016_workshops/w27/papers/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.pdf
- **publication_venue**:
  > BP4D: Image and Vision Computing (IVC) 2014 (also FG 2013); BP4D+: CVPR Workshops 2016 (Multimodal Spontaneous Emotion Corpus, MMSE)

**Dataset**

- **num_images**:
  > BP4D: ~368,000 frames across 328 sequences (41 subjects x 8 tasks); approximately 140,000 frames have frame-level FACS AU annotations. BP4D+: ~1,400,000 frames across 1,400 sequences (140 subjects x 10 tasks).
- **num_subjects**: BP4D: 41 (23 female, 18 male); BP4D+: 140 (82 female, 58 male)
- **num_classes**:
  > 27 AUs annotated in BP4D (most studies use a 12-AU subset: AU1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24); BP4D also includes 8 emotion task labels
- **label_taxonomy**:
  > action-units (FACS AU occurrence and intensity for selected AUs); also task-elicited emotion categories (happy, sad, surprise, embarrassment, fear, physical pain, anger, disgust)
- **in_the_wild_or_lab**: lab-controlled
- **license**:
  > research-only (EULA required; distributed by Binghamton University via signed end-user license agreement, free for academic research)
- **download_url**: http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html (BP4D and BP4D+ request forms)
- **known_issues**:
  > Lab-controlled with limited demographic diversity (predominantly young adult subjects, mostly North American university population); BP4D ethnicity skews toward Euro-American/Asian. AU annotations are subset (only ~12 AUs commonly used) and frame-level coding is partial (~140K of 368K frames in BP4D). Class imbalance across AUs (some AUs very rare, e.g., AU2, AU23, AU24) leading to inflated F1 scores when using easy AUs only. Subject-independent splits are not officially fixed, so cross-paper comparisons can be inconsistent (typical practice: 3-fold subject-independent CV). Access requires institutional EULA, limiting reproducibility.
- **modality**:
  > multimodal (3D dynamic mesh + 2D RGB video + thermal video for BP4D; BP4D+ adds physiological signals: heart rate, blood pressure, skin conductance, respiration)
- **annotation_method**: expert (FACS-certified coders performed manual frame-level AU coding)
- **resolution_or_fps**:
  > 2D video: 1040x1392 px at 25 fps; 3D model sequences at 25 fps (~30K-50K vertices per mesh); thermal: 640x480
- **train_val_test_split**:
  > No official train/val/test split. Standard protocol is subject-independent 3-fold cross-validation on the 12-AU subset; some works use 5-fold CV. For BP4D+: often subject-independent 3-fold CV.
- **demographic_metadata**:
  > BP4D: 41 subjects, 23 female / 18 male, ages 18-29; ethnicity: 11 Asian, 6 African-American, 4 Hispanic, 20 Euro-American. BP4D+: 140 subjects, 82 female / 58 male, ages 18-66; ethnicity diversified to include Black, White, Asian, Hispanic/Latino. Both skew toward young adult North American university populations.
- **top_sota_model**:
  > ME-GraphAU (Luo et al., IJCAI 2022) reported average F1 ~64.1 on BP4D 12-AU. More recent: KS (Knowledge-Spread) and KDSRL approaches; AU-LLaVA and FAU-LLM-style methods report ~65-66 average F1. JAA-Net, LP-Net, SRERL, and HMP-PS are common baselines. BP4D average F1 SOTA hovers around 64-66 percent.

**Uncertain (skipped) fields**

- demographic_breakdown_BP4D_plus_exact_counts
- exact_total_frame_count_BP4D_plus
- top_sota_model_current_2026_value

---

### <a id="bu-3dfe-bu-4dfe-31"></a>31. BU-3DFE / BU-4DFE

**Common**

- **name**: BU-3DFE / BU-4DFE
- **type**: dataset
- **subcategory**: 3d / 4d (dynamic 3d video)
- **release_year**: BU-3DFE: 2006; BU-4DFE: 2008
- **authors_or_creators**:
  > Lijun Yin, Xiaozhou Wei, Yi Sun, Jun Wang, Matthew J. Rosato (BU-3DFE); Lijun Yin, Xiaochen Chen, Yi Sun, Tony Worm, Michael Reale (BU-4DFE). Department of Computer Science, State University of New York at Binghamton (Binghamton University).
- **paper_link**:
  > BU-3DFE: https://ieeexplore.ieee.org/document/1613022 (FG 2006); BU-4DFE: https://ieeexplore.ieee.org/document/4813324 (FG 2008)
- **publication_venue**: IEEE International Conference on Automatic Face and Gesture Recognition (FG) 2006 and 2008

**Dataset**

- **num_images**:
  > BU-3DFE: 2,500 static 3D face scans (100 subjects x 6 expressions x 4 intensity levels + neutral); BU-4DFE: ~606 dynamic 3D sequences yielding ~60,600 3D meshes (101 subjects x 6 expressions, ~100 frames per sequence at 25 fps for ~4 seconds).
- **num_subjects**: BU-3DFE: 100 subjects (56 female, 44 male); BU-4DFE: 101 subjects (58 female, 43 male)
- **num_classes**: 7
- **label_taxonomy**:
  > basic-7 (Ekman's six basic emotions: anger, disgust, fear, happiness, sadness, surprise — plus neutral). BU-3DFE additionally encodes 4 intensity levels (low, middle, high, highest) per expression.
- **in_the_wild_or_lab**: lab-controlled
- **license**:
  > research-only EULA (academic / non-commercial use). Requires signed license agreement with Binghamton University.
- **download_url**:
  > https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html (request form via Binghamton University CS department)
- **known_issues**:
  > Lab-controlled acquisition with posed (non-spontaneous) expressions limits generalization to in-the-wild affect. BU-4DFE sequences begin and end at neutral with the apex in the middle, so onset/apex/offset must be inferred. Subjects were instructed to perform expressions, raising concerns about ecological validity vs. spontaneous emotion. The 3D scan resolution and mesh density vary across subjects; some occlusions and missing data exist around hair, ears, and mouth interiors. Demographic distribution is skewed toward young adults (largely undergraduates) and although ethnically diverse for its era, the absolute counts per group are small. Distribution under EULA prohibits redistribution, complicating reproducibility. No intensity annotation in BU-4DFE.
- **modality**:
  > 3d (BU-3DFE: static 3D meshes + paired frontal/side 2D textures); 4d (BU-4DFE: temporal sequences of 3D meshes with synchronized 2D video texture)
- **annotation_method**:
  > expert / self-report (acted expressions performed by subjects under guidance; expression categories assigned by acquisition protocol; intensity ratings validated by Binghamton researchers)
- **resolution_or_fps**:
  > BU-3DFE: ~1,300 polygons per face mesh range, paired texture images ~1,040x1,329 px (varies); BU-4DFE: 3D meshes captured at 25 fps with ~35,000 vertices per frame; texture video at 1,040x1,329 px, 25 fps.
- **train_val_test_split**:
  > No official split. Common protocol: 10-fold subject-independent cross-validation on the 100 (BU-3DFE) or 101 (BU-4DFE) subjects, often using the two highest intensity levels for BU-3DFE.
- **demographic_metadata**:
  > BU-3DFE: 100 subjects ages 18-70, 56% female / 44% male; ethnicities include White (51), Black (8), East-Asian (24), Middle-Eastern (3), Hispanic-Latino (3), Indian (11). BU-4DFE: 101 subjects ages 18-45, 58 female / 43 male; mixed ethnicities (Asian, Black, Hispanic-Latino, White). Skew toward young adults; small absolute counts in non-White groups limit demographic-fair evaluation.

**Uncertain (skipped) fields**

- BU-4DFE total frame count (commonly cited as ~60K but exact figure depends on per-sequence frame count)
- exact resolution_or_fps figures (mesh vertex counts and texture resolutions vary across releases)
- top_sota_model

---

### <a id="c-expr-db-compound-expression-database-32"></a>32. C-EXPR-DB (Compound Expression Database)

**Common**

- **name**: C-EXPR-DB (Compound Expression Database)
- **type**: dataset
- **subcategory**: dynamic-video
- **authors_or_creators**: Dimitrios Kollias et al. — Queen Mary University of London / Imperial College London (iBUG group)
- **publication_venue**: CVPR Workshops (ABAW6 Challenge, 2024); related work in IJCV 2023

**Dataset**

- **num_images**: ~400 videos / ~200,000 annotated frames
- **num_classes**: 7
- **label_taxonomy**:
  > compound (7 compound expressions: Fearfully Surprised, Happily Surprised, Sadly Surprised, Disgustedly Surprised, Angrily Surprised, Sadly Fearful, Sadly Angry)
- **in_the_wild_or_lab**: in-the-wild (YouTube / web video sources)
- **license**:
  > research-only (EULA / End User License Agreement; non-commercial academic use via iBUG resources page)
- **download_url**:
  > https://ibug.doc.ic.ac.uk/resources/abaw/ (request form / EULA required); access also granted to ABAW6 Challenge participants
- **known_issues**:
  > Compound expression labels are inherently ambiguous and harder to annotate consistently than basic emotions; inter-annotator agreement on compound categories is lower than on basic-7. Class distribution is imbalanced — 'Happily Surprised' and 'Fearfully Surprised' are far more common than rarer combinations like 'Disgustedly Surprised' or 'Sadly Angry'. As an in-the-wild YouTube collection, demographic balance is not formally controlled and skews toward online-content creators. The benchmark was released specifically for the ABAW6 zero-shot / few-shot Compound Expression Recognition track, so test labels are withheld from public release. Limited public baselines and evaluation history relative to mature benchmarks. Some videos contain multiple faces, head pose variation, and occlusions.
- **modality**: video (audio-visual track available; primarily evaluated as visual-only)
- **annotation_method**:
  > expert (trained annotators following the Du, Tao & Martinez compound expression taxonomy); per-frame compound expression labels
- **train_val_test_split**:
  > ABAW6 (2024) Compound Expression Recognition track used C-EXPR-DB as a zero-shot / cross-corpus test benchmark — participants train on Aff-Wild2 (basic expressions) and are evaluated on C-EXPR-DB compound labels. No standard internal train/val/test split was released publicly; full 400 videos / 200K frames serve as the evaluation set in the challenge protocol.

**Uncertain (skipped) fields**

- demographic_metadata
- num_subjects
- paper_link
- release_year
- resolution_or_fps
- top_sota_model

---

### <a id="caer-s-context-aware-emotion-recognition---static-33"></a>33. CAER-S (Context-Aware Emotion Recognition - Static)

**Common**

- **name**: CAER-S (Context-Aware Emotion Recognition - Static)
- **type**: dataset
- **subcategory**: static-image (context-aware FER; derived from CAER dynamic-video dataset)
- **release_year**: 2019
- **authors_or_creators**: Jiyoung Lee, Seungryong Kim, Sunok Kim, Jungin Park, Kwanghoon Sohn. Yonsei University.
- **paper_link**: https://arxiv.org/abs/1908.05913
- **publication_venue**: ICCV 2019

**Dataset**

- **num_images**: 70000
- **num_classes**: 7
- **label_taxonomy**: basic-7 (anger, disgust, fear, happiness, sadness, surprise, neutral)
- **in_the_wild_or_lab**: in-the-wild (frames sampled from 79 TV shows)
- **license**: research-only (academic use; available via the project page) [uncertain on exact license text]
- **download_url**: https://caer-dataset.github.io/
- **known_issues**:
  > Labels are derived from clip-level emotion tags (originally based on TV-show context and crowd verification), so static frames may not always show the peak expression at the moment captured, leading to label noise. Source material (TV shows) introduces selection bias toward dramatized / acted emotion and Western/Korean media demographics. Class distribution is imbalanced (neutral and happy dominate). Some images contain multiple people, but only one face per image is the labeled subject, which can confuse context-aware models. Privacy/consent concerns are typical of TV-show-derived datasets. The number of frames is sometimes reported differently across papers (~70k total in the static set; the ~13,201 figure cited in the user note corresponds to the test split or a specific subset rather than the full CAER-S).
- **modality**: image
- **annotation_method**:
  > crowdsourced (Amazon Mechanical Turk verification of emotion labels on the parent CAER video clips, propagated to extracted static frames)
- **train_val_test_split**:
  > Official split into train / validation / test, approximately 70% / 10% / 20% (subject-independent at the show/clip level). Reported counts: ~44,996 train, ~7,131 val, ~13,201 test (test count matches the ~13,201 figure in the user note). [uncertain on exact counts]
- **demographic_metadata**:
  > Sourced from 79 TV shows (mostly North American and some Korean productions per the parent CAER paper); skews toward adult actors, with limited explicit annotation of age / gender / ethnicity. Western/East-Asian media bias.
- **top_sota_model**:
  > GLAMOR-Net and follow-ups (e.g., context-aware transformer methods) report ~77-89% accuracy on CAER-S; CAER-Net (the original baseline) reported ~73.5% accuracy. [uncertain on current SOTA - leaderboard not actively maintained]

**Uncertain (skipped) fields**

- license (exact terms)
- num_subjects
- resolution_or_fps
- top_sota_model (current best)
- train_val_test_split (exact counts)

---

### <a id="casme3-34"></a>34. CAS(ME)^3

**Common**

- **name**: CAS(ME)^3
- **type**: dataset
- **subcategory**: micro-expression
- **release_year**: 2022
- **authors_or_creators**:
  > Jingting Li, Zizhao Dong, Shaoyuan Lu, Su-Jing Wang, Wen-Jing Yan, Yinhuan Ma, Yong Liu, Changbing Huang, Xiaolan Fu — Institute of Psychology, Chinese Academy of Sciences (CAS)
- **paper_link**: https://doi.org/10.1109/TPAMI.2022.3174895
- **publication_venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023

**Dataset**

- **num_classes**: 7
- **label_taxonomy**:
  > mixed: 7 emotion categories (happiness, disgust, fear, anger, sadness, surprise, others) plus Action Units (AUs), self-report emotion labels, and onset/apex/offset frame annotations
- **in_the_wild_or_lab**: lab-controlled
- **license**: research-only (EULA / agreement required from Institute of Psychology, CAS)
- **download_url**: https://psych.ac.cn/pyu_lab/services/casme3.html
- **known_issues**:
  > Strong class imbalance across emotion categories (very few fear/sadness samples). Subjects are predominantly Chinese (Han ethnicity), which limits cross-cultural/ethnic generalization despite the cross-cultural design intent. Mock-crime / deception scenarios in Part C raise ethical considerations regarding deception research; access requires signed agreement. Annotation of micro-expression onset/apex/offset frames is inherently noisy and inter-rater agreement is moderate. Depth and physiological signals in Part B are not synchronized perfectly with all video frames in some sessions.
- **modality**:
  > multimodal: video (RGB) + depth (Part B) + physiological/collateral cues (Part B); deception scenario video (Part C)
- **annotation_method**:
  > expert (trained FACS coders annotated AUs and onset/apex/offset; emotion labels combined self-report and expert coding)
- **resolution_or_fps**: 1280x720 RGB at 30 fps (Part A primary); high-speed and depth streams in Part B
- **demographic_metadata**:
  > Subjects are mainly Chinese (East Asian) university students/young adults; gender roughly balanced; narrow age range (approximately 20-30 years); limited ethnic diversity

**Uncertain (skipped) fields**

- num_images
- num_subjects
- top_sota_model
- train_val_test_split

---

### <a id="ck-extended-cohn-kanade-dataset-35"></a>35. CK+ (Extended Cohn-Kanade Dataset)

**Common**

- **name**: CK+ (Extended Cohn-Kanade Dataset)
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2010
- **authors_or_creators**:
  > Patrick Lucey, Jeffrey F. Cohn, Takeo Kanade, Jason Saragih, Zara Ambadar, Iain Matthews. Affiliations: Robotics Institute, Carnegie Mellon University; University of Pittsburgh; Commonwealth Scientific and Industrial Research Organisation (CSIRO).
- **paper_link**: https://ieeexplore.ieee.org/document/5543262
- **publication_venue**:
  > CVPR Workshops (CVPR-W) 2010 - IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops

**Dataset**

- **num_images**:
  > 593 video sequences (123 subjects). 327 sequences are emotion-labeled; commonly extracted as ~981 still images (typically 3 frames per sequence: neutral + onset + peak/apex) or just 327 apex frames depending on the protocol.
- **num_subjects**: 123
- **num_classes**:
  > 7 (basic-7: anger, contempt, disgust, fear, happiness, sadness, surprise) when including contempt; sometimes used as 8 classes when neutral is added
- **label_taxonomy**:
  > mixed: basic-7 emotion labels (with contempt instead of the more common Ekman basic-6 + neutral) on 327 sequences; full FACS Action Unit (AU) coding on the apex frame of all 593 sequences
- **in_the_wild_or_lab**: lab-controlled
- **license**: research-only (EULA / signed end-user license agreement required for academic, non-commercial use)
- **download_url**:
  > https://www.jeffcohn.net/Resources/ (Cohn-Kanade page) — also distributed via http://www.consortium.ri.cmu.edu/ckagree/ historically; access requires a signed release form
- **known_issues**:
  > Posed (acted) expressions rather than spontaneous, limiting ecological validity. Small size (only 327 emotion-labeled sequences) is prone to overfitting; many published >99% accuracies use random splits with subject leakage and are not directly comparable to leave-one-subject-out (LOSO) results. Strong demographic imbalance: predominantly North American university student subjects, ~81% Euro-American, ~13% Afro-American, ~6% other; ~69% female. Frontal pose, controlled lighting, and uniform background do not reflect in-the-wild conditions, so models trained only on CK+ generalize poorly. Contempt class is very small (~18 sequences), creating severe class imbalance.
- **modality**: video (image sequences from neutral to peak expression)
- **annotation_method**:
  > expert (FACS-certified coders performed AU annotation; emotion labels validated against FACS-based emotion prediction tables)
- **resolution_or_fps**: 640x490 or 640x480 pixels, 8-bit grayscale or 24-bit color; frame rate approximately 30 fps
- **train_val_test_split**:
  > No official train/val/test split. Two community-standard protocols: (1) leave-one-subject-out (LOSO) cross-validation (subject-independent, recommended); (2) k-fold cross-validation (often 10-fold) on extracted apex frames (frequently NOT subject-independent in older work). Many papers use last 1-3 frames per sequence as samples.
- **demographic_metadata**:
  > 210 adult subjects originally recruited (123 included in CK+), ages 18-50; approximately 69% female, 31% male; 81% Euro-American, 13% Afro-American, 6% other groups (Asian/Latino). Western, university-population bias; limited age range (predominantly young adults).

**Uncertain (skipped) fields**

- top_sota_model

---

### <a id="dfew-dynamic-facial-expression-in-the-wild-36"></a>36. DFEW (Dynamic Facial Expression in the Wild)

**Common**

- **name**: DFEW (Dynamic Facial Expression in the Wild)
- **type**: dataset
- **subcategory**: dynamic-video
- **release_year**: 2020
- **authors_or_creators**:
  > Xingxun Jiang, Yuan Zong, Wenming Zheng, Chuangao Tang, Wanchuang Xia, Cheng Lu, Jiateng Liu. Anhui University and Southeast University (Key Laboratory of Child Development and Learning Science).
- **paper_link**: https://arxiv.org/abs/2008.05924
- **publication_venue**: ACM Multimedia (ACM MM) 2020

**Dataset**

- **num_images**: 16,372 video clips selected from over 1,500 movies
- **num_classes**: 7
- **label_taxonomy**:
  > basic-7 (happy, sad, neutral, angry, surprise, disgust, fear); each clip has a single-label annotation and a 7-dimensional probability/distribution annotation aggregated from ~10-12 annotators
- **in_the_wild_or_lab**: in-the-wild
- **license**: research-only; available via signed End User License Agreement (EULA) submitted to the authors
- **download_url**: https://dfew-dataset.github.io
- **known_issues**:
  > Significant class imbalance: happy, sad and neutral classes dominate while disgust and fear are heavily underrepresented (disgust has the fewest clips, often <500). Annotation ambiguity in movie clips with mixed or transitional emotions; the 7-dimensional probability label partially mitigates this. Demographic bias toward actors in (largely Chinese and Hollywood) commercial cinema, with limited child and elderly representation. Subject identity is not labeled, so the official 5-fold split is movie-independent rather than strictly subject-independent, which can leak similar visual contexts across folds. Some clips contain motion blur, occlusion, low light, and multiple faces, which is intentional for the in-the-wild setting but increases label noise. Distribution restricted by movie copyright; only cropped face regions / clip indices are shared.
- **modality**: video
- **annotation_method**:
  > crowdsourced/expert hybrid: each clip independently annotated by ~10-12 trained annotators under a double-blind professional protocol; final single label by majority vote and 7-d probability label by normalized vote distribution
- **train_val_test_split**:
  > Standard 5-fold subject/movie-independent cross-validation provided by the authors; metrics reported are unweighted average recall (UAR) and weighted average recall (WAR) averaged across the 5 folds. No fixed train/val/test split outside of the 5-fold CV.
- **demographic_metadata**:
  > Drawn from 1,500+ movies including a substantial proportion of Chinese-language films alongside English-language productions, providing better East Asian representation than many Western-centric benchmarks; however, no formal age/gender/ethnicity statistics are released and child/elderly classes are sparse.

**Uncertain (skipped) fields**

- num_subjects
- resolution_or_fps
- top_sota_model

---

### <a id="disfa-disfa-denver-intensity-of-spontaneous-facial-action-database-37"></a>37. DISFA / DISFA+ (Denver Intensity of Spontaneous Facial Action Database)

**Common**

- **name**: DISFA / DISFA+ (Denver Intensity of Spontaneous Facial Action Database)
- **type**: dataset
- **subcategory**: au
- **release_year**: DISFA: 2013; DISFA+: 2016
- **authors_or_creators**:
  > S. Mohammad Mavadati, Mohammad H. Mahoor, Kevin Bartlett, Philip Trinh, Jeffrey F. Cohn (University of Denver, University of Pittsburgh, Carnegie Mellon University)
- **paper_link**:
  > DISFA: https://doi.org/10.1109/T-AFFC.2013.4 ; DISFA+: https://ieeexplore.ieee.org/document/7477605 (FG 2016 Workshop)
- **publication_venue**: DISFA: IEEE Transactions on Affective Computing (TAFFC) 2013; DISFA+: IEEE FG 2016 / FG Workshops

**Dataset**

- **num_images**:
  > DISFA: ~130,000 frames (27 videos, ~4,845 frames per video at 4 min 48 s, 20 fps). DISFA+: additional ~57,000 frames of posed expressions over 9 subjects.
- **num_subjects**:
  > DISFA: 27 (12 female, 15 male); DISFA+: 9 subjects (overlapping subset re-recorded with posed expressions)
- **num_classes**:
  > 12 AUs with intensity labels: AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26 (each on 0-5 ordinal intensity scale)
- **label_taxonomy**: action-units (AU occurrence and 6-level intensity 0-5, FACS-coded)
- **in_the_wild_or_lab**: lab-controlled
- **license**:
  > research-only (EULA via University of Denver / Mohammad H. Mahoor lab; free for non-commercial academic research)
- **download_url**:
  > http://mohammadmahoor.com/disfa/ (request form to Mahoor lab); DISFA+: http://mohammadmahoor.com/disfa-plus-request-form/
- **known_issues**:
  > Small subject pool (only 27 subjects) limits generalization and makes subject-independent evaluation high-variance. Severe class imbalance: many AU intensity labels are dominated by 0 (neutral) and rare for higher intensities (3-5), inflating F1 if not stratified. Limited demographic diversity (mostly young adults of mixed ethnicity from Denver area). Stimulus is a single 4-min YouTube video clip per subject, so contextual diversity is low. Standard subject-independent 3-fold CV protocol is informal (no official split file), causing cross-paper inconsistency. DISFA+ has only 9 subjects and posed expressions, so it should not be used as primary intensity benchmark alone.
- **modality**: video (RGB stereo); single high-resolution camera commonly used
- **annotation_method**:
  > expert (two FACS-certified coders performed frame-by-frame intensity coding; high inter-coder reliability reported, ICC > 0.8 for most AUs)
- **resolution_or_fps**:
  > 1024x768 px at 20 fps (some references cite 1024x768 or 640x480 depending on stream); ~4 min 48 s per subject
- **train_val_test_split**:
  > No official split. Standard protocol: subject-independent 3-fold cross-validation across the 27 subjects, training on 18 and testing on 9 per fold. Reported metrics: F1-frame and ICC (intraclass correlation) for intensity estimation.
- **demographic_metadata**:
  > DISFA: 27 adults (12 female, 15 male), ages ~18-50; ethnicity self-reported: ~14 Euro-American, 4 Asian, 4 Hispanic, 4 African-American, 1 other (approximate). Skews toward young adult North American university population.
- **top_sota_model**:
  > ME-GraphAU (IJCAI 2022) reports average F1 ~63.1 on DISFA; more recent transformer/LLM-augmented methods (e.g., KS, AU-LLaVA, FAN-Trans, MDHR) report 64-67 average F1. For intensity estimation, ICC SOTA ~0.65-0.70 mean across 12 AUs (e.g., HMP-PS, CCNN-IT, BG-CNN).

**Uncertain (skipped) fields**

- DISFA_demographic_exact_breakdown
- DISFA_plus_total_frame_count_exact
- DISFA_video_resolution_canonical
- top_sota_model_current_2026_value

---

### <a id="emoset-38"></a>38. EmoSet

**Common**

- **name**: EmoSet
- **type**: dataset
- **subcategory**: affective-image (visual emotion analysis, broader than face)
- **release_year**: 2023
- **authors_or_creators**:
  > Jingyuan Yang, Qirui Huang, Tingting Ding, Dani Lischinski, Daniel Cohen-Or, Hui Huang. Visual Computing Research Center (VCC), Shenzhen University; with collaborators from The Hebrew University of Jerusalem and Tel Aviv University.
- **paper_link**: https://arxiv.org/abs/2308.00890
- **publication_venue**: ICCV 2023 (IEEE/CVF International Conference on Computer Vision)

**Dataset**

- **num_images**:
  > 3.3M images in total (EmoSet-3.3M); 118,102 images carefully labeled with rich attributes (EmoSet-118K). The paper describes ~1.2M weakly labeled subset variations across releases.
- **num_subjects**: N/A (web-collected affective images, not identity-centric; no subject identity annotation)
- **num_classes**: 8
- **label_taxonomy**:
  > basic-8 — Mikels' wheel of emotions: amusement, awe, contentment, excitement (positive); anger, disgust, fear, sadness (negative). Plus 6 additional attribute annotations: brightness, colorfulness, scene type, object class, facial expression, human action.
- **in_the_wild_or_lab**:
  > in-the-wild (collected from social media platforms and stock-image search engines including Flickr, Instagram, Pinterest, Unsplash, Pixabay)
- **download_url**: https://vcc.tech/EmoSet (project page); code and labels: https://github.com/JingyuanYY/EmoSet
- **known_issues**:
  > Class imbalance across the 8 Mikels categories (e.g., 'contentment' and 'amusement' tend to be overrepresented vs. 'anger' or 'disgust'). Labels combine crowdsourced annotation with automatically inferred attributes — emotion labels can be subjective and culturally dependent (Mikels' taxonomy is Western-centric). Web-scraped sources introduce demographic and content biases (skew toward Western, English-speaking platforms). Some images contain multiple emotional cues, making single-label classification noisy. Copyright/redistribution concerns: while research use is intended, images sourced from third-party platforms may have evolving licensing terms. The 6 attribute annotations are partly automatic (pseudo-labeled by detectors), introducing label noise. No standardized identity / face protection beyond what original platforms provide.
- **modality**: image
- **annotation_method**:
  > crowdsourced (emotion labels by human annotators on the 118K rich-labeled subset, with multi-annotator agreement filtering) + automatic-pseudo-labeled (attributes such as scene/object/facial-expression/human-action obtained via off-the-shelf detectors and classifiers)
- **demographic_metadata**:
  > No explicit demographic metadata for depicted humans. Images cover diverse scenes, objects, and human actions globally, but the source platforms (Flickr, Pinterest, Unsplash, Pixabay, Instagram) skew Western / English-speaking. Mikels' emotion taxonomy itself is grounded in Western affective psychology, which may not transfer cleanly across cultures. No age/gender/ethnicity balance audit reported.

**Uncertain (skipped) fields**

- license (exact terms not always specified; inherits original image licenses)
- num_images breakdown between weakly- and strongly-labeled subsets across releases
- resolution_or_fps (heterogeneous web sources)
- top_sota_model (rapidly evolving since ICCV 2023 release)
- train_val_test_split exact counts

---

### <a id="emotic-emotion-recognition-in-context-39"></a>39. EMOTIC (Emotion Recognition in Context)

**Common**

- **name**: EMOTIC (Emotion Recognition in Context)
- **type**: dataset
- **subcategory**: context-emotion (static-image, person-in-scene)
- **release_year**: 2017
- **authors_or_creators**:
  > Ronak Kosti, Jose M. Alvarez, Adria Recasens, Agata Lapedriza — Universitat Oberta de Catalunya (UOC) and MIT CSAIL
- **paper_link**: https://arxiv.org/abs/2003.13401
- **publication_venue**:
  > CVPR 2017 (initial); IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2019 (extended)

**Dataset**

- **num_images**: 23,571 images containing 34,320 annotated person instances
- **num_subjects**:
  > Not identity-controlled; ~34,320 annotated person bounding boxes (subjects are not unique identities — images sourced from web/COCO/Ade20K)
- **num_classes**: 26
- **label_taxonomy**:
  > mixed: 26 fine-grained discrete emotion categories (multi-label) + 3-D continuous Valence-Arousal-Dominance (VAD)
- **in_the_wild_or_lab**: in-the-wild
- **license**: research-only (custom license; non-commercial academic use)
- **download_url**: http://sunai.uoc.edu/emotic/ (also mirrored via http://emotic.csail.mit.edu/)
- **known_issues**:
  > Multi-label discrete annotations are subjective and inter-annotator agreement is moderate (each image annotated by ~3 annotators via AMT). Class imbalance is severe — categories such as 'Engagement', 'Happiness', 'Excitement' dominate, while 'Suffering', 'Embarrassment', 'Aversion' are underrepresented. Demographic bias inherited from MS-COCO and Ade20K (Western-centric scenes, skewed age/ethnicity distribution). Bounding boxes around persons can be loose; small/occluded faces are common, making face-only methods underperform. Continuous VAD labels are scaled 1-10 with annotator variance.
- **modality**: image (RGB, person bounding box + full scene context)
- **annotation_method**: crowdsourced (Amazon Mechanical Turk; multiple annotators per instance, aggregated)
- **resolution_or_fps**:
  > Variable resolution (inherited from MS-COCO, Ade20K, and Google-collected images); typical sizes range from ~300x300 up to HD
- **demographic_metadata**:
  > No explicit demographic metadata; person instances span varied ages, genders, and ethnicities but distribution mirrors MS-COCO/Ade20K (Western bias, adult-skewed). No identity or fine-grained demographic labels are released.

**Uncertain (skipped) fields**

- top_sota_model
- train_val_test_split

---

### <a id="emotionet-40"></a>40. EmotioNet

**Common**

- **name**: EmotioNet
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2016
- **authors_or_creators**: C. Fabian Benitez-Quiroz, Ramprakash Srinivasan, Aleix M. Martinez (The Ohio State University)
- **paper_link**:
  > https://openaccess.thecvf.com/content_cvpr_2016/papers/Benitez-Quiroz_EmotioNet_An_Accurate_CVPR_2016_paper.pdf
- **publication_venue**: CVPR 2016

**Dataset**

- **num_images**: approximately 1,000,000 (1M) facial images; ~25,000 manually annotated/validated
- **num_classes**: 23 compound emotion categories (basic + compound) plus 12 Action Units
- **label_taxonomy**:
  > mixed: action-units (12 AUs: 1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43) and compound emotions (23 categories following Du et al. 2014 taxonomy including 6 basic + neutral + 17 compound emotions)
- **in_the_wild_or_lab**: in-the-wild (web-crawled images)
- **download_url**:
  > http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/ (originally); also referenced via http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html
- **known_issues**:
  > Most labels (~975K of 1M) are auto-generated by an algorithm (AU detector trained on smaller manually annotated data) rather than manually verified, which introduces significant label noise. Only ~25K images are manually validated. Web-crawl introduces demographic biases toward Western/celebrity faces. Many original image URLs have become dead links over time, making full reproduction difficult. The dataset was used for the EmotioNet Challenge (2017-2018) which is no longer active. Access has historically required institutional email and an EULA.
- **modality**: image
- **annotation_method**:
  > mixed: automatic-pseudo-labeled (algorithmic AU detection) for ~975K images and expert/manual annotation for ~25K validation subset
- **demographic_metadata**:
  > No explicit demographic balancing; web-crawled internet images skew toward Western/Caucasian and celebrity faces. No official age/gender/ethnicity statistics are published with the release.

**Uncertain (skipped) fields**

- license
- num_subjects
- resolution_or_fps
- top_sota_model
- train_val_test_split

---

### <a id="expw-expression-in-the-wild-41"></a>41. ExpW (Expression in-the-Wild)

**Common**

- **name**: ExpW (Expression in-the-Wild)
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2018
- **authors_or_creators**:
  > Zhanpeng Zhang, Ping Luo, Chen Change Loy, Xiaoou Tang (Multimedia Laboratory, The Chinese University of Hong Kong - CUHK)
- **paper_link**:
  > https://link.springer.com/article/10.1007/s11263-017-1055-1 (DOI: 10.1007/s11263-017-1055-1); arXiv:1511.03447 (earlier version 'Learning Social Relation Traits from Face Images')
- **publication_venue**:
  > International Journal of Computer Vision (IJCV) 2018; earlier ICCV 2015 (related work on social relation traits)

**Dataset**

- **num_images**: 91,793 face images
- **num_classes**: 7
- **label_taxonomy**: basic-7 (angry, disgust, fear, happy, sad, surprise, neutral)
- **in_the_wild_or_lab**: in-the-wild (web-crawled images)
- **download_url**: https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html (CUHK MMLAB project page)
- **known_issues**:
  > Labels were assigned via crowdsourced majority vote among annotators, which introduces label noise especially for ambiguous expressions (fear/disgust often confused). Class imbalance is significant: 'happy' and 'neutral' classes dominate while 'disgust' and 'fear' are heavily underrepresented. Web-crawl source introduces demographic biases. No official train/val/test split is provided, so cross-paper comparisons can be inconsistent. Some images contain multiple faces; the released bounding boxes are needed to extract individual face crops. Released alongside social relation work, so primary curation focus was social context not pure expression balance.
- **modality**: image
- **annotation_method**: crowdsourced (majority vote among multiple annotators)
- **demographic_metadata**:
  > No official demographic metadata provided; web-crawled images contain a mix of ages, genders, and ethnicities but with no documented balancing. Often used for crowd-emotion analysis where multiple faces per image are common.

**Uncertain (skipped) fields**

- license
- num_subjects
- resolution_or_fps
- top_sota_model
- train_val_test_split

---

### <a id="fer-2013-42"></a>42. FER-2013

**Common**

- **name**: FER-2013
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2013
- **authors_or_creators**:
  > Ian J. Goodfellow, Dumitru Erhan, Pierre Luc Carrier, Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski, Yichuan Tang, David Thaler, Dong-Hyun Lee, Yingbo Zhou, Chetan Ramaiah, Fangxiang Feng, Ruifan Li, Xiaojie Wang, Dimitris Athanasakis, John Shawe-Taylor, Maxim Milakov, John Park, Radu Ionescu, Marius Popescu, Cristian Grozea, James Bergstra, Jingjing Xie, Lukasz Romaszko, Bing Xu, Zhang Chuang, Yoshua Bengio. Primary affiliation: Université de Montréal (LISA Lab); created for the ICML 2013 Representation Learning Workshop challenge organized via Kaggle.
- **paper_link**: https://arxiv.org/abs/1307.0414
- **publication_venue**: ICML 2013 Workshop on Challenges in Representation Learning

**Dataset**

- **num_images**: 35887
- **num_classes**: 7
- **label_taxonomy**: basic-7 (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
- **in_the_wild_or_lab**: in-the-wild
- **download_url**:
  > https://www.kaggle.com/datasets/msambare/fer2013 (also https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- **known_issues**:
  > FER-2013 has well-documented label noise: independent re-annotation studies estimate human accuracy on the test set at roughly 65-68%, which serves as an empirical upper bound and indicates a substantial fraction of mislabeled or ambiguous examples. The 'disgust' class is severely underrepresented (~547 training images vs ~7,000+ for happy), causing class imbalance. Faces were collected via Google image search using emotion keywords, so labels reflect query terms rather than verified expressions; many images contain occlusions, watermarks, cartoons/non-faces, low resolution (48x48 grayscale), or extreme poses. The auto-aligned crops are often poorly centered. Demographic distribution is biased toward Western, adult, lighter-skinned subjects with no documented age/gender/ethnicity metadata. Microsoft's FER+ relabeling project (Barsoum et al., 2016) provided crowdsourced multi-label re-annotations and showed the original labels disagree with majority crowd votes on ~25-30% of samples; FER+ is now commonly used as a cleaner alternative. No subject identity information, so subject-independent evaluation is impossible.
- **modality**: image
- **annotation_method**:
  > crowdsourced (single-label keyword-based labels collected via Google image-search queries; later relabeled by crowd consensus in FER+)
- **resolution_or_fps**: 48x48 grayscale
- **train_val_test_split**:
  > Standard split: 28,709 Training / 3,589 PublicTest (validation) / 3,589 PrivateTest (final test). Splits are random (not subject-independent — no subject IDs are available).
- **demographic_metadata**:
  > No official demographic labels released. Empirical analyses report skew toward Western, adult, lighter-skinned faces with under-representation of children, elderly, and darker-skinned subjects; gender distribution is roughly mixed but not formally documented.

**Uncertain (skipped) fields**

- license
- num_subjects
- top_sota_model

---

### <a id="ferv39k-43"></a>43. FERV39k

**Common**

- **name**: FERV39k
- **type**: dataset
- **subcategory**: dynamic-video
- **release_year**: 2022
- **authors_or_creators**:
  > Yan Wang, Yixuan Sun, Yiwen Huang, Zhongying Liu, Shuyong Gao, Wei Zhang, Weifeng Ge, Wenqiang Zhang. Hefei University of Technology / Fudan University.
- **paper_link**: https://arxiv.org/abs/2203.09463
- **publication_venue**: CVPR 2022

**Dataset**

- **num_images**: 38935
- **num_classes**: 7
- **label_taxonomy**: basic-7 (anger, disgust, fear, happiness, neutral, sadness, surprise)
- **in_the_wild_or_lab**: in-the-wild
- **download_url**: https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html
- **known_issues**:
  > Class imbalance is significant: 'happy', 'neutral', and 'sad' dominate while 'disgust' and 'fear' are very rare, leading to lower per-class recall on minority classes. Clips were sourced from web videos (movies, TV shows, daily-life clips, talk shows, etc.), so distribution skews toward Western/East Asian media and the dataset inherits actor-portrayed expression bias for cinematic scenes. Each clip is annotated by 30 crowdsourced annotators followed by professional re-check; some clips with low inter-annotator agreement were filtered out, but residual label ambiguity remains for subtle/compound expressions. No explicit demographic balancing across age/gender/ethnicity. Access requires signing a research-use agreement.
- **modality**: video
- **annotation_method**: crowdsourced (30 annotators per clip) with professional expert re-check / filtering
- **train_val_test_split**:
  > Official 80/20 train/test split (~31,148 train / ~7,787 test clips). Splits are scene-balanced; the benchmark also provides 4 super-scene-specific splits (Daily Life, Weak-Interactive Shows, Strong-Interactive Activities, Anomaly Issues) for cross-scene evaluation.

**Uncertain (skipped) fields**

- demographic_metadata
- license
- num_subjects
- resolution_or_fps
- top_sota_model

---

### <a id="jaffe-japanese-female-facial-expression-44"></a>44. JAFFE (Japanese Female Facial Expression)

**Common**

- **name**: JAFFE (Japanese Female Facial Expression)
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 1998
- **authors_or_creators**:
  > Michael J. Lyons, Shigeru Akamatsu, Miyuki Kamachi, Jiro Gyoba — ATR (Advanced Telecommunications Research Institute) / Kyushu University, Japan
- **paper_link**: https://doi.org/10.1109/AFGR.1998.670949
- **publication_venue**:
  > IEEE International Conference on Automatic Face and Gesture Recognition (FG 1998) — 'Coding Facial Expressions with Gabor Wavelets'

**Dataset**

- **num_images**: 213
- **num_subjects**: 10
- **num_classes**: 7
- **label_taxonomy**:
  > basic-7 (anger, disgust, fear, happiness, sadness, surprise, neutral). Each image is additionally rated by 60 Japanese subjects on 6 emotion adjectives (semantic ratings on a 5-point scale).
- **in_the_wild_or_lab**: lab-controlled
- **download_url**:
  > https://zenodo.org/record/3451524 (current controlled-release archive maintained by Michael Lyons); historical site http://www.kasrl.org/jaffe.html (deprecated)
- **known_issues**:
  > Severe demographic bias: only 10 subjects, all young adult Japanese females, photographed in a single lab session under controlled frontal lighting — extremely limited diversity in age, gender, ethnicity, pose, illumination, and background. Tiny size (213 images) makes deep-learning evaluation prone to overfitting and unstable cross-validation results. Expressions are posed (not spontaneous), so models trained or evaluated on JAFFE do not generalize to in-the-wild settings. The original distribution site was withdrawn in 2018 amid concerns over uncontrolled redistribution and use beyond the original research purpose; subsequent re-release imposes stricter access terms. Per-image semantic ratings show that many images are mixtures of emotions rather than pure prototypes, so single-label accuracy can be misleading.
- **modality**: image
- **annotation_method**:
  > expert (posed expressions directed by experimenters) + crowdsourced semantic ratings (60 Japanese university student raters per image on 6 emotion adjectives)
- **resolution_or_fps**: 256 x 256 grayscale TIFF
- **demographic_metadata**:
  > 10 Japanese female subjects (university-aged adults). 0% male, 0% non-Japanese, narrow age band. No annotated age/ethnicity metadata beyond the subject identity.

**Uncertain (skipped) fields**

- license
- top_sota_model
- train_val_test_split

---

### <a id="mafw-multi-modal-affective-facial-expressions-in-the-wild-45"></a>45. MAFW (Multi-modal Affective Facial expressions in the Wild)

**Common**

- **name**: MAFW (Multi-modal Affective Facial expressions in the Wild)
- **type**: dataset
- **subcategory**: dynamic-video
- **release_year**: 2022
- **authors_or_creators**:
  > Yuanyuan Liu, Wei Dai, Chuanxu Feng, Wenbin Wang, Guanghao Yin, Jiabei Zeng, Shiguang Shan. Beijing University of Posts and Telecommunications (BUPT) and Institute of Computing Technology, Chinese Academy of Sciences.
- **paper_link**:
  > https://dl.acm.org/doi/10.1145/3503161.3548190 (ACM MM 2022); extended journal version at IEEE TPAMI 2023
- **publication_venue**: ACM MM 2022 (extended in IEEE TPAMI 2023)

**Dataset**

- **num_images**: 10045
- **num_classes**: 43
- **label_taxonomy**:
  > mixed: 11 single-emotion classes (anger, disgust, fear, happiness, neutral, sadness, surprise, contempt, anxiety, helplessness, disappointment) + 32 compound-emotion classes; each clip additionally annotated with bilingual (Chinese + English) free-text emotion captions describing facial expression and emotional cause.
- **in_the_wild_or_lab**: in-the-wild
- **license**: research-only; access requires signing an End User License Agreement (EULA) via the official site.
- **download_url**: https://mafw-database.github.io
- **known_issues**:
  > Heavy class imbalance: the 11 single-emotion classes are dominated by happiness/neutral/sadness/anger while contempt, helplessness, disappointment, and most of the 32 compound classes are rare, making evaluation of compound emotions very low-shot. Clips are sourced from movies, TV dramas, and short videos (predominantly Chinese-language and Hollywood content), so cultural/linguistic bias skews Chinese + Western. Annotation is performed by 11 trained annotators with majority voting; subjective compound-emotion labels still suffer from inter-annotator disagreement. Faces of well-known actors recur across clips, raising potential identity leakage between train and test for subject-dependent evaluation. Audio track is included (multi-modal), but background music/non-speech audio can bias audio-visual models. EULA-restricted, so redistribution is prohibited.
- **modality**: audio-visual (also provides bilingual text captions, so effectively multimodal-text)
- **annotation_method**:
  > expert (11 trained annotators per clip with majority voting); captions written by trained annotators in both Chinese and English
- **train_val_test_split**:
  > Official 5-fold subject-independent cross-validation protocol for both the 11-class single-emotion task and the 43-class (11+32) compound task. Reported metrics are UAR (unweighted average recall) and WAR (weighted average recall) averaged over the 5 folds.

**Uncertain (skipped) fields**

- demographic_metadata
- num_subjects
- resolution_or_fps
- top_sota_model

---

### <a id="mead-multi-view-emotional-audio-visual-dataset-46"></a>46. MEAD (Multi-view Emotional Audio-visual Dataset)

**Common**

- **name**: MEAD (Multi-view Emotional Audio-visual Dataset)
- **type**: dataset
- **subcategory**: audio-visual
- **release_year**: 2020
- **authors_or_creators**:
  > Kaisiyuan Wang, Qianyi Wu, Linsen Song, Zhuoqian Yang, Wayne Wu, Chen Qian, Ran He, Yu Qiao, Chen Change Loy. Primary affiliations: SenseTime Research, CUHK, Chinese Academy of Sciences (CASIA), NTU.
- **paper_link**: https://wywu.github.io/projects/MEAD/support/MEAD.pdf
- **publication_venue**: ECCV 2020

**Dataset**

- **num_subjects**: 60
- **num_classes**: 8
- **label_taxonomy**:
  > basic-8 (neutral, happy, sad, angry, surprised, disgusted, fearful, contempt) with 3 intensity levels per emotion.
- **in_the_wild_or_lab**: lab-controlled
- **download_url**: https://wywu.github.io/projects/MEAD/MEAD.html
- **known_issues**:
  > Lab-controlled studio recordings limit ecological validity (clean background, controlled lighting, scripted utterances) so models trained on MEAD may not transfer well to in-the-wild settings. Acted (posed) emotions rather than spontaneous expressions. Demographic distribution skews toward East Asian / Chinese actors with limited ethnic diversity, and gender balance / age coverage details are not extensively documented. Some viewpoints have lower utility (extreme angles) and the contempt class is rare in many downstream FER taxonomies, complicating cross-dataset comparison. Access requires signing a release form.
- **modality**: audio-visual
- **annotation_method**:
  > expert (scripted recording protocol; emotion label and intensity assigned by experimental design rather than post-hoc annotation)
- **resolution_or_fps**: 1920x1080 video at 30 fps with synchronized 48 kHz audio [uncertain on exact fps/audio rate]
- **demographic_metadata**:
  > 60 actors balanced across genders; predominantly East Asian (Chinese) ethnicity with some other ethnicities; adult age range. Cultural diversity is limited compared to in-the-wild datasets.

**Uncertain (skipped) fields**

- license (exact terms)
- num_images (exact clip count)
- resolution_or_fps (exact fps and audio sample rate)
- top_sota_model (no canonical FER classification leaderboard)
- train_val_test_split (no universal protocol)

---

### <a id="mmi-facial-expression-database-47"></a>47. MMI Facial Expression Database

**Common**

- **name**: MMI Facial Expression Database
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2005
- **authors_or_creators**:
  > Maja Pantic, Michel F. Valstar, Ron Rademaker, Ludo Maat. Imperial College London (Department of Computing, iBUG group) and University of Twente (Human Media Interaction group).
- **paper_link**: https://doi.org/10.1109/ICME.2005.1521424
- **publication_venue**: Proceedings of IEEE International Conference on Multimedia and Expo (ICME) 2005

**Dataset**

- **num_classes**:
  > 6 basic emotions (anger, disgust, fear, happiness, sadness, surprise) plus neutral; in addition, a separate Action Unit annotation track covers around 30+ FACS AUs.
- **label_taxonomy**:
  > mixed: basic-6 emotion labels at sequence level, plus action-units (FACS AUs with onset/apex/offset frame-level temporal segmentation). Some parts include neutral, making it effectively basic-7 when neutral is included.
- **in_the_wild_or_lab**: lab-controlled
- **license**: research-only (web-accessible after registration via EULA on the MMI / iBUG project page)
- **download_url**:
  > https://mmifacedb.eu/ (also linked from https://ibug.doc.ic.ac.uk/resources/mmi-facial-expression-database/ — registration and signed EULA required)
- **known_issues**:
  > The dataset mixes posed and (a smaller set of) spontaneous expressions, with most early FER literature using only the posed subset, so 'MMI accuracy' numbers in the literature are not directly comparable across studies. Sequences begin and end at neutral and peak in the middle (apex), so frame-selection is non-trivial — the common practice is to extract the three peak frames per sequence, but exact protocols vary. There is no official train/val/test split, and 10-fold subject-independent cross-validation is the de facto standard, again with non-standard fold definitions across papers. Subjects wear glasses, accessories, and have moustaches/beards in several sequences which is realistic but introduces confounds. Demographic coverage is limited and Eurocentric (mostly European subjects, predominantly young adults). The AU annotations are expert-coded but only a subset of sequences are fully AU-coded; coverage is uneven. Profile-view recordings exist for some sessions but are far less commonly used. Access requires registration and the website has occasionally been intermittently available.
- **modality**:
  > video (frontal-view sequences, with profile-view also available for a subset) plus a smaller set of still images
- **annotation_method**:
  > expert (FACS-certified coders provided AU labels with onset/apex/offset frame indices; emotion labels were assigned by the experimenters based on the requested or perceived expression)
- **train_val_test_split**:
  > No official split. Standard protocol is 10-fold subject-independent cross-validation on the posed subset of approximately 200-250 sequences from ~30 subjects, using the three apex frames per sequence (a setup popularized by Liu et al. 2014 and used in most subsequent CNN-based FER papers). AU-detection works typically use leave-one-subject-out cross-validation.

**Uncertain (skipped) fields**

- demographic_metadata
- num_images
- num_subjects
- resolution_or_fps
- top_sota_model

---

### <a id="oulu-casia-nirvis-facial-expression-database-48"></a>48. Oulu-CASIA NIR&VIS Facial Expression Database

**Common**

- **name**: Oulu-CASIA NIR&VIS Facial Expression Database
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2011
- **authors_or_creators**:
  > Guoying Zhao, Xiaohua Huang, Matti Taini, Stan Z. Li, Matti Pietikainen. Joint work between the Center for Machine Vision Research, University of Oulu (Finland) and the Institute of Automation, Chinese Academy of Sciences (CASIA, China).
- **paper_link**: https://doi.org/10.1016/j.imavis.2011.07.002
- **publication_venue**: Image and Vision Computing (IVC), Vol. 29, Issue 9, 2011

**Dataset**

- **num_subjects**: 80
- **num_classes**: 6
- **label_taxonomy**:
  > basic-6 (anger, disgust, fear, happiness, sadness, surprise) — neutral is implicitly the onset frame and is not part of the 6-way classification
- **in_the_wild_or_lab**: lab-controlled
- **download_url**:
  > https://www.oulu.fi/cmvs/node/41316 (University of Oulu CMVS dataset page; access requires a signed EULA request)
- **known_issues**:
  > Lab-controlled with a relatively small subject pool (80) and only posed expressions, leading to limited generalization to in-the-wild conditions. Demographic coverage is biased: roughly half subjects from Finland and half from China, predominantly young adults (≈23-58 years), with a male-skewed gender distribution; African and other ethnicities are not represented. Each expression sequence runs from neutral onset to peak apex, so frame-selection (apex vs. all frames) strongly affects reported accuracy and makes cross-paper comparison tricky. The standard 10-fold subject-independent protocol is not officially fixed, so different papers use different fold splits. Some sequences are missing or incomplete for certain (subject x illumination x modality) combinations. Only 6 basic emotions are labeled; no compound, AU, or valence-arousal annotations are provided.
- **modality**: image (dual modality: Near-Infrared NIR + Visible-light VIS, captured simultaneously)
- **annotation_method**:
  > expert (posed expressions performed by subjects on instruction; labels assigned per the requested expression category, no separate FACS coding)
- **train_val_test_split**:
  > No official train/val/test split. The de facto standard protocol is 10-fold subject-independent cross-validation on the VIS strong-illumination subset using the last 1-3 apex frames of each sequence; results are reported as mean accuracy across folds. Some works also report NIR-only or NIR+VIS fusion results.

**Uncertain (skipped) fields**

- demographic_metadata
- license
- num_images
- resolution_or_fps
- top_sota_model

---

### <a id="raf-db-real-world-affective-faces-database-49"></a>49. RAF-DB (Real-world Affective Faces Database)

**Common**

- **name**: RAF-DB (Real-world Affective Faces Database)
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2017
- **authors_or_creators**:
  > Shan Li, Weihong Deng, JunPing Du. Pattern Recognition and Intelligent System Laboratory, Beijing University of Posts and Telecommunications (BUPT).
- **paper_link**:
  > https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
- **publication_venue**: CVPR 2017 (extended in IEEE TIP 2019)

**Dataset**

- **num_images**: 29672
- **num_classes**: 7 (basic) / 11 (compound, RAF-ML extension uses 12 compound categories)
- **label_taxonomy**:
  > mixed: basic-7 (surprise, fear, disgust, happiness, sadness, anger, neutral) for the basic subset; compound (e.g., happily surprised, sadly fearful) for the compound subset; RAF-ML extension provides multi-label compound annotations
- **in_the_wild_or_lab**: in-the-wild
- **license**: research-only (EULA / signed agreement required via release form on the BUPT PRIS lab site)
- **download_url**: http://www.whdeng.cn/RAF/model1.html
- **known_issues**:
  > Demographic imbalance: skewed toward Caucasian faces and adults; provided age/gender/race attributes are coarse. Class imbalance is significant in the basic subset (happiness and neutral dominate; fear and disgust are rare), which inflates Overall Accuracy (WAR) over Mean-Class Accuracy (UAR). Compound subset is much smaller and noisier than the basic subset. Some label noise remains despite the 40-annotator EM-based reliable crowdsourcing protocol. Images sourced from the internet, raising consent and copyright considerations; redistribution is forbidden under the EULA.
- **modality**: image
- **annotation_method**:
  > crowdsourced (about 40 annotators per image, aggregated with an EM-based reliable estimation algorithm); attribute labels (age group, gender, race) also provided
- **resolution_or_fps**: Aligned faces released at 100x100 pixels; original images vary in resolution
- **train_val_test_split**:
  > Basic subset: 15,339 images split into 12,271 training and 3,068 test images (random split, not subject-independent). Compound subset: 3,954 training and 989 test images. No official validation split.
- **demographic_metadata**:
  > Annotated attributes: 5 age groups (0-3, 4-19, 20-39, 40-69, 70+), 3 gender categories (male, female, unsure), 3 race categories (Caucasian, African-American, Asian). Caucasian and adult subjects dominate; African-American and elderly subjects are underrepresented.

**Uncertain (skipped) fields**

- num_subjects
- top_sota_model

---

### <a id="sfew-20-static-facial-expressions-in-the-wild-20-50"></a>50. SFEW 2.0 (Static Facial Expressions in the Wild 2.0)

**Common**

- **name**: SFEW 2.0 (Static Facial Expressions in the Wild 2.0)
- **type**: dataset
- **subcategory**: static-image
- **release_year**: 2015
- **authors_or_creators**:
  > Abhinav Dhall, Roland Goecke, Simon Lucey, Tamas Gedeon — Australian National University (ANU) and University of Canberra. Original SFEW introduced at ICCV-W 2011; SFEW 2.0 released as the static benchmark for the EmotiW 2015 challenge.
- **paper_link**:
  > https://doi.org/10.1109/ICCVW.2011.6130508 (original SFEW, 2011); EmotiW 2015 description: https://doi.org/10.1145/2818346.2829994
- **publication_venue**:
  > IEEE ICCV Workshops (BEFIT) 2011 — 'Static Facial Expressions in Tough Conditions: Data, Evaluation Protocol and Benchmark'; updated as SFEW 2.0 in the EmotiW 2015 Challenge (ACM ICMI Workshops).

**Dataset**

- **num_images**: 1766
- **num_classes**: 7
- **label_taxonomy**: basic-7 (anger, disgust, fear, happiness, sadness, surprise, neutral)
- **in_the_wild_or_lab**:
  > in-the-wild (frames extracted from feature-length movies, exhibiting unconstrained pose, illumination, occlusion, and expression intensity)
- **license**:
  > Research-only EULA; redistribution prohibited. Access granted by the authors after signing an end-user license agreement (request via the EmotiW / SFEW website).
- **download_url**:
  > https://cs.anu.edu.au/few/emotiw.html (EmotiW challenge page — request form for SFEW/AFEW); historical: https://users.cecs.anu.edu.au/~few/AFEW.html
- **known_issues**:
  > Small dataset (~1.8k images) with significant class imbalance (happy and neutral are over-represented; disgust and fear are under-represented). Faces are extracted from movies, so they depict acted (not fully spontaneous) emotion and contain motion blur, occlusion, low resolution, and extreme poses — making absolute accuracy low (early baselines ~26-39%). Test-set labels are withheld (server-side evaluation only during EmotiW), which limits open SOTA comparison after the challenge cycle. Demographic skew toward Western (Hollywood) actors. Some label noise reported because frames are sampled near AFEW clip peaks but expressions evolve over the clip, occasionally yielding ambiguous single-frame labels. Subject overlap between train/val/test was minimized but not formally verified subject-independent across all splits in earlier releases.
- **modality**: image
- **annotation_method**:
  > expert (semi-automatic — frames selected by the Recommender system from AFEW clips and labeled by trained human annotators using clip-level emotion labels propagated to frames; verified by experts)
- **train_val_test_split**:
  > Subject-independent (best-effort) split: Train ~958 images / Val ~436 images / Test ~372 images. Test labels are withheld; Val accuracy is the de-facto reported number in the literature.

**Uncertain (skipped) fields**

- demographic_metadata
- num_subjects
- resolution_or_fps
- top_sota_model

---
