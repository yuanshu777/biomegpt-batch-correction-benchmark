# BiomeGPT Real Study-ID Batch Correction: Full Workflow Report

Date: 2026-05-21  
Scope: BiomeGPT / taxonomy-aware phase-2 gut checkpoint / real study-id batch correction  

## 0. 一句话总结

我们围绕 BiomeGPT phase-2 gut pretrained model 做了完整的 batch diagnostics 和 batch correction 探索。核心发现是：

1. BiomeGPT phase-2 `<cls>` sample embedding 里确实有很强的 real-study signal。
2. real study id 和 disease/phenotype 存在严重 confounding。
3. 直接做 model-level GRL、centroid alignment、distillation、batch-conditioned decoder、batch-token continued pretraining，都没有真正把 real-study probe 降下来。
4. 目前唯一明确有效的 batch-corrected representation 是 **cross-fitted real-study mean centering**，尤其是 conservative-safe panel。
5. 如果要继续做 model-level correction，应该从更早的 phase2 pretraining 开始加入 batch-aware architecture，或者使用 frozen embedding correction adapter，而不是继续无约束 fine-tune backbone。

---

## 1. 背景和目标

我们最初是在做 taxonomy-aware BiomeGPT。这个模型的主要 pretraining objective 不是疾病预测，而是：

```text
masked abundance prediction
```

也就是随机 mask nonzero species abundance bin，让模型根据剩余 species abundance structure 预测被 mask 的 abundance bin。

因此 batch correction 的目标不是简单提高 Healthy-vs-Diseased classification，而是：

```text
reduce study/batch information in sample embedding
preserve masked abundance reconstruction
preserve useful biological / disease signal
```

换句话说，我们关心三件事：

| Layer | Question |
|---|---|
| Batch removal | `<cls>` embedding 还能不能预测 study id？希望下降 |
| Foundation objective | masked abundance MSE 有没有坏？希望不坏，最好下降 |
| Biological sanity | H/D probe 有没有崩？希望不崩 |

---

## 2. 数据和 real study-id annotation

后来上传了：

```text
BiomeGPT_species_samples_studyIDs.csv
```

这个文件包含：

```text
sample_id -> study_name
```

我们因此从之前的 proxy batch labels 转向真实 study id。

### 2.1 数据处理

脚本：

```text
dataset_v3/prepare_real_study_batch_annotation.py
```

做了这些事：

1. 读取 `BiomeGPT_species_samples_studyIDs.csv`。
2. 因为文件中部分 HMP rows 有不规则 quote，所以用 `quoting=csv.QUOTE_NONE` 读取。
3. 清理 `study_name`：
   - strip whitespace
   - strip quotes
   - whitespace 转 underscore
4. 用 `sample_id` 和 phase2 gut metadata 对齐。
5. 生成：

```text
batch_label_external_recommended = real_study:<study_name>
```

6. 根据 phenotype distribution 重新计算 conservative-safe label。

### 2.2 覆盖率

| Quantity | Count |
|---|---:|
| Uploaded study-id rows | 13,524 |
| Phase-2 gut samples | 13,332 |
| Matched to real study id | 13,326 |
| Missing real study id | 6 |
| Unique real studies in phase2 gut | 80 |
| Conservative-safe samples | 4,751 |
| Conservative-safe studies | 26 |

结论：real study-id coverage 很好，可以作为主要 batch annotation。

---

## 3. Real Study-ID Batch Diagnostics

脚本：

```text
dataset_v3/batch_effect_diagnostics.py
```

输入：

```text
dataset_v3/meta_pretraining_phase2_gut_real_study_annotation.csv
dataset_v3/outputs_taxonomy_notebook/taxonomy_checkpoint_stage2.pt
```

输出目录：

```text
dataset_v3/outputs_batch_diagnostics_real_study/
```

### 3.1 Probe 方法

我们先从 phase2 checkpoint 提取每个 sample 的 512-dim `<cls>` embedding：

```text
phase2_sample_prompt_embeddings.npz
```

然后训练 shallow logistic regression probe：

```text
embedding -> real study id
embedding -> Healthy vs Diseased
embedding -> phenotype
```

probe 的意义：

```text
如果一个简单线性 probe 能预测 study id，说明 embedding 里包含 study/batch information。
```

### 3.2 Real-study probe 结果

| Probe panel | Samples | Classes | Macro-F1 | Balanced acc. |
|---|---:|---:|---:|---:|
| Real study, high-only | 13,326 | 80 | 0.456 | 0.495 |
| Real study, conservative-safe | 4,751 | 26 | 0.518 | 0.537 |
| Healthy vs Diseased | 13,332 | 2 | 0.796 | 0.801 |

解释：

80-class study classification 的 dummy balanced accuracy 只有 0.0125，所以 0.495 balanced accuracy 很高。这说明 BiomeGPT phase2 embedding 明显包含 real-study signal。

### 3.3 Phenotype confounding

| Panel | Studies | Cramer's V | NMI | Studies with top phenotype >= 0.8 |
|---|---:|---:|---:|---:|
| High-only real study | 80 | 0.792 | 0.526 | 54 |
| Conservative-safe real study | 26 | 0.634 | 0.528 | 0 |

关键解释：

high-only real study labels 里，很多 study 几乎对应单一 phenotype。也就是说：

```text
study id and disease phenotype are strongly confounded
```

所以如果直接 remove all study signal，很可能把 disease biology 一起删掉。

这就是为什么我们后续主要使用：

```text
conservative_safe labels
```

---

## 4. Baseline: GRL / DAB Adversarial Correction

脚本：

```text
dataset_v3/batch_adversarial_correction.py
```

核心 loss：

```text
loss =
  masked_abundance_reconstruction_loss
  + dab_weight * batch_cross_entropy_with_GRL
```

GRL = gradient reversal layer。  
DAB = domain adversarial batch classifier。

目标：

```text
batch discriminator learns to predict study
encoder receives reversed gradient
so encoder should hide study information
```

### 4.1 加入 warm-up

我们发现固定 DAB pressure 可能不稳定，所以加了：

```text
--dab_warmup_epochs
--grl_warmup_epochs
```

### 4.2 Real-study GRL trials

| Trial | Study F1 before | Study F1 after | H/D F1 before | H/D F1 after |
|---|---:|---:|---:|---:|
| Safe, DAB 0.05, batch 8 | 0.518 | 0.580 | 0.796 | 0.798 |
| Safe, DAB 0.10, batch 8 | 0.518 | 0.547 | 0.796 | 0.810 |
| High-only, DAB 0.05, batch 16 | 0.456 | 0.493 | 0.796 | 0.804 |

结论：

GRL 没有 remove batch。study probe 反而更强。  
但是 H/D 不坏，甚至提升。

解释：

```text
model continues learning abundance / phenotype / cohort structure
but adversarial signal is not strong or stable enough to make <cls> study-invariant
```

---

## 5. Masked-Abundance Reconstruction Diagnostics

脚本：

```text
dataset_v3/batch_reconstruction_diagnostics.py
```

我们用 deterministic mask，在不同 checkpoints 上比较 masked abundance MSE。

### 5.1 GRL checkpoint reconstruction

| Checkpoint | Overall MSE | Safe-panel MSE | High-only MSE |
|---|---:|---:|---:|
| Base phase2 | 34.856 | 36.221 | 34.857 |
| Safe GRL, DAB 0.05 | 33.235 | 33.852 | 33.235 |
| Safe GRL, DAB 0.10 | 33.164 | 33.755 | 33.165 |
| High-only GRL, DAB 0.05 | 32.911 | 34.225 | 32.911 |

结论：

GRL trials 没有 catastrophic forgetting。  
相反，masked abundance reconstruction MSE 变好。

所以失败点不是 pretraining objective 崩了，而是：

```text
reconstruction improves while study separability also improves
```

---

## 6. Post-hoc Real-Study Embedding Correction

脚本：

```text
dataset_v3/real_study_embedding_correction.py
```

这是目前最成功的 batch correction baseline。

### 6.1 方法

对 base phase2 embeddings 做 cross-fitted real-study mean centering：

```text
x_corrected = x - study_mean + global_mean
```

cross-fitted 的意思是：

```text
sample 的 corrected embedding 不用它自己的 embedding 来估计 study mean
```

这样避免 trivial leakage。

### 6.2 结果

| Panel | Method | Study F1 before | Study F1 after | H/D F1 before | H/D F1 after |
|---|---|---:|---:|---:|---:|
| High-only | Mean-center | 0.456 | 0.002 | 0.795 | 0.473 |
| High-only | Mean+scale | 0.456 | 0.008 | 0.795 | 0.471 |
| Conservative-safe | Mean-center | 0.518 | 0.004 | 0.690 | 0.637 |
| Conservative-safe | Mean+scale | 0.518 | 0.031 | 0.690 | 0.644 |

结论：

1. real-study signal 是可以去掉的。
2. high-only broad correction 会严重伤害 H/D signal。
3. conservative-safe correction 更合理，study signal 几乎去掉，同时 H/D 损失较小。

保存的 corrected embeddings：

```text
dataset_v3/outputs_real_study_embedding_correction_saved/
  real_study_high_only_mean_center_corrected_embeddings.npz
  real_study_conservative_safe_mean_center_corrected_embeddings.npz
```

目前最可用的是：

```text
real_study_conservative_safe_mean_center_corrected_embeddings.npz
```

---

## 7. Model-level Representation Alignment

我们尝试把 post-hoc mean centering 的成功转成 model-level loss。

### 7.1 Minibatch centroid alignment

脚本：

```text
dataset_v3/batch_adversarial_correction.py
```

新增：

```text
--alignment_loss centroid/coral/mmd
--alignment_weight
--balanced_study_batches
```

我们主要跑了 centroid alignment：

```text
loss =
  reconstruction_loss
  + alignment_weight * distance(study_centroids, global_centroid)
```

为了让 centroid 有意义，我们加了 balanced study sampler：

```text
each minibatch contains multiple studies
each study has multiple samples
```

### 7.2 Centroid alignment 结果

| Method | Study F1 before | Study F1 after | H/D before | H/D after | Recon MSE |
|---|---:|---:|---:|---:|---:|
| Base phase2 | 0.518 | - | 0.796 | - | 34.856 |
| Centroid align w10 | 0.518 | 0.598 | 0.796 | 0.807 | 33.697 |
| Centroid align w50 | 0.518 | 0.579 | 0.796 | 0.819 | 33.628 |

结论：

minibatch centroid loss 不能 remove real-study signal。  
它改善 reconstruction 和 H/D，但 independent study probe 变强。

---

## 8. Cross-fitted Centroid Distillation

脚本：

```text
dataset_v3/batch_centroid_distillation.py
```

思路：

既然 post-hoc corrected embedding 有效，那就把它作为 training target：

```text
target = base_embedding - cross_fitted_study_mean + cross_fitted_global_mean
```

然后训练：

```text
loss =
  masked_abundance_reconstruction_loss
  + target_weight * MSE(unmasked_cls_embedding, target)
```

关键 debug：

最初 target loss 加在 masked encoder output 上，效果不对。  
后来改成约束：

```text
model.sample_prompt(bins)
```

也就是 probe 真正用的 unmasked `<cls>` embedding。

### 8.1 Distillation 结果

| Method | Study F1 before | Study F1 after | H/D before | H/D after | Recon MSE |
|---|---:|---:|---:|---:|---:|
| Unmasked distill w10 | 0.518 | 0.532 | 0.796 | 0.801 | 33.656 |
| Unmasked distill w50 | 0.518 | 0.556 | 0.796 | 0.802 | 33.587 |
| Unmasked distill w200 | 0.518 | 0.576 | 0.796 | 0.800 | 33.656 |

结论：

target loss 确实下降，但 independent study probe 仍然没有下降。  
说明 encoder dynamics 仍然保留/强化 study-separable structure。

---

## 9. scGPT-style Batch-Conditioned Decoder

用户指出 scGPT 里有 batch-aware mechanism，例如 batch labels / batch embeddings / domain-specific batch norm。  
我们因此测试：BiomeGPT 是不是因为没有 batch side-channel，才被迫把 study effect 写进 embedding。

脚本：

```text
dataset_v3/batch_conditioned_decoder_correction.py
```

### 9.1 方法

加入 batch-conditioned residual reconstruction decoder：

```text
pred =
  base_reconstruction_head(h_species)
  + residual_head(h_species, study_id)
```

约束：

```text
study id only goes to reconstruction residual decoder
study id does not go to sample_prompt
downstream probe still uses unconditioned sample_prompt
```

### 9.2 Trials

| Trial | Study F1 before | Study F1 after | H/D F1 before | H/D F1 after |
|---|---:|---:|---:|---:|
| Batch decoder + normalized target w50 | 0.518 | 0.565 | 0.796 | 0.805 |
| Batch decoder + DAB 0.1 | 0.518 | 0.577 | 0.796 | 0.804 |
| Batch decoder + detached recon encoder | 0.518 | 0.592 | 0.796 | 0.796 |
| Batch decoder + raw target w1 | 0.518 | 0.558 | 0.796 | 0.803 |

结论：

后加 residual decoder 不够。  
H/D 保持，但 study probe 仍然变强。

解释：

```text
scGPT-style batch-aware design likely needs to be part of pretraining from earlier,
not patched onto an already formed phase2 representation.
```

---

## 10. Batch-Token Continued Pretraining

为了更接近 scGPT，我们另起了一个新文件，从 architecture 上加入 STUDY token。

脚本：

```text
dataset_v3/batch_token_pretraining.py
```

### 10.1 方法

训练时序列变成：

```text
[CLS, STUDY, species_1, species_2, ...]
```

attention mask 设计：

```text
species tokens can attend to STUDY
CLS cannot attend to STUDY
downstream sample_prompt still uses original unconditioned model.sample_prompt(bins)
```

目标是：

```text
reconstruction can use study side-channel
but final embedding does not directly receive study id
```

### 10.2 Batch-token trials

| Trial | Study F1 before | Study F1 after | H/D F1 before | H/D F1 after | Recon MSE |
|---|---:|---:|---:|---:|---:|
| Base phase2 | 0.518 | - | 0.796 | - | 34.856 |
| Batch token, reconstruction only | 0.518 | 0.529 | 0.796 | 0.799 | 33.582 |
| Batch token + DAB 0.1 | 0.518 | 0.548 | 0.796 | 0.805 | 33.594 |
| Batch token + corrected-target distill w10 | 0.518 | 0.544 | 0.796 | 0.800 | 33.584 |

结论：

batch-token continued pretraining 是目前最稳定的 model-level batch-aware 设计：

```text
reconstruction improves
H/D preserved
study separability only mildly increases in reconstruction-only version
```

但它仍然没有真正 remove batch。

解释：

```text
Adding STUDY token after phase2 is probably too late.
To fully test scGPT-style design, batch token should be present before or at the start of phase2.
```

---

## 11. 当前最重要的结论

### 11.1 关于数据

real study id 是现在最好的 batch label。  
但是 full high-only real study labels 和 phenotype 高度 confounded。

因此：

```text
do not blindly remove all real-study signal
```

### 11.2 关于模型

目前所有 model-level continued fine-tuning 都有类似现象：

```text
masked abundance reconstruction improves
H/D signal preserved or improves
real-study probe does not decrease
```

这说明：

```text
the model is still learning study-separable abundance / phenotype structure
```

### 11.3 关于最可用结果

目前最实际可用的 corrected representation 是：

```text
cross-fitted conservative-safe real-study mean-centered embedding
```

文件：

```text
dataset_v3/outputs_real_study_embedding_correction_saved/
  real_study_conservative_safe_mean_center_corrected_embeddings.npz
```

### 11.4 关于 scGPT-style batch token

这个 idea 是合理的。  
但是当前结果说明：

```text
late continued pretraining with batch token is not enough
```

真正要测试这个 idea，应该从更早 checkpoint 开始：

```text
phase1 -> phase2 with batch token
or at least stage1 checkpoint -> phase2 batch-token pretraining
```

---

## 12. Recommended Next Steps

### Step 1: Use corrected embeddings for downstream prediction

先不要继续盲目 fine-tune backbone。  
用已经成功的 conservative-safe corrected embedding 做 downstream：

```text
raw phase2 embedding
vs
conservative-safe corrected embedding
```

评估：

```text
Healthy-vs-Diseased
specific disease
leave-one-study-out
external validation
```

### Step 2: Leave-one-study-out validation

这是最能说服教授的实验：

```text
train on studies A, B, C, ...
test on held-out study K
```

看 corrected embedding 是否比 raw embedding 更稳。

### Step 3: Start batch-token pretraining from earlier checkpoint

如果一定要 model-level batch-aware BiomeGPT：

```text
start from stage1 checkpoint
run phase2 with STUDY token from the beginning
```

而不是从已经完成的 stage2 checkpoint 后加。

### Step 4: Frozen adapter / study-offset correction

另一个更工程可控的路线：

```text
z_raw = frozen BiomeGPT sample embedding
z_corrected = z_raw - learned_study_offset[study_id]
```

这和我们 post-hoc mean-centering 成功结果最一致。

---

## 13. Files Created / Modified

### Data annotation

```text
dataset_v3/prepare_real_study_batch_annotation.py
dataset_v3/meta_pretraining_phase2_gut_real_study_annotation.csv
```

### Diagnostics

```text
dataset_v3/batch_effect_diagnostics.py
dataset_v3/batch_reconstruction_diagnostics.py
dataset_v3/outputs_batch_diagnostics_real_study/
```

### GRL / adversarial correction

```text
dataset_v3/batch_adversarial_correction.py
```

### Post-hoc embedding correction

```text
dataset_v3/real_study_embedding_correction.py
dataset_v3/outputs_real_study_embedding_correction_saved/
```

### Model-level alignment

```text
dataset_v3/batch_centroid_distillation.py
```

### Batch-conditioned decoder

```text
dataset_v3/batch_conditioned_decoder_correction.py
```

### Batch-token continued pretraining

```text
dataset_v3/batch_token_pretraining.py
```

### Reports

```text
reports/batch_diagnostics_report.tex
reports/batch_diagnostics_report.pdf
reports/batch_correction_full_workflow_report.md
dataset_v3/outputs_batch_diagnostics_real_study/*status.md
```

---

## 14. Final Professor-facing Summary

The most concise explanation is:

```text
We found that BiomeGPT phase-2 embeddings strongly encode real study identity.
However, study identity is heavily confounded with phenotype, so naive batch removal can
remove biological signal. We tested GRL adversarial correction, centroid alignment,
corrected-target distillation, batch-conditioned decoding, and batch-token continued
pretraining. These model-level methods preserved or improved masked abundance
reconstruction and H/D signal, but did not reduce independent real-study predictability.
The only clearly effective correction so far is cross-fitted real-study mean centering,
especially on conservative-safe studies. The next step is to validate these corrected
embeddings in cross-study prediction, and if model-level correction is required, start
batch-token phase2 pretraining from an earlier checkpoint.
```

