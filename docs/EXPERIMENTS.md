# Experimental Design Details

This document describes the detailed experimental design and methodology used in the paper.

## Table of Contents

1. [Dataset Configuration](#dataset-configuration)
2. [Encoder Experiments](#encoder-experiments)
3. [Decoder Experiments](#decoder-experiments)
4. [Statistical Analysis](#statistical-analysis)
5. [Reproducibility](#reproducibility)

## Dataset Configuration

### Data Split

The complete dataset (1,000 CT reports) was split as follows:

| Split | Size | Purpose |
|-------|------|---------|
| Train | 700 | Model training (encoder), Few-shot examples source (decoder SFT) |
| Val | 100 | Hyperparameter tuning (encoder), ICL demonstration pool (decoder) |
| Test | 200 | Final evaluation (all models) |

**Important**: The same test set was used for all experiments to ensure fair comparison.

### Class Imbalance Configurations

We systematically varied the positive rate in the training set while maintaining:
- Total training size: 700 samples
- Same validation and test sets across all configurations

| Configuration | Positive Rate | Ratio | # Positive | # Negative |
|---------------|--------------|-------|------------|------------|
| Base | 14.4% | 1:6 | ~101 | ~599 |
| Ratio-10 | 10.0% | 1:10 | 70 | 630 |
| Ratio-20 | 5.0% | 1:20 | 35 | 665 |
| Ratio-50 | 2.0% | 1:50 | 14 | 686 |

### Preprocessing Steps

1. **Text Concatenation**: Combine "Findings" and "Impression" sections
2. **Whitespace Normalization**: Remove extra spaces and line breaks
3. **Punctuation**: Append Japanese period ("。") if missing
4. **Character Normalization**:
   - Katakana → Full-width
   - Alphanumeric → Half-width
   - Remove symbols
5. **Number Normalization**: Replace all numbers with '0'

## Encoder Experiments

### Models Evaluated

| Model | Parameters | Domain | Notes |
|-------|-----------|--------|-------|
| ModernBERT-ja-310m | 310M | General | Selected as primary model |
| DeBERTa-v3-base-japanese | 110M | General | Strong baseline |
| UTH-BERT-base | 110M | Medical | Medical domain-specific |
| MedBERT-CR-base | 110M | Medical | Clinical report focused |
| BERT-base-japanese | 110M | General | Original Japanese BERT |
| RoBERTa-base-japanese | 110M | General | Japanese RoBERTa |
| ELECTRA-base-japanese | 110M | General | ELECTRA variant |

Selection criteria:
- Maximum sequence length ≥512 tokens
- Support for Japanese text
- Publicly available
- Released within last 3 years (2022-2024)

### Loss Functions

#### 1. Cross Entropy (CE)
Standard binary classification loss:

```python
loss = -[y * log(p) + (1-y) * log(1-p)]
```

#### 2. Focal Loss
Emphasizes hard examples:

```python
loss = -α * (1-p)^γ * y * log(p) + α * p^γ * (1-y) * log(1-p)
```

Parameters:
- α = [0.25, 0.75] (class weights)
- γ = 2.0 (focusing parameter)

#### 3. Class-Balanced (CB) Loss
Uses effective number of samples:

```python
E_n = (1 - β^n) / (1 - β)
weight = 1 / E_n
```

Parameters:
- β = 0.9999

#### 4. Influence-Balanced (IB) Loss
Normalizes gradient magnitudes:

```python
weight = 1 / (n_pos + n_neg) * gradient_norm
```

#### 5. IB-Focal Loss
Combines IB weighting with Focal Loss focusing.

### Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Batch size | 32 | Constant across all experiments |
| Learning rate | 2×10⁻⁵ | Optimized on validation set |
| Max epochs | 20 | Early stopping enabled |
| Optimizer | AdamW | Default parameters |
| Warmup steps | 500 | Linear warmup |
| Weight decay | 0.01 | L2 regularization |
| Gradient clipping | 1.0 | Prevent exploding gradients |

**Early Stopping**:
- Monitor: Validation F1 score
- Patience: 3 epochs
- Mode: Maximize

### Evaluation Metrics

All metrics computed on test set:

1. **Accuracy**: Overall correctness
2. **Precision**: Positive predictive value
3. **Recall**: True positive rate (sensitivity)
4. **F1 Score**: Harmonic mean of precision and recall
5. **AUROC**: Area Under ROC Curve
6. **AUPRC**: Area Under Precision-Recall Curve
7. **MCC**: Matthews Correlation Coefficient

**Primary Metric**: F1 Score (balances precision and recall)

## Decoder Experiments

### Models Evaluated

| Model | Size | Family | Notes |
|-------|------|--------|-------|
| Llama-3-ELYZA-JP-8B | 8B | Llama 3 | Primary model |
| Swallow-8B-Instruct | 8B | Llama 3.1 | Japanese-focused |
| Qwen2.5-7B-Instruct | 7B | Qwen | Multilingual |
| Gemma-3-4b-it | 4B | Gemma 3 | Google model |
| MedGemma-4b-it | 4B | Gemma 3 | Medical variant |
| Llama-3.2-3B-Instruct | 3B | Llama 3.2 | Smaller variant |
| Gemma-2-2b-jpn-it | 2B | Gemma 2 | Japanese-tuned |

Selection criteria:
- On-premises deployment feasible
- Japanese language support
- Instruction-following capability
- Publicly available weights

### Few-Shot In-Context Learning (ICL)

#### Experimental Factors

**1. Number of Demonstrations**

Tested: 0, 1, 2, 5, 10, 15, 25 (ELYZA model)
Tested: 0, 5, 10 (other models)

**2. Presentation Order** (10 demonstrations, 1:1 ratio)

- `alternating`: Alternate positive and negative
- `label0_first`: All negative examples first
- `label1_first`: All positive examples first

**3. Label Ratio** (10 demonstrations, randomized order)

- 0:10 (only negative)
- 1:9
- 3:7
- 5:5 (balanced)
- 7:3
- 9:1
- 10:0 (only positive)

#### Prompt Template

```
Task: Classify whether the following CT findings contain actionable findings.

[Few-shot demonstrations]
Example 1:
Report: {report_1}
Label: {label_1}

Example 2:
Report: {report_2}
Label: {label_2}

...

Target report:
{target_report}

Classification (output 0 or 1 only):
```

#### Inference Settings

**Few-shot prompting**:
- Temperature: 1×10⁻⁸ (nearly deterministic)
- do_sample: False
- max_new_tokens: 2

**Rationale**: Binary classification requires deterministic output.

#### Demonstration Sampling

- Source: Validation set
- Method: Random sampling with replacement
- Seeds: 12, 22, 32, 42, 52
- Repeats: 5 evaluations per configuration
- `resample_examples`: True (different examples per seed)

**Important**: VAL evaluation skipped by default (TEST only) for efficiency.

### Supervised Fine-Tuning (SFT)

#### Method: QLoRA

**LoRA Configuration**:
```python
lora_config = {
    "r": 16,              # Rank
    "lora_alpha": 32,     # Scaling factor
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "SEQ_CLS"
}
```

**Quantization**: 4-bit (NF4) for memory efficiency

**Training Parameters**:
- Batch size: 32
- Learning rate: 2×10⁻⁴
- Epochs: 3
- Optimizer: AdamW (paged_adamw_8bit)
- Warmup: 100 steps

**Loss Computation**: Only on generated label token ('0' or '1')

### Chain-of-Thought (CoT)

#### Zero-shot CoT

Prompt suffix: "Let's think step by step"

Inference settings:
- Temperature: 0.3 (enable sampling)
- do_sample: True
- max_new_tokens: 128 (allow reasoning)

#### CoT with SFT

Training data format:
```
Report: {report}
Reasoning: {reasoning_steps}
Classification: {label}
```

**Note**: Due to deterministic nature of CoT reasoning (temp≈0 in our setup), only descriptive statistics reported (no significance testing).

## Statistical Analysis

### Design Principles

1. **Non-parametric tests**: No distribution assumptions
2. **Multiple comparison correction**: Control family-wise error rate
3. **Effect size reporting**: Practical significance
4. **Seed-based replication**: 3-5 seeds per configuration

### Tests Performed

#### 1. Kruskal-Wallis H-test

**Purpose**: Multi-group comparison (≥3 groups)

**Null hypothesis**: All groups have same distribution

**Application**: Compare multiple loss functions or models

#### 2. Mann-Whitney U test

**Purpose**: Pairwise comparison (2 groups)

**Null hypothesis**: Two groups have same distribution

**Application**: Post-hoc pairwise comparisons after significant Kruskal-Wallis

#### 3. Cliff's Delta

**Purpose**: Non-parametric effect size

**Interpretation**:
- |δ| < 0.147: Negligible
- 0.147 ≤ |δ| < 0.33: Small
- 0.33 ≤ |δ| < 0.474: Medium
- |δ| ≥ 0.474: Large

### Multiple Testing Correction

**Methods**:
1. **Bonferroni**: α_corrected = α / n_comparisons (conservative)
2. **Holm**: Step-down procedure (less conservative)
3. **FDR (Benjamini-Hochberg)**: Control false discovery rate

**Default**: Holm method (good balance)

### Statistical Decision Criteria

#### When to Apply Significance Testing

✅ **Apply tests for**:
- Encoder loss function comparisons
- Encoder data augmentation effects
- Decoder SFT vs non-SFT comparisons
- Same architecture across different datasets

❌ **Descriptive statistics only for**:
- ICL (high variance due to demonstration selection)
- CoT (deterministic, std≈0)
- Cross-architecture comparisons (encoder vs decoder)

#### Significance Levels

- **Standard**: α = 0.05
- **Strict**: α = 0.01 (for critical comparisons)

### Reporting Standards

For each comparison, report:
1. Median and IQR (primary statistics)
2. Mean ± SD (for context)
3. Test statistic and p-value
4. Corrected p-value (if multiple comparisons)
5. Cliff's delta and interpretation
6. Sample size (number of seeds)

## Reproducibility

### Fixed Elements

1. **Random Seeds**: [12, 22, 32, 42, 52]
2. **Data Split**: Same train/val/test partition
3. **Model Versions**: Specified in requirements.txt
4. **Hyperparameters**: All stored in config files

### Variable Elements

1. **GPU Type**: Results may vary slightly across different GPUs
2. **Library Versions**: Minor differences in PyTorch/Transformers versions
3. **System Architecture**: CPU/memory may affect speed but not results

### Reproducibility Checklist

- [ ] Use provided random seeds
- [ ] Same data preprocessing steps
- [ ] Same model versions (HuggingFace model IDs)
- [ ] Same hyperparameters (from config files)
- [ ] Same evaluation metrics (exact implementations)
- [ ] Multiple runs (≥3 seeds) for statistical validity

### Expected Variability

**Encoder models**:
- F1 score std dev: ±0.02-0.05 across seeds
- Larger variance under severe imbalance

**Decoder ICL**:
- F1 score std dev: ±0.05-0.15 (higher due to demonstration sampling)
- More stable with larger number of demonstrations

**Decoder SFT**:
- F1 score std dev: ±0.02-0.04 across seeds
- Similar stability to encoder models

## Computational Resources

### Typical Runtimes

| Experiment | GPU | Time | Notes |
|------------|-----|------|-------|
| Encoder training (1 seed) | RTX 3090 | 10-15 min | Per loss function |
| ICL evaluation (1 pattern) | RTX 3090 | 5-10 min | 200 test samples |
| Decoder SFT | A100 40GB | 2-3 hours | 3 epochs |
| Full experiment suite | - | 2-3 days | Parallelizable |

### Memory Requirements

| Task | GPU VRAM | System RAM |
|------|----------|------------|
| Encoder training | 8GB | 16GB |
| ICL inference (8B) | 16GB | 32GB |
| Decoder SFT (8B, QLoRA) | 24GB | 64GB |

### Optimization Strategies

1. **Gradient Checkpointing**: Reduce memory, increase time
2. **Mixed Precision**: FP16/BF16 for speed
3. **Batch Size Tuning**: Trade-off memory vs. speed
4. **Parallel Seeds**: Run multiple seeds simultaneously
5. **Model Quantization**: 4-bit/8-bit for inference

## Ethical Considerations

### Data Privacy

- All patient data anonymized
- Mock datasets for public release
- No PHI (Protected Health Information) in repository

### Clinical Validation

- Results are research findings only
- Not validated for clinical deployment
- Regulatory approval required for clinical use

### Bias and Fairness

- Single-institution data (generalization limits)
- Japanese language only
- Head CT only (not whole-body)
- No demographic stratification analysis

### Responsible AI

- Transparency: All code and methods public
- Reproducibility: Seeds and configs provided
- Limitations: Clearly stated in paper
- Intended Use: Support (not replace) radiologists

## References

See main paper for complete citations.

Key methodological references:
- Transformer architecture: Vaswani et al., 2017
- BERT: Devlin et al., 2018
- Few-shot learning: Brown et al., 2020
- Focal Loss: Lin et al., 2020
- Class-Balanced Loss: Cui et al., 2019
- QLoRA: Dettmers et al., 2023
- Chain-of-Thought: Wei et al., 2022; Kojima et al., 2022
- Statistical testing: Non-parametric methods, standard references
