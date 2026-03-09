# Experimental Design (Minimal)

This file summarizes only the core experiment settings.

## Dataset

- Primary design in paper: total 1,000 reports
  - Train: 700
  - Val: 100
  - Test: 200
- Demo repository dataset is much smaller and intended for code verification.

## Encoder Experiments

### Main model families
- ModernBERT (primary)
- Japanese BERT/DeBERTa/RoBERTa variants
- Medical-domain Japanese RoBERTa variants (JMedRoBERTa)

### Losses compared
- `ce`
- `focal`
- `cbloss`
- `ib`
- `ibfocal`

### Typical training setup
- Batch size: 32
- Learning rate: 2e-5
- Max epochs: 20
- Optimizer: AdamW
- Early stopping on validation F1

### Main metrics
- F1 (primary)
- Accuracy / Precision / Recall
- AUROC / AUPRC / MCC

## Decoder Experiments

### Main model families
- Llama-3-ELYZA-JP-8B (primary)
- Swallow / Qwen / Gemma variants

### ICL factors
- # demonstrations: 0, 1, 2, 5, 10, 15, 25
- Order strategy: `alternating`, `label0_first`, `label1_first`
- Label ratio patterns (e.g., 5:5, 7:3, 9:1)
- Repeated across multiple seeds

### SFT (QLoRA)
- LoRA rank: 16
- LoRA alpha: 32
- Quantization: 4-bit (NF4)
- Typical epochs: 3

### CoT
- Zero-shot CoT and CoT-SFT evaluated as separate settings.

## Statistical Reporting

- Multi-comparison correction: Bonferroni / Holm / FDR
- Recommended default: Holm
- Report median/IQR, mean/SD, p-values, and effect sizes

## Compute (Typical)

- Encoder (1 seed): ~10-15 min (GPU class: RTX 3090)
- ICL (1 pattern): ~5-10 min
- SFT (8B QLoRA): ~2-3 h (A100 40GB class)

## Reproduction Keys

- Keep train/val/test split fixed
- Use fixed seeds (e.g., 12, 22, 32, 42, 52)
- Keep model IDs and major hyperparameters fixed across comparisons
