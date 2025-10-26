# CT Report Classification: Encoder vs Decoder Comparison

This repository contains the implementation code for the research paper:

**"Evaluation of Encoder- and Decoder-Based Approaches for Classifying Actionable Findings in CT Reports"**

## Overview

This study systematically compares encoder-based (ModernBERT) and decoder-based (Llama-3-ELYZA-JP-8B) models for automatic classification of actionable findings in CT diagnostic reports, with a focus on performance under class-imbalance conditions.

### Key Findings

- **Encoder-based SFT** achieved F1=0.870 with positive rates ≥10%
- **Decoder-based few-shot prompting** maintained F1=0.717 even with limited demonstrations
- Both approaches degraded sharply when positive rate fell below 5%
- Loss function selection had less impact than dataset composition

## Repository Structure

```
ct_report_classification_repo/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # License information
├── data/                       # Mock datasets
│   ├── raw_ct_reports.csv     # Raw dataset (before preprocessing, 50 samples)
│   ├── supplementary_negative_reports.csv  # Additional label-0 samples (20 entries)
│   ├── supplementary_negative_reports.json # Metadata for supplementary negatives
│   └── dataset_info.json      # Dataset metadata
├── scripts/                    # Experimental scripts
│   ├── complete_fixed_preprocessing.py    # Data preprocessing
│   ├── losses_fixed.py                    # Custom loss functions
│   ├── loss_comparison_fixed.py           # Encoder experiments
│   ├── multi_model_comparison.py          # Model comparison
│   ├── run_icl_llama_val.py              # ICL experiments
│   ├── llama_sft_val.py                  # Decoder SFT
│   ├── llama_sft_cot.py                  # CoT-SFT experiments
│   └── llama_cot_eval.py                 # Zero-shot CoT evaluation
└── docs/                       # Documentation
    ├── USAGE.md               # Detailed usage instructions
    └── EXPERIMENTS.md         # Experimental design details
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for decoder models)
- 16GB+ RAM (32GB+ recommended for LLM experiments)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ct_report_classification_repo

# Install dependencies
pip install -r requirements.txt
```

### Prepare Mock Dataset

⚠️ **Important**: This repository uses MOCK DATA (50 samples) for quick demonstration purposes only.

The mock raw dataset (`data/raw_ct_reports.csv`) is already bundled with the repository.

```bash
cd scripts

# Preprocess the data (create train/val/test splits)
# Note: Use smaller test/val sizes for small datasets
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test_size 0.15 \
  --val_size 0.15

# Optional: maintain total train size with additional negatives
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed_balanced \
  --supplementary-data ../data/supplementary_negative_reports.csv \
  --maintain-size
```

**Note**: The small dataset (50 samples) is only for testing the code. For actual experiments matching the paper, you would need ~1000 samples.

### Run Basic Experiments

#### 1. Encoder-based Classification

```bash
# Train encoder model with multiple loss functions
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --seeds 12 22 32 \
  --use_juman
```

#### 2. Decoder-based Few-shot Learning

```bash
# Run ICL experiments with various configurations
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --gpu \
  --run_patterns
```

## Experimental Design

### Models Evaluated

**Encoder Models:**
- ModernBERT-ja-310m (primary)
- DeBERTa-v3-base-japanese
- UTH-BERT-base
- MedBERT-CR-base
- Others (see paper)

**Decoder Models:**
- Llama-3-ELYZA-JP-8B (primary)
- Swallow-8B-Instruct
- Qwen2.5-7B-Instruct
- Gemma-3-4b-it
- MedGemma-4b-it
- Others (see paper)

### Loss Functions

- Cross Entropy (CE)
- Focal Loss
- Class-Balanced (CB) Loss
- Influence-Balanced (IB) Loss
- IB-Focal (combined)

### Dataset Configurations

| Dataset | Total Samples | Split | Positive Rate |
|---------|--------------|-------|---------------|
| Raw | 50 | Full dataset | ~14% |
| Train | ~35 | 70% | ~14% |
| Val | ~8 | 15% | ~14% |
| Test | ~7 | 15% | ~14% |

**Note**: This is a small demonstration dataset. For actual research, use larger datasets (1000+ samples).

**Optional imbalanced datasets** (created via preprocessing):
- Configurable imbalance ratios through preprocessing script

## Key Scripts

### Data Preparation

- `complete_fixed_preprocessing.py`: Text preprocessing, normalization, and train/val/test splitting

### Training & Evaluation

- `loss_comparison_fixed.py`: Encoder model training with various loss functions
- `run_icl_llama_val.py`: Few-shot in-context learning experiments
- `llama_sft_val.py`: Supervised fine-tuning for decoder models
- `llama_cot_eval.py`: Chain-of-Thought prompting evaluation

## Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
SEEDS = [12, 22, 32, 42, 52]
```

### Environment Management

This project uses standard `requirements.txt` for dependency management:

```bash
pip install -r requirements.txt
```

For more controlled environments, consider using virtual environments:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

This research was conducted to improve patient safety by preventing missed findings in medical imaging reports.

## Contact

For questions or issues, please open a GitHub issue or contact: fyusuke@belle.shiga-med.ac.jp

## Disclaimer

⚠️ **Important Notes:**

1. **Mock Data Only**: The dataset in this repository contains only 50 synthetic samples created solely for demonstration purposes. They do not contain real patient data and are insufficient for actual research.

2. **Small Dataset Size**: The 50-sample dataset is only suitable for:
   - Testing the code functionality
   - Understanding the workflow
   - Quick demonstrations

   **NOT suitable for:**
   - Actual research experiments
   - Model performance evaluation
   - Publication-quality results

3. **Research Purpose**: This code is provided for research and educational purposes. Clinical deployment requires proper validation, regulatory approval, and adherence to medical device regulations.

4. **Not for Clinical Use**: This software is NOT intended for clinical diagnosis or treatment decisions without proper validation and regulatory clearance.

## References

See the main paper for complete references to:
- Transformer architectures (Vaswani et al., 2017)
- BERT and variants (Devlin et al., 2018)
- Large Language Models (Brown et al., 2020)
- Loss functions for imbalanced learning (Lin et al., 2020; Cui et al., 2019)
- Medical report classification (Nakamura et al., 2021; Wataya et al., 2024)
