# Usage Guide

This document provides detailed instructions for running experiments and reproducing the results from the paper.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Encoder Experiments](#encoder-experiments)
4. [Decoder Experiments](#decoder-experiments)

## Environment Setup

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU with 16GB+ VRAM (for decoder models)
- **RAM**: 32GB+ recommended for LLM experiments
- **Disk**: 50GB+ free space for models and results

### Installation Steps

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Data Preparation

### Using Mock Data (Default)

The repository ships with a small mock dataset (`data/raw_ct_reports.csv`, 50 samples) and optional supplementary negatives (`data/supplementary_negative_reports.csv`). Generate train/val/test splits with the preprocessing script:

```bash
cd scripts
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test_size 0.2 \
  --val_size 0.1
```

This produces:
- `data/preprocessed/train.csv` (~35 samples)
- `data/preprocessed/val.csv` (~8 samples)
- `data/preprocessed/test.csv` (~7 samples)
- Optional imbalance variants (e.g., `train_ratio_10.csv`, `train_ratio_20.csv`, `train_ratio_50.csv`) if you enable the corresponding options inside the script.

### Using Your Own Data

If you have real CT reports, preprocess them first:

```bash
python complete_fixed_preprocessing.py \
  --input your_raw_data.csv \
  --output_dir preprocessed_data \
  --test_size 0.2 \
  --val_size 0.1 \
  --random_seed 42
```

Expected input format:
```csv
text,label
"CT report text here",0
"Report with actionable finding",1
```

## Encoder Experiments

### 1. Basic Model Comparison

Compare different encoder models on the base dataset:

```bash
python multi_model_comparison.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --output_dir outputs_model_comparison
```

### 2. Loss Function Comparison

Test various loss functions with ModernBERT:

```bash
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal cb ib ib_focal \
  --seeds 12 22 32 42 52 \
  --use_juman \
  --output_dir outputs_loss_comparison
```

Options:
- `--losses`: Space-separated list of loss functions
  - `ce`: Cross Entropy
  - `focal`: Focal Loss
  - `cb`: Class-Balanced Loss
  - `ib`: Influence-Balanced Loss
  - `ib_focal`: IB-Focal combined
- `--seeds`: Random seeds for reproducibility
- `--use_juman`: Use Juman tokenizer (for Japanese)

### 3. Class Imbalance Experiments

Test robustness under different imbalance ratios:

```bash
# 1:10 ratio (10% positive)
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train_ratio_10.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --output_dir outputs_ratio_10

# 1:20 ratio (5% positive)
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train_ratio_20.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --output_dir outputs_ratio_20

# 1:50 ratio (2% positive)
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train_ratio_50.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --output_dir outputs_ratio_50
```

## Decoder Experiments

### 1. Few-Shot In-Context Learning

Run comprehensive ICL experiments (25 patterns):

```bash
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --gpu \
  --run_patterns \
  --repeat_eval 5
```

This automatically tests:
- Number of demonstrations: 0, 1, 2, 5, 10, 15, 25
- Presentation orders: positive-first, negative-first, alternating
- Label ratios: 0:10, 1:9, 3:7, 5:5, 7:3, 9:1, 10:0

Options:
- `--gpu`: Use GPU for inference
- `--run_patterns`: Run all 25 predefined patterns
- `--repeat_eval N`: Repeat each evaluation N times with different demonstration samples
- `--enable-val-eval`: Also evaluate on validation set (for debugging)

### 2. Single ICL Configuration

Test a specific configuration:

```bash
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --num_demonstrations 10 \
  --order_strategy label0_first \
  --gpu
```

Order strategies:
- `alternating`: Alternate positive/negative examples
- `label0_first`: All negative examples first
- `label1_first`: All positive examples first

### 3. Model Comparison (ICL)

Compare different decoder models:

```bash
# Test each model with 0, 5, 10 demonstrations
for model in elyza swallow qwen gemma; do
  python run_icl_llama_val.py \
    --model_name $model \
    --num_demonstrations 10 \
    --gpu
done
```

### 4. Supervised Fine-Tuning (SFT)

Fine-tune decoder model with QLoRA:

```bash
python llama_sft_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --output_dir outputs_sft \
  --num_epochs 3 \
  --batch_size 32
```

### 5. Chain-of-Thought (CoT)

#### Zero-shot CoT:
```bash
python llama_cot_eval.py \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --gpu
```

#### CoT with SFT:
```bash
python llama_sft_cot.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --output_dir outputs_cot_sft
```

## Typical Workflow

### Complete Experiment Pipeline

```bash
#!/bin/bash
# Complete reproduction of paper experiments

# 1. Preprocess mock data (run once)
cd scripts
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test_size 0.2 \
  --val_size 0.1

# 2. Encoder experiments
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv --val_csv ../data/preprocessed/val.csv --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib --seeds 12 22 32 42 52 --use_juman

# 3. ICL experiments (all patterns)
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv --val_csv ../data/preprocessed/val.csv --test_csv ../data/preprocessed/test.csv \
  --gpu --run_patterns

# 4. SFT experiments
python llama_sft_val.py \
  --train_csv ../data/preprocessed/train.csv --val_csv ../data/preprocessed/val.csv --test_csv ../data/preprocessed/test.csv \
  --output_dir outputs_sft

echo "All experiments completed!"
```

## Output Files

### Encoder Outputs

Each encoder experiment creates:
```
outputs_*/
├── config.json                    # Experiment configuration
├── results_seed_{seed}.json       # Per-seed results
├── predictions_seed_{seed}.csv    # Predictions and probabilities
└── model_seed_{seed}/            # Saved model checkpoints
```

### Decoder Outputs

ICL experiments create:
```
icl_results/
├── pattern_{N}_seed_{seed}.json   # Results for each pattern
└── summary.csv                    # Aggregated summary
```

SFT experiments create:
```
outputs_sft/
├── adapter_config.json            # LoRA configuration
├── adapter_model.bin             # Fine-tuned weights
└── training_args.json            # Training parameters
```

## Best Practices

1. **Always use fixed seeds** for reproducibility (12, 22, 32, 42, 52)
2. **Run multiple seeds** (at least 3) for statistical robustness
3. **Monitor GPU memory** usage during training
4. **Save intermediate results** to avoid re-running expensive experiments
5. **Document experiment configurations** in config files
6. **Use version control** for tracking experimental changes

## Additional Resources

- [Experimental Design](EXPERIMENTS.md) - Detailed experimental methodology
- [GitHub Issues](https://github.com/your-repo/issues) - Report bugs or ask questions
- Paper - Full methodology and results
