# Quick Start Guide

Get started with the CT Report Classification experiments in 5 minutes.

## Prerequisites

- Python 3.8+
- pip package manager
- (Optional) CUDA-compatible GPU for decoder experiments

## Installation (3 steps)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ct_report_classification_repo

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Quick Test (5 minutes)

### Step 1: Prepare Data

The mock dataset (`data/raw_ct_reports.csv`) is already included in the repository.

```bash
cd scripts

# Preprocess the data
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test_size 0.2 \
  --val_size 0.1

# (Optional) Keep train size while reducing positive rate
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed_balanced \
  --supplementary-data ../data/supplementary_negative_reports.csv \
  --maintain-size
```

**Output**:
- `data/raw_ct_reports.csv` (50 raw samples)
- `data/preprocessed/train.csv` (~35 samples)
- `data/preprocessed/val.csv` (~8 samples)
- `data/preprocessed/test.csv` (~7 samples)
- Optional run stores rebalanced splits under `data/preprocessed_balanced/` using `data/supplementary_negative_reports.csv` (150 additional label 0 samples)

**Note**: This is a small demo dataset. Adjust preprocessing split ratios for best results with small data.

### Step 2: Run Basic Encoder Experiment

```bash
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce \
  --seeds 42 \
  --use_juman
```

**Expected time**: 5-10 minutes on GPU, 15-20 minutes on CPU

**Output**:
- Training logs printed to console
- Results saved to `outputs_ce/`

### Step 3: View Results

```bash
# Results are saved as JSON
cat outputs_ce/results_seed_42.json
```

Example output:
```json
{
  "test_accuracy": 0.92,
  "test_precision": 0.85,
  "test_recall": 0.78,
  "test_f1": 0.81,
  "test_auroc": 0.93,
  "test_auprc": 0.87
}
```

## What's Next?

### Run Complete Encoder Experiments

```bash
# Compare multiple loss functions
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --seeds 12 22 32 42 52 \
  --use_juman
```

### Run Decoder Few-Shot Experiments (GPU recommended)

```bash
# Run all 25 ICL patterns
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --gpu \
  --run_patterns
```

**Note**: This requires ~16GB GPU VRAM and takes 1-2 hours.

## Common Issues

### Issue 1: ModuleNotFoundError

```bash
# Make sure you activated the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 2: CUDA out of memory

```bash
# Use smaller batch size
python loss_comparison_fixed.py --batch_size 16 ...

# Or use CPU
python loss_comparison_fixed.py --device cpu ...
```

### Issue 3: Tokenizer not found

```bash
# Install Juman++ for Japanese tokenization
# macOS:
brew install jumanpp

# Ubuntu/Debian:
sudo apt-get install jumanpp

# Then install Python wrapper
pip install pyknp
```

## Example Workflows

### Workflow 1: Quick Model Comparison (30 minutes)

```bash
cd scripts

# Test 3 loss functions with 1 seed
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --seeds 42

# View results
ls -lh outputs_*/results_seed_42.json
```

### Workflow 2: Class Imbalance Study (1 hour)

```bash
# First, create imbalanced datasets
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --create_imbalanced

# Test different imbalance ratios
for ratio in train train_ratio_10 train_ratio_20 train_ratio_50; do
  python loss_comparison_fixed.py \
    --train_csv ../data/preprocessed/${ratio}.csv \
    --val_csv ../data/preprocessed/val.csv \
    --test_csv ../data/preprocessed/test.csv \
    --losses ce \
    --seeds 42 \
    --output_dir outputs_${ratio}
done

# Compare results
grep -h "test_f1" outputs_*/results_seed_42.json
```

### Workflow 3: Few-Shot Learning (15 minutes)

```bash
# Test different numbers of demonstrations
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --num_demonstrations 5 \
  --gpu

# Change to 10 demonstrations
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --num_demonstrations 10 \
  --gpu
```

## Directory Structure After Quick Start

```
ct_report_classification_repo/
├── data/                          # Datasets
│   ├── raw_ct_reports.csv        # Raw data (before preprocessing)
│   ├── dataset_info.json         # Metadata
│   └── preprocessed/             # After preprocessing
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── (optional) train_ratio_*.csv
├── scripts/                       # Experiment scripts
├── outputs_ce/                    # Results from CE loss
│   ├── results_seed_42.json
│   ├── predictions_seed_42.csv
│   └── model_seed_42/
├── icl_results/                   # ICL experiment results
└── aggregated_results/            # Combined analysis
    └── unified_results.csv
```

## Getting Help

- **Detailed Usage**: See [docs/USAGE.md](docs/USAGE.md)
- **Experimental Details**: See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- **Issues**: Open a GitHub issue
- **Questions**: Contact the authors

## Next Steps

1. ✅ **You've completed the quick start!**
2. 📖 Read [USAGE.md](docs/USAGE.md) for detailed instructions
3. 🔬 Read [EXPERIMENTS.md](docs/EXPERIMENTS.md) for methodology
4. 🚀 Run the full experiment pipeline
5. 📊 Reproduce the paper results

## Tips for Success

- **Start small**: Use 1-2 seeds for quick testing, then scale to 5 seeds for publication
- **Monitor resources**: Check GPU memory and disk space regularly
- **Save configs**: Keep track of what worked with JSON config files
- **Version control**: Commit your results to git (but not large model files!)
- **Document changes**: Note any modifications to hyperparameters

Happy experimenting! 🎉
