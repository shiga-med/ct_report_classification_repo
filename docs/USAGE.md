# Usage Guide (Minimal)

This document keeps only the essential commands.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data Preparation

```bash
cd scripts
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test-size 0.15 \
  --val-size 0.15
```

Expected files:
- `../data/preprocessed/train.csv`
- `../data/preprocessed/val.csv`
- `../data/preprocessed/test.csv`

## Encoder

### Single baseline run

```bash
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce \
  --seeds 42 \
  --use_juman
```

### Multi-loss / multi-seed run

```bash
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce focal ib \
  --seeds 12 22 32 42 52
```

## Decoder

### ICL all patterns

```bash
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --gpu \
  --run_patterns
```

### ICL single configuration

```bash
python run_icl_llama_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --n_fewshot 10 \
  --order-strategy label0_first \
  --gpu
```

### SFT / CoT

```bash
python llama_sft_val.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --output_dir outputs_sft

python llama_cot_eval.py \
  --test_csv ../data/preprocessed/test.csv \
  --gpu
```

## Output Locations

- Encoder: `scripts/outputs_*/`
- ICL: `scripts/icl_results/`
- SFT: `scripts/outputs_sft/`

## Notes

- Included data is mock/demo data.
- For full options, run each script with `--help`.
