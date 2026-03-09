# Quick Start (Minimal)

Run a minimal end-to-end check in a few commands.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Prepare Data

```bash
cd scripts
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test-size 0.15 \
  --val-size 0.15
```

## 3. Run One Encoder Experiment

```bash
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce \
  --seeds 42 \
  --use_juman
```

## 4. Check Result

```bash
cat outputs_ce/results_seed_42.json
```

## Next

- More commands: [docs/USAGE.md](docs/USAGE.md)
- Experiment settings: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
