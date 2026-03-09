# CT Report Classification: Encoder vs Decoder Comparison

Implementation code for the paper:

**Comparative Evaluation of Encoder- and Decoder-Based Models for Actionable Findings in CT Reports**

- Paper URL: https://doi.org/10.1007/s10278-026-01888-1

## Overview

This repository compares encoder-based and decoder-based approaches for classifying actionable findings in CT reports, including class-imbalance settings.

## Repository Structure

```text
ct_report_classification_repo/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── data/
├── scripts/
└── docs/
    ├── USAGE.md
    └── EXPERIMENTS.md
```

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
cd scripts
python complete_fixed_preprocessing.py \
  --input ../data/raw_ct_reports.csv \
  --output_dir ../data/preprocessed \
  --test-size 0.15 \
  --val-size 0.15
```

### 3. Run Encoder Baseline

```bash
python loss_comparison_fixed.py \
  --train_csv ../data/preprocessed/train.csv \
  --val_csv ../data/preprocessed/val.csv \
  --test_csv ../data/preprocessed/test.csv \
  --losses ce \
  --seeds 42 \
  --use_juman
```

## Documentation

- Quick start details: [QUICKSTART.md](QUICKSTART.md)
- Command-level usage: [docs/USAGE.md](docs/USAGE.md)
- Experimental methodology: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

## Notes

- Included dataset is a small mock dataset for code demonstration.
- This repository is for research and educational use; not for clinical deployment as-is.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Please open an issue or contact: fyusuke@belle.shiga-med.ac.jp
