#!/usr/bin/env python
"""
loss_comparison_end2end_juman.py – FIXED VERSION

Major improvements:
- Early stopping with best model selection
- Learning rate scheduling
- Optional class weighting
- Better reproducibility
- Detailed metrics and visualization
"""

import argparse, json
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, matthews_corrcoef, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding,
                          EvalPrediction, logging, EarlyStoppingCallback,
                          get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup)
logging.set_verbosity_error()

# Optional Juman++
try:
    from pyknp import Juman
except ImportError:
    Juman = None

# Full-width conversion for JMedRoBERTa
try:
    import mojimoji
except ImportError:
    mojimoji = None

# ---------- Juman helpers ----------
_juman = None
def juman_segmenter():
    if Juman is None:
        raise ImportError("pyknp is not installed. Install via `pip install pyknp`.")
    return Juman()

def pre_tokenize(text: str, use_juman: bool, use_fullwidth: bool = False) -> str:
    global _juman
    
    # Apply full-width conversion for JMedRoBERTa models
    if use_fullwidth and mojimoji is not None:
        # Convert to full-width (keep ASCII as half-width, convert kana to full-width)
        text = mojimoji.han_to_zen(text, digit=True, ascii=True, kana=True)
    
    if not use_juman:
        return text
    if _juman is None:
        _juman = juman_segmenter()
    try:
        return " ".join(m.midasi for m in _juman.analysis(text).mrph_list())
    except Exception as e:
        print(f"Warning: Juman++ processing failed: {e}. Using original text.")
        return text

def tokenize(ds: Dataset, tokenizer, text_col: str, use_juman: bool=False, use_fullwidth: bool=False):
    def _tok(batch):
        proc = [pre_tokenize(t, use_juman, use_fullwidth) for t in batch[text_col]]
        return tokenizer(proc, truncation=True, max_length=512)
    keep = [c for c in ds.column_names if c == "label"]
    return ds.map(_tok, batched=True, remove_columns=[c for c in ds.column_names if c not in keep])

# ---------- Losses ----------
# 追加: 共通損失関数のインポート
from losses_fixed import FocalLoss, CBLoss, IBLoss

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_name="ce", loss_params=None,
                 class_counts=None, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        loss_params = loss_params or {}
        
        
        # Get the correct device
        device = self.args.device if hasattr(self, 'args') and hasattr(self.args, 'device') else 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        
        # Convert class_weights to tensor if provided
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            class_weights = class_weights.to(device)
        
        # Store parameters for epoch-based switching
        self.loss_name = loss_name
        self.loss_params = loss_params
        self.class_counts = class_counts
        self.class_weights = class_weights
        self.device = device
        
        # Initialize Cross Entropy as the initial loss (for epochs 1-5)
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize the target loss function (for epochs 6+)
        if loss_name == "ce":
            self.target_criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_name == "focal":
            self.target_criterion = FocalLoss(loss_params.get("gamma", 1.0), class_weights=class_weights)
        elif loss_name == "cbloss":
            self.target_criterion = CBLoss(class_counts, beta=loss_params.get("beta", 0.9999),
                                   gamma=loss_params.get("gamma", 0.0))
        elif loss_name == "ib":
            self.target_criterion = IBLoss(weight=class_weights,
                                  alpha=loss_params.get("alpha", 1000.0))
        elif loss_name == "ibfocal":
            self.target_criterion = IBLoss(weight=class_weights,
                                   alpha=loss_params.get("alpha", 1000.0),
                                   gamma=loss_params.get("gamma", 1.0))
        elif loss_name == "ibcb":
            self.target_criterion = IBLoss(alpha=loss_params.get("alpha", 1000.0),
                                   cb_counts=class_counts,
                                   cb_beta=loss_params.get("cb_beta", 0.9999))
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Current criterion starts with CE
        self.criterion = self.ce_criterion
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None,):
        labels = inputs.pop("labels")
        
        # Determine current epoch from global step
        current_epoch = self.state.epoch if hasattr(self.state, 'epoch') and self.state.epoch is not None else 0
        
        # Switch loss function based on epoch (only for IB losses)
        # IB/IBFocal: Epochs 1-5 use Cross Entropy, Epochs 6+ use target loss
        # Other losses: Always use target loss
        if current_epoch < 5 and self.loss_name.startswith("ib"):
            # Use Cross Entropy for first 5 epochs (IB losses only)
            current_criterion = self.ce_criterion
            use_ib_loss = False
        else:
            # Use target loss (all losses from epoch 6+, or non-IB losses from start)
            current_criterion = self.target_criterion
            use_ib_loss = self.loss_name.startswith("ib")
        
        # Get outputs with hidden states if needed for IB loss
        outputs = model(**inputs, output_hidden_states=use_ib_loss)
        logits = outputs.logits
        
        # Ensure everything is on the same device
        if use_ib_loss:
            feats = outputs.hidden_states[-1][:, 0, :]  # CLS token
            feats = feats.to(logits.device)
            labels = labels.to(logits.device)
            loss = current_criterion(logits, labels, feats)
        else:
            labels = labels.to(logits.device)
            loss = current_criterion(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ---------- Metrics ----------
from typing import Union  # ファイル上部のインポートセクションに追加
def compute_metrics_detailed(eval_pred: Union[EvalPrediction, tuple]):
    """
    Robust metric computation for *binary* classification with enhanced metrics for imbalanced data.

    Accepts logits shaped:
        (N,)              – single logit            → sigmoid
        (N, 1)            – single logit            → sigmoid
        (N, 2)            – two-logit soft-max      → softmax[:,1]
        (N, seq_len, 2)   – take CLS token (index 0)
    Returns a dict of comprehensive metrics including minority class performance.
    """
    # ── unpack ───────────────────────────────────────────────────────────
    if isinstance(eval_pred, EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred

    # ── make numpy & squeeze/reshape ─────────────────────────────────────
    if isinstance(logits, (tuple, list)) and not isinstance(logits, np.ndarray):
        logits = logits[0]                       # 1st element (common for HF)

    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim == 3:                         # (N, seq_len, C)
        logits = logits[:, 0, :]                 # 先頭 (=CLS) だけ残す
    if logits.ndim == 1:                         # (N,) → (N,1)
        logits = logits[:, None]

    # ── probability & prediction ────────────────────────────────────────
    t_logits = torch.from_numpy(logits)

    if t_logits.size(1) == 1:                    # シングル logit → sigmoid
        probs = torch.sigmoid(t_logits).squeeze(1).numpy()
    else:                                        # 2-logit → soft-max
        probs = torch.softmax(t_logits, dim=1)[:, 1].numpy()

    preds = (probs >= 0.5).astype(int)

    # ── basic metrics ────────────────────────────────────────────────────
    metrics = dict(
        accuracy = accuracy_score(labels, preds),
        balanced_accuracy = balanced_accuracy_score(labels, preds),
        precision = precision_score(labels, preds, zero_division=0),
        recall = recall_score(labels, preds, zero_division=0),
        f1 = f1_score(labels, preds, zero_division=0),
        prob_mean = float(np.mean(probs)),
        prob_std  = float(np.std(probs)),
        prob_min  = float(np.min(probs)),
        prob_max  = float(np.max(probs)),
    )

    # ── macro and weighted metrics for imbalanced data ──────────────────
    metrics.update(
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0),
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0),
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0),
        precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0),
        recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0),
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0),
    )

    # ── class-specific metrics (minority/majority) ──────────────────────
    if len(np.unique(labels)) == 2:
        # Determine minority class (class with fewer samples)
        class_counts = np.bincount(labels.astype(int))
        minority_class = np.argmin(class_counts)
        majority_class = 1 - minority_class
        
        # Class-specific metrics
        metrics.update(
            # Minority class metrics (most important for imbalanced data)
            precision_minority = precision_score(labels, preds, pos_label=minority_class, zero_division=0),
            recall_minority = recall_score(labels, preds, pos_label=minority_class, zero_division=0),
            f1_minority = f1_score(labels, preds, pos_label=minority_class, zero_division=0),
            
            # Majority class metrics
            precision_majority = precision_score(labels, preds, pos_label=majority_class, zero_division=0),
            recall_majority = recall_score(labels, preds, pos_label=majority_class, zero_division=0),
            f1_majority = f1_score(labels, preds, pos_label=majority_class, zero_division=0),
            
            # Class distribution info
            minority_class_id = int(minority_class),
            majority_class_id = int(majority_class),
            minority_class_count = int(class_counts[minority_class]),
            majority_class_count = int(class_counts[majority_class]),
            imbalance_ratio = float(class_counts[majority_class] / max(class_counts[minority_class], 1)),
        )

        # ── AUC metrics ──────────────────────────────────────────────────
        metrics["roc_auc"] = roc_auc_score(labels, probs)
        metrics["pr_auc"]  = average_precision_score(labels, probs)
        
        # ── confusion matrix and derived metrics ────────────────────────
        if len(np.unique(preds)) == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            metrics.update(
                true_negatives = int(tn),
                false_positives = int(fp),
                false_negatives = int(fn),
                true_positives  = int(tp),
                specificity = tn / (tn + fp) if (tn + fp) else 0.0,
                sensitivity = tp / (tp + fn) if (tp + fn) else 0.0,
                mcc = matthews_corrcoef(labels, preds),
                
                # Additional imbalanced-specific metrics
                geometric_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) if (tp + fn) > 0 and (tn + fp) > 0 else 0.0,
                youdens_j = (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp + fn) > 0 and (tn + fp) > 0 else 0.0,
            )
        else:
            # Handle case where model predicts only one class
            metrics.update(
                true_negatives = 0,
                false_positives = 0, 
                false_negatives = 0,
                true_positives = 0,
                specificity = 0.0,
                sensitivity = 0.0,
                mcc = 0.0,
                geometric_mean = 0.0,
                youdens_j = 0.0,
            )
    else:  # 片方しか出てこないケース（不均衡データの極端例など）
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"]  = float("nan")
        
        # Add NaN values for missing class-specific metrics
        nan_metrics = [
            "precision_minority", "recall_minority", "f1_minority",
            "precision_majority", "recall_majority", "f1_majority",
            "minority_class_id", "majority_class_id", 
            "minority_class_count", "majority_class_count", "imbalance_ratio",
            "geometric_mean", "youdens_j"
        ]
        for metric in nan_metrics:
            metrics[metric] = float("nan")

    return metrics
  

# ---------- utils ----------
def ensure_reproducibility(seed: int):
    """Ensure complete reproducibility"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)

def to_ds(df: pd.DataFrame, text_col: str) -> Dataset:
    return Dataset.from_pandas(df[[text_col, "label"]])

def compute_class_weights_auto(labels: np.ndarray, method: str = "balanced") -> np.ndarray:
    """Compute class weights automatically"""
    if method == "balanced":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
    elif method == "effective":
        # Effective Number of Samples
        beta = 0.999
        samples_per_class = np.bincount(labels)
        effective_num = (1 - np.power(beta, samples_per_class)) / (1 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(weights)
    elif method == "sqrt":
        # Square root balancing
        samples_per_class = np.bincount(labels)
        weights = 1.0 / np.sqrt(samples_per_class)
        weights = weights / weights.sum() * len(weights)
    else:
        weights = np.ones(len(np.unique(labels)))
    
    return weights

def plot_loss_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """Visualize loss comparison results"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc"]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Bar plot with error bars
        x = range(len(summary_df))
        y = summary_df[metric]
        yerr = summary_df[f"{metric}_sd"]
        
        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
        
        # Color bars by performance
        colors = plt.cm.viridis(y / y.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel("Loss Function")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Loss Function")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["loss"], rotation=45)
        ax.set_ylim(0, 1.05)
        
        # Add value labels
        for j, (val, err) in enumerate(zip(y, yerr)):
            ax.text(j, val + err + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def sanitize_logits(logits):
    """
    Returns a well-formed numpy array shaped (N, C) or (N, 1).
    Accepts tuple/list wrappers and (N, seq_len, C) tensors.
    """
    # tuple/list → 1st element
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    logits = np.asarray(logits, dtype=np.float32)

    # (N, seq_len, C) → take CLS (= index 0)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    # (N,) → (N,1)
    if logits.ndim == 1:
        logits = logits[:, None]

    return logits

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser("End-to-End loss comparison (FIXED)")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--text_column", default="sentence")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--model_name", default="sbintuitions/modernbert-ja-310m")
    parser.add_argument("--losses", nargs="*", default=["ce","focal","cbloss","ib","ibfocal","ibcb"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[12, 22, 32])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    # Loss-specific parameters (aligned with IB-Loss experimental settings)
    parser.add_argument("--focal_gamma", type=float, default=1.0, help="Focal Loss gamma (IB-Loss experiment: 1.0)")
    parser.add_argument("--cbloss_beta", type=float, default=0.9999, help="CB Loss beta (IB-Loss experiment: 0.9999)")
    parser.add_argument("--cbloss_gamma", type=float, default=0.0, help="Focal gamma for CB-Focal variant")
    parser.add_argument("--ib_alpha", type=float, default=1000.0, help="IB Loss alpha (IB-Loss experiment: 1000.0)")
    parser.add_argument("--ibfocal_alpha", type=float, default=1000.0, help="IB-Focal alpha (IB-Loss experiment: 1000.0)")
    parser.add_argument("--ibfocal_gamma", type=float, default=1.0, help="IB-Focal gamma (IB-Loss experiment: 1.0)")
    parser.add_argument("--ibcb_alpha", type=float, default=1000.0, help="IB-CB alpha (IB-Loss experiment: 1000.0)")
    parser.add_argument("--ibcb_beta", type=float, default=0.9999, help="IB-CB beta (IB-Loss experiment: 0.9999)")
    parser.add_argument("--ibcb_gamma", type=float, default=0.0, help="IB-CB-Focal gamma (recommended: 1.0 for focal effect)")
    parser.add_argument("--class_weight_method", choices=["none", "balanced", "effective", "sqrt"], default="none")
    parser.add_argument("--scheduler_type", choices=["linear", "cosine", "none"], default="linear")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs_juman")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_juman", action="store_true")
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--aggressive_memory_cleanup", action="store_true", help="Enable aggressive CUDA memory cleanup")
    parser.add_argument("--cleanup", action="store_true", help="Remove all checkpoints except for the best one based on F1 score.")
    parser.add_argument("--save_models", action="store_true", help="Save model checkpoints and weights (disabled by default to save disk space)")
    parser.add_argument("--save_seed", type=int, help="Only save models for this specific seed (requires --save_models)")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(exist_ok=True, parents=True)

    # Load data
    def load_df(p):
        df = pd.read_csv(p)
        df[args.label_column] = df[args.label_column].astype(int)
        return df
    
    train_df, val_df, test_df = load_df(args.train_csv), load_df(args.val_csv), load_df(args.test_csv)
    
    # Compute class weights if requested
    if args.class_weight_method != "none":
        class_weights = compute_class_weights_auto(
            train_df[args.label_column].values,
            method=args.class_weight_method
        )
        print(f"Class weights ({args.class_weight_method}): {class_weights}")
    else:
        class_weights = None
    
    class_counts = train_df[args.label_column].value_counts().sort_index().tolist()
    print(f"Class distribution: {class_counts}")

    # Tokenize datasets
    tokzr = AutoTokenizer.from_pretrained(args.model_name)
    
    # Check if model is JMedRoBERTa and needs full-width conversion
    use_fullwidth = "jmedroberta" in args.model_name.lower()
    
    if use_fullwidth and mojimoji is None:
        print(f"Warning: mojimoji not available, skipping full-width conversion for JMedRoBERTa model")
        use_fullwidth = False
    
    if use_fullwidth:
        print(f"Applying full-width conversion for JMedRoBERTa model: {args.model_name}")
    
    train_ds = tokenize(to_ds(train_df, args.text_column), tokzr, args.text_column, args.use_juman, use_fullwidth)
    val_ds = tokenize(to_ds(val_df, args.text_column), tokzr, args.text_column, args.use_juman, use_fullwidth)
    test_ds = tokenize(to_ds(test_df, args.text_column), tokzr, args.text_column, args.use_juman, use_fullwidth)
    collator = DataCollatorWithPadding(tokzr)

    summary = []
    all_histories = {}
    best_run_dirs = {}  # Track best run directory for each loss function
    
    for loss_name in args.losses:
        # Prepare loss-specific parameters
        loss_params = {}
        if loss_name == "focal":
            loss_params["gamma"] = args.focal_gamma
        elif loss_name == "cbloss":
            loss_params["beta"] = args.cbloss_beta
            loss_params["gamma"] = args.cbloss_gamma
        elif loss_name == "ib":
            loss_params["alpha"] = args.ib_alpha
        elif loss_name == "ibfocal":
            loss_params["alpha"] = args.ibfocal_alpha
            loss_params["gamma"] = args.ibfocal_gamma
        elif loss_name == "ibcb":
            loss_params["alpha"] = args.ibcb_alpha
            loss_params["cb_beta"] = args.ibcb_beta
            loss_params["gamma"] = args.ibcb_gamma
        
        each_seed = []
        histories = []
        best_f1_for_loss = -1.0
        best_run_dir_for_loss = None
        
        for s in args.seeds:
            ensure_reproducibility(s)
            
            # Clear CUDA cache before each experiment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Include class_weight_method in directory name if not default
            weight_suffix = "" if args.class_weight_method == "none" else f"_cw{args.class_weight_method}"
            run_dir = out_root / f"{loss_name}_s{s}{weight_suffix}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we should save models for this seed
            should_save_models = args.save_models and (args.save_seed is None or args.save_seed == s)

            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
            
            # Calculate training steps
            num_training_steps = len(train_ds) // (args.batch_size * args.gradient_accumulation_steps) * args.epochs * 2  # *2 for early stopping
            
            # Reduce eval batch size to prevent CUDA OOM during evaluation
            eval_batch_size = max(1, args.batch_size // 2)
            
            tr_args = TrainingArguments(
                output_dir=str(run_dir),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                num_train_epochs=args.epochs * 2,  # Allow more for early stopping
                eval_strategy="steps",
                save_strategy="steps" if should_save_models else "epoch",  # Save at epoch end for early stopping even when not saving models
                save_total_limit=1 if not should_save_models else 3,  # Keep only 1 for early stopping, 3 if saving models
                load_best_model_at_end=True,  # Always load best model for proper evaluation
                metric_for_best_model="f1",
                greater_is_better=True,
                logging_strategy="steps",
                logging_dir=str(run_dir / "logs"),
                fp16=args.fp16 and torch.cuda.is_available(),
                report_to="none",
                seed=s,
                remove_unused_columns=False,  # Important for IB loss
                dataloader_pin_memory=False,  # Reduce GPU memory usage
                gradient_checkpointing=True,  # Save memory during training
            )
            
            # Initialize trainer
            trainer = CustomTrainer(
                model=model,
                args=tr_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokzr,
                data_collator=collator,
                compute_metrics=compute_metrics_detailed,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=args.early_stopping_patience,
                        early_stopping_threshold=0.001
                    )
                ],
                loss_name=loss_name,
                loss_params=loss_params,
                class_counts=class_counts,
                class_weights=class_weights,
            )
            print("Loaded class:", model.__class__)
            print("num_labels :", model.config.num_labels)

            # Custom optimizer and scheduler if requested
            if args.scheduler_type != "none":
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )
                
                num_warmup_steps = int(args.warmup_ratio * num_training_steps)
                
                if args.scheduler_type == "linear":
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps
                    )
                else:  # cosine
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps
                    )
                
                trainer.optimizer = optimizer
                trainer.lr_scheduler = scheduler
            
            # Train
            trainer.train()
            
            # Clear unnecessary memory before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Disable gradient computation for inference
            model.eval()
            with torch.no_grad():
                # Test evaluation with reduced batch size if needed
                try:
                    test_result = trainer.predict(test_ds)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("CUDA OOM during evaluation. Reducing batch size further and retrying...")
                        # Further reduce batch size for evaluation
                        original_eval_batch = trainer.args.per_device_eval_batch_size
                        trainer.args.per_device_eval_batch_size = 1
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        test_result = trainer.predict(test_ds)
                        # Restore original batch size
                        trainer.args.per_device_eval_batch_size = original_eval_batch
                    else:
                        raise e
                        
            test_metrics = compute_metrics_detailed((test_result.predictions, np.array(test_ds["label"])))
            
            # Save results
            each_seed.append(test_metrics)
            histories.append(trainer.state.log_history)
            
            # Detailed results
            results_dict = {
                "test_metrics": test_metrics,
                "training_history": trainer.state.log_history,
                "best_epoch": trainer.state.best_model_checkpoint,
                "total_steps": trainer.state.global_step,
                "stopped_early": trainer.state.global_step < num_training_steps,
                "loss_name": loss_name,
                "loss_params": loss_params,
                "args": vars(args),
            }
            
            # Save predictions (test_result already computed above)
            logits_clean = sanitize_logits(test_result.predictions)
            np.save(run_dir / "test_logits.npy", logits_clean)
            np.save(run_dir / "test_labels.npy", test_result.label_ids)
            
            # Clear CUDA cache after each experiment
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Aggressive memory cleanup (always enabled to prevent OOM)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
            
            with open(run_dir / "results.json", "w") as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)

            # Clean up temporary model files if not saving models
            if not should_save_models:
                import shutil
                for item in run_dir.iterdir():
                    if item.is_dir() and item.name.startswith("checkpoint-"):
                        shutil.rmtree(item)
                        print(f"Removed temporary checkpoint: {item}")

            # Check if this is the best model for the current loss function
            current_f1 = test_metrics['f1']
            if current_f1 > best_f1_for_loss:
                best_f1_for_loss = current_f1
                best_run_dir_for_loss = run_dir
            
        
        # Aggregate results for this loss
        mean = {k: float(np.mean([m[k] for m in each_seed if not np.isnan(m[k])])) 
                for k in each_seed[0]}
        std = {f"{k}_sd": float(np.std([m[k] for m in each_seed if not np.isnan(m[k])], ddof=1)) 
               for k in each_seed[0]}
        summary.append(dict(loss=loss_name, **loss_params, **mean, **std))
        all_histories[loss_name] = histories
        
        # Store the best run directory for this loss function
        if best_run_dir_for_loss:
            best_run_dirs[loss_name] = best_run_dir_for_loss

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_root / 'summary.csv', index=False)
    
    # Plot results if requested
    if args.plot_results:
        plot_loss_comparison(summary_df, out_root)
    
    # Save all histories
    with open(out_root / "all_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2, ensure_ascii=False)
    
    # Identify best loss function
    best_by_f1 = summary_df.loc[summary_df['f1'].idxmax()]
    best_by_auc = summary_df.loc[summary_df['roc_auc'].idxmax()]
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total runs: {len(args.losses)} losses × {len(args.seeds)} seeds = {len(args.losses) * len(args.seeds)} experiments")
    print(f"\nBest loss by F1: {best_by_f1['loss']} (F1={best_by_f1['f1']:.4f} ± {best_by_f1['f1_sd']:.4f})")
    print(f"Best loss by ROC-AUC: {best_by_auc['loss']} (AUC={best_by_auc['roc_auc']:.4f} ± {best_by_auc['roc_auc_sd']:.4f})")
    print(f"\nResults saved to: {out_root}")
    print("="*70)
    
    # Print detailed comparison table
    print("\nDetailed Results:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

    # Cleanup non-best models if requested (keep best seed for each loss function)
    if args.cleanup and args.save_models and best_run_dirs:
        print(f"\nCleaning up... Keeping the best seed for each loss function:")
        for loss_name, best_dir in best_run_dirs.items():
            print(f"  {loss_name}: {best_dir}")
        
        import shutil
        for loss_name in args.losses:
            best_dir_for_this_loss = best_run_dirs.get(loss_name)
            for s in args.seeds:
                weight_suffix = "" if args.class_weight_method == "none" else f"_cw{args.class_weight_method}"
                run_dir_to_check = out_root / f"{loss_name}_s{s}{weight_suffix}"
                if run_dir_to_check.exists() and run_dir_to_check != best_dir_for_this_loss:
                    print(f"Removing: {run_dir_to_check}")
                    shutil.rmtree(run_dir_to_check)
    elif args.cleanup and not args.save_models:
        print("\nNote: --cleanup requires --save_models to be enabled")


if __name__ == "__main__":
    main()