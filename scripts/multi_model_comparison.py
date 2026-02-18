#!/usr/bin/env python
"""
multi_model_comparison.py – Japanese Model Selection Script

Based on loss_comparison_fixed.py, optimized for efficient model selection.
Compares 8 HuggingFace-available Japanese models to identify the best performer.

Models included:
- JMedRoBERTa (wordpiece & sentencepiece) - Medical specialized, standard tokenization
- BERT Large Japanese v2 - 24 layers, 1024 hidden, Unidic tokenization  
- DeBERTa v3 Base Japanese - No preprocessing required, built-in tokenization
- Waseda RoBERTa Large - Manual Juman++ preprocessing (like loss_comparison_fixed.py)
- DeBERTa v2 Large Japanese - Manual Juman++ preprocessing (like loss_comparison_fixed.py)
- ModernBERT-ja-130m - Latest architecture with 8K context support
- ModernBERT-ja-310m - Latest architecture with 8K context support

Features:
- VRAM 11G compatible settings with unified batch configuration
- All models use batch_size=4, gradient_accumulation_steps=8 (effective batch_size=32)
- Early Stopping enabled (patience=5) with max 20 epochs for optimal convergence
- Follows HuggingFace pretraining tokenization for each model
- RoBERTa Large and DeBERTa v2 Large require manual Juman++ preprocessing
- Identical training/inference pipeline as loss_comparison_fixed.py
- Supports all loss functions (CE, Focal, CB, IB, etc.)
- Multiple seeds for statistical significance
- Automatic result aggregation and comparison
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

# ---------- Model Configurations ----------
MODEL_CONFIGS = {
    "jmedroberta-wordpiece": {
        "model_name": "alabnii/jmedroberta-base-manbyo-wordpiece",
        "use_juman": False,
        "use_fullwidth": True,
        "batch_size": 4,
        "tokenizer_kwargs": {
            "mecab_kwargs": {
                "mecab_option": "MANBYO_201907_Dic-utf8.dic"
            }
        },
        "description": "JMedRoBERTa Base Wordpiece + Medical Dictionary"
    },
    "jmedroberta-sentencepiece": {
        "model_name": "alabnii/jmedroberta-base-sentencepiece",
        "use_juman": False,
        "use_fullwidth": True,
        "batch_size": 4,
        "description": "JMedRoBERTa Base SentencePiece"
    },
    "bert-large-japanese": {
        "model_name": "tohoku-nlp/bert-large-japanese-v2",
        "use_juman": False,  # Uses Unidic dictionary (MeCab-like)
        "use_fullwidth": False,
        "batch_size": 4,
        "description": "BERT Large Japanese v2 (24 layers, 1024 hidden, Unidic tokenization)"
    },
    "deberta-v3-base-japanese": {
        "model_name": "ku-nlp/deberta-v3-base-japanese",
        "use_juman": False,  # No pre-segmentation required for v3
        "use_fullwidth": False,
        "batch_size": 4,
        "description": "DeBERTa v3 Base Japanese (No Juman++ preprocessing required)"
    },
    "roberta-large-japanese": {
        "model_name": "nlp-waseda/roberta-large-japanese",
        "use_juman": True,  # Manual Juman++ preprocessing required
        "use_fullwidth": False,
        "batch_size": 4,
        "description": "Waseda RoBERTa Large (Manual Juman++ preprocessing required)"
    },
    "deberta-v2-large-japanese": {
        "model_name": "ku-nlp/deberta-v2-large-japanese",
        "use_juman": True,  # Requires manual Juman++ preprocessing
        "use_fullwidth": False,
        "batch_size": 4,
        "description": "DeBERTa v2 Large Japanese (Manual Juman++ preprocessing required)"
    },
    "modernbert-ja-130m": {
        "model_name": "sbintuitions/modernbert-ja-130m",
        "use_juman": False,  # Uses modern tokenization approach
        "use_fullwidth": False,
        "batch_size": 4,
        "description": "ModernBERT Japanese 130M (Latest Architecture, 8K context)"
    },
    "modernbert-ja-310m": {
        "model_name": "sbintuitions/modernbert-ja-310m",
        "use_juman": False,  # Uses modern tokenization approach
        "use_fullwidth": False,
        "batch_size": 4,
        "description": "ModernBERT Japanese 310M (Latest Architecture, 8K context)"
    }
}

# ---------- Juman helpers (unchanged from original) ----------
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

# ---------- Losses (unchanged from original) ----------
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
            self.target_criterion = FocalLoss(loss_params.get("gamma", 2.0), class_weights=class_weights)
        elif loss_name == "cbloss":
            self.target_criterion = CBLoss(class_counts, beta=loss_params.get("beta", 0.999),
                                   gamma=loss_params.get("gamma", 0.0))
        elif loss_name == "ib":
            self.target_criterion = IBLoss(weight=class_weights,
                                  alpha=loss_params.get("alpha", 100.0))
        elif loss_name == "ibfocal":
            self.target_criterion = IBLoss(weight=class_weights,
                                   alpha=loss_params.get("alpha", 50.0),
                                   gamma=loss_params.get("gamma", 2.0))
        elif loss_name == "ibcb":
            self.target_criterion = IBLoss(alpha=loss_params.get("alpha", 100.0),
                                   cb_counts=class_counts,
                                   cb_beta=loss_params.get("cb_beta", 0.999))
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Current criterion starts with CE
        self.criterion = self.ce_criterion
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None,):
        labels = inputs.pop("labels")
        
        # Determine current epoch from global step
        current_epoch = self.state.epoch if hasattr(self.state, 'epoch') and self.state.epoch is not None else 0
        
        # Switch loss function based on epoch (only for IB losses)
        if current_epoch < 5 and self.loss_name.startswith("ib"):
            current_criterion = self.ce_criterion
            use_ib_loss = False
        else:
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

# ---------- Metrics (unchanged from original) ----------
def compute_metrics_detailed(eval_pred: Union[EvalPrediction, tuple]):
    """Compute detailed metrics for binary classification"""
    # ── unpack ───────────────────────────────────────────────────────────
    if isinstance(eval_pred, EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred

    # ── make numpy & squeeze/reshape ─────────────────────────────────────
    if isinstance(logits, (tuple, list)) and not isinstance(logits, np.ndarray):
        logits = logits[0]

    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if logits.ndim == 1:
        logits = logits[:, None]

    # ── probability & prediction ────────────────────────────────────────
    t_logits = torch.from_numpy(logits)

    if t_logits.size(1) == 1:
        probs = torch.sigmoid(t_logits).squeeze(1).numpy()
    else:
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

    # ── macro and weighted metrics ──────────────────────────────────────
    metrics.update(
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0),
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0),
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0),
        precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0),
        recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0),
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0),
    )

    # ── class-specific metrics ──────────────────────────────────────────
    if len(np.unique(labels)) == 2:
        class_counts = np.bincount(labels.astype(int))
        minority_class = np.argmin(class_counts)
        majority_class = 1 - minority_class
        
        metrics.update(
            precision_minority = precision_score(labels, preds, pos_label=minority_class, zero_division=0),
            recall_minority = recall_score(labels, preds, pos_label=minority_class, zero_division=0),
            f1_minority = f1_score(labels, preds, pos_label=minority_class, zero_division=0),
            precision_majority = precision_score(labels, preds, pos_label=majority_class, zero_division=0),
            recall_majority = recall_score(labels, preds, pos_label=majority_class, zero_division=0),
            f1_majority = f1_score(labels, preds, pos_label=majority_class, zero_division=0),
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
                geometric_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) if (tp + fn) > 0 and (tn + fp) > 0 else 0.0,
                youdens_j = (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp + fn) > 0 and (tn + fp) > 0 else 0.0,
            )
        else:
            metrics.update(
                true_negatives = 0, false_positives = 0, false_negatives = 0, true_positives = 0,
                specificity = 0.0, sensitivity = 0.0, mcc = 0.0, geometric_mean = 0.0, youdens_j = 0.0,
            )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"]  = float("nan")
        
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

# ---------- Utils (unchanged from original) ----------
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
        beta = 0.999
        samples_per_class = np.bincount(labels)
        effective_num = (1 - np.power(beta, samples_per_class)) / (1 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(weights)
    elif method == "sqrt":
        samples_per_class = np.bincount(labels)
        weights = 1.0 / np.sqrt(samples_per_class)
        weights = weights / weights.sum() * len(weights)
    else:
        weights = np.ones(len(np.unique(labels)))
    
    return weights

def sanitize_logits(logits):
    """Returns a well-formed numpy array shaped (N, C) or (N, 1)."""
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if logits.ndim == 1:
        logits = logits[:, None]

    return logits

def plot_model_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """Visualize multi-model comparison results"""
    plt.style.use('default')  # Use default instead of seaborn-v0_8-darkgrid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
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
        
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Model")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["model"], rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        
        # Add value labels
        for j, (val, err) in enumerate(zip(y, yerr)):
            ax.text(j, val + err + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

# ---------- Main Function ----------
def run_single_experiment(model_config: dict, loss_name: str, loss_params: dict, 
                         train_ds, val_ds, test_ds, collator, class_counts, class_weights,
                         args, seed: int, run_dir: Path):
    """Run a single experiment for one model-loss-seed combination"""
    
    ensure_reproducibility(seed)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_config["model_name"], num_labels=2)
    
    # Use model-specific batch size if available
    model_batch_size = model_config.get("batch_size", args.batch_size)
    
    # Calculate training steps with model-specific batch size
    num_training_steps = len(train_ds) // (model_batch_size * args.gradient_accumulation_steps) * args.epochs * 2
    
    tr_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=model_batch_size,
        per_device_eval_batch_size=model_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs * 2,  # Allow more for early stopping
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_dir=str(run_dir / "logs"),
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
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
    
    # Train
    train_result = trainer.train()
    
    # Test evaluation
    test_result = trainer.predict(test_ds)
    test_metrics = compute_metrics_detailed((test_result.predictions, np.array(test_ds["label"])))
    
    # Save results
    results_dict = {
        "test_metrics": test_metrics,
        "training_history": trainer.state.log_history,
        "best_epoch": trainer.state.best_model_checkpoint,
        "total_steps": trainer.state.global_step,
        "stopped_early": trainer.state.global_step < num_training_steps,
        "model_config": model_config,
        "loss_name": loss_name,
        "loss_params": loss_params,
        "args": vars(args),
    }
    
    # Save predictions
    logits_clean = sanitize_logits(test_result.predictions)
    np.save(run_dir / "test_logits.npy", logits_clean)
    np.save(run_dir / "test_labels.npy", test_result.label_ids)
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    # Clear memory
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if args.aggressive_memory_cleanup:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser("Multi-Model Japanese NLP Comparison")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--text_column", default="sentence")
    parser.add_argument("--label_column", default="label")
    
    # Model selection
    parser.add_argument("--models", nargs="*", 
                        default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Models to compare")
    
    # Training parameters (VRAM 11G optimized)
    parser.add_argument("--losses", nargs="*", default=["ce"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[12, 22, 32])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (unified across all models)")  # VRAM optimized
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps (unified for all models)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # Loss-specific parameters
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--cbloss_beta", type=float, default=0.999)
    parser.add_argument("--cbloss_gamma", type=float, default=0.0)
    parser.add_argument("--ib_alpha", type=float, default=100.0)
    parser.add_argument("--ibfocal_alpha", type=float, default=50.0)
    parser.add_argument("--ibfocal_gamma", type=float, default=2.0)
    parser.add_argument("--ibcb_alpha", type=float, default=100.0)
    parser.add_argument("--ibcb_beta", type=float, default=0.999)
    parser.add_argument("--ibcb_gamma", type=float, default=0.0)
    
    parser.add_argument("--class_weight_method", choices=["none", "balanced", "effective", "sqrt"], default="none")
    parser.add_argument("--scheduler_type", choices=["linear", "cosine", "none"], default="linear")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--output_dir", default="multi_model_outputs")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--plot_results", action="store_true", default=True)
    parser.add_argument("--aggressive_memory_cleanup", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    
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

    print("="*70)
    print("MULTI-MODEL JAPANESE NLP COMPARISON")
    print("="*70)
    print(f"Models: {args.models}")
    print(f"Loss functions: {args.losses}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(args.models)} × {len(args.losses)} × {len(args.seeds)} = {len(args.models) * len(args.losses) * len(args.seeds)}")
    print("="*70)

    all_results = []
    all_histories = {}
    best_run_dirs = {}
    
    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            print(f"Warning: Unknown model {model_key}, skipping...")
            continue
            
        model_config = MODEL_CONFIGS[model_key]
        print(f"\n" + "="*50)
        print(f"EVALUATING MODEL: {model_key}")
        print(f"HuggingFace: {model_config['model_name']}")
        print(f"Description: {model_config['description']}")
        print("="*50)
        
        # Load tokenizer for this model
        try:
            tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})
            tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], **tokenizer_kwargs)
        except Exception as e:
            print(f"Error loading tokenizer for {model_key}: {e}")
            continue
        
        # Tokenize datasets with appropriate settings
        use_juman = model_config["use_juman"]
        use_fullwidth = model_config.get("use_fullwidth", False)
        
        if use_juman and Juman is None:
            print(f"Warning: Juman++ not available, using standard tokenization for {model_key}")
            use_juman = False
        
        if use_fullwidth and mojimoji is None:
            print(f"Warning: mojimoji not available, skipping full-width conversion for {model_key}")
            use_fullwidth = False
        
        train_ds = tokenize(to_ds(train_df, args.text_column), tokenizer, args.text_column, use_juman, use_fullwidth)
        val_ds = tokenize(to_ds(val_df, args.text_column), tokenizer, args.text_column, use_juman, use_fullwidth)
        test_ds = tokenize(to_ds(test_df, args.text_column), tokenizer, args.text_column, use_juman, use_fullwidth)
        collator = DataCollatorWithPadding(tokenizer)
        
        # Run experiments for each loss function
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
            best_f1_for_config = -1.0
            best_run_dir_for_config = None
            
            for s in args.seeds:
                # Include class_weight_method in directory name if not default
                weight_suffix = "" if args.class_weight_method == "none" else f"_cw{args.class_weight_method}"
                run_dir = out_root / f"{model_key}_{loss_name}_s{s}{weight_suffix}"
                run_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Running: {model_key} + {loss_name} + seed{s}")
                
                try:
                    test_metrics = run_single_experiment(
                        model_config, loss_name, loss_params,
                        train_ds, val_ds, test_ds, collator, 
                        class_counts, class_weights,
                        args, s, run_dir
                    )
                    
                    each_seed.append(test_metrics)
                    
                    # Track best run
                    current_f1 = test_metrics['f1']
                    if current_f1 > best_f1_for_config:
                        best_f1_for_config = current_f1
                        best_run_dir_for_config = run_dir
                    
                    print(f"  Results: F1={test_metrics['f1']:.4f}, Acc={test_metrics['accuracy']:.4f}, AUC={test_metrics['roc_auc']:.4f}")
                    
                except Exception as e:
                    print(f"  Error in experiment: {e}")
                    continue
            
            if each_seed:
                # Aggregate results for this model-loss combination
                mean = {k: float(np.mean([m[k] for m in each_seed if not np.isnan(m[k])])) 
                        for k in each_seed[0]}
                std = {f"{k}_sd": float(np.std([m[k] for m in each_seed if not np.isnan(m[k])], ddof=1)) 
                       for k in each_seed[0]}
                
                result_entry = dict(
                    model=model_key,
                    loss=loss_name,
                    model_name=model_config["model_name"],
                    description=model_config["description"],
                    **loss_params,
                    **mean,
                    **std
                )
                all_results.append(result_entry)
                
                if best_run_dir_for_config:
                    best_run_dirs[f"{model_key}_{loss_name}"] = best_run_dir_for_config

    # Save comprehensive results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(out_root / 'comprehensive_results.csv', index=False)
        
        # Generate model ranking
        model_ranking = results_df.groupby('model')['f1'].agg(['mean', 'std', 'max']).round(4)
        model_ranking = model_ranking.sort_values('mean', ascending=False)
        
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("="*70)
        
        # Best overall configuration
        best_overall = results_df.loc[results_df['f1'].idxmax()]
        print(f"\n🏆 Best Overall: {best_overall['model']} + {best_overall['loss']}")
        print(f"   F1: {best_overall['f1']:.4f} ± {best_overall['f1_sd']:.4f}")
        print(f"   Model: {best_overall['model_name']}")
        
        # Top configurations
        print(f"\n📊 Top 10 Configurations by F1 Score:")
        print("-" * 70)
        top_configs = results_df.nlargest(10, 'f1')[['model', 'loss', 'f1', 'f1_sd', 'accuracy', 'roc_auc']]
        print(top_configs.to_string(index=False, float_format='%.4f'))
        
        # Model performance summary
        print(f"\n🤖 Model Performance Summary (Mean F1 Score):")
        print("-" * 50)
        print(model_ranking.to_string())
        
        # Plot results if requested
        if args.plot_results:
            print(f"\n📈 Generating visualization plots...")
            plot_model_comparison(results_df, out_root)
            print("✓ Plots generated successfully")
        
        # Save summary
        summary_dict = {
            "best_overall": {
                "model": best_overall['model'],
                "loss": best_overall['loss'],
                "f1_mean": float(best_overall['f1']),
                "f1_std": float(best_overall['f1_sd']),
                "model_name": best_overall['model_name']
            },
            "model_ranking": model_ranking.to_dict(),
            "total_experiments": len(all_results),
            "experiment_config": vars(args)
        }
        
        with open(out_root / "comprehensive_summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        
        print("="*70)
        print("✅ ANALYSIS COMPLETE")
        print(f"📁 Results saved to: {out_root}")
        print(f"📊 Key files:")
        print(f"   comprehensive_results.csv - All experimental results")
        print(f"   comprehensive_summary.json - Statistical summary")
        if args.plot_results:
            print(f"   model_comparison.png - Comparison plots")
        print("="*70)
        
        # Cleanup if requested
        if args.cleanup and best_run_dirs:
            print(f"\nCleaning up non-best runs...")
            import shutil
            for config_name, best_dir in best_run_dirs.items():
                print(f"  Best for {config_name}: {best_dir}")
            
            for model_key in args.models:
                for loss_name in args.losses:
                    best_dir_for_config = best_run_dirs.get(f"{model_key}_{loss_name}")
                    for s in args.seeds:
                        weight_suffix = "" if args.class_weight_method == "none" else f"_cw{args.class_weight_method}"
                        run_dir_to_check = out_root / f"{model_key}_{loss_name}_s{s}{weight_suffix}"
                        if run_dir_to_check.exists() and run_dir_to_check != best_dir_for_config:
                            print(f"  Removing: {run_dir_to_check}")
                            shutil.rmtree(run_dir_to_check)
    else:
        print("❌ No successful experiments completed.")

if __name__ == "__main__":
    main()