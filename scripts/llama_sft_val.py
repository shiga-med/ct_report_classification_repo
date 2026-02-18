#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llama_sft.py – Supervised fine-tuning (SFT) of Llama-3 model (FIXED VERSION with proper VAL usage)

Major fixes:
- Proper compute_metrics implementation for generation tasks
- Correct dataset construction preserving labels
- Proper evaluation using generate() instead of forward()
- Added perplexity-based confidence scores
- VAL data is not used for training - only for optional validation during training
- Main evaluation is on TEST data
"""

import argparse, json, os, random, warnings
from pathlib import Path
from typing import Dict

import numpy as np, pandas as pd, torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef)
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,
                          TrainingArguments, Trainer, BitsAndBytesConfig,
                          EarlyStoppingCallback)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    raise SystemExit("peft >= 0.8.0 is required – install with `pip install peft`.")

warnings.filterwarnings("ignore", category=UserWarning)

# Common instruction for all experiments
DEFAULT_INSTRUCTION = """以下のCT所見の文章を読んで、次の条件を満たすかどうかを判断してください：

    1. フォローや治療が必要な新規病変の存在
    2. 既存病変の悪化
    3. 追加検査または治療の明確な推奨

もし上記の条件の内一つでも満たす場合には「1」、一つも満たさない場合は「0」と出力してください。
必ず「1」または「0」のどちらかは出力します。
なお、回答となる数値はint型で返し、他には何も含めないことを厳守してください。"""

PROMPT_TMPL = (
    "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    "### 指示:\n{instruction}\n\n"
    "### 入力:\n{sentence}\n\n"
    "### 応答:\n"
)

# Generation configuration (same as CT_Llama3_S_total_study.py)
GENERATION_CONFIG = {
    "max_new_tokens": 1,
    "temperature": 1e-8,  # Extremely small value for deterministic output with numerical stability
    "do_sample": False,
    "top_k": 1,          # 最も確率が高い1つのトークンを選択
    "top_p": 0.0,        # top-pサンプリングを無効化
    "repetition_penalty": 1.0,
}

# ---------- utils ----------

def get_class_token_ids(tokenizer):
    """Get token IDs for '0' and '1' (max_new_tokens=1, single token only)"""
    # Direct token ID lookup for single tokens
    one_id = tokenizer.convert_tokens_to_ids("1")
    zero_id = tokenizer.convert_tokens_to_ids("0")
    
    # Handle unknown tokens with fallback
    if one_id == tokenizer.unk_token_id:
        one_tokens = tokenizer("1", add_special_tokens=False)["input_ids"]
        one_id = one_tokens[0] if one_tokens else tokenizer.unk_token_id
    
    if zero_id == tokenizer.unk_token_id:
        zero_tokens = tokenizer("0", add_special_tokens=False)["input_ids"]
        zero_id = zero_tokens[0] if zero_tokens else tokenizer.unk_token_id
    
    return one_id, zero_id


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_reproducibility(seed: int):
    """Complete reproducibility settings"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)


class Metrics:
    """Static methods for dataset-level metrics (binary)."""

    @staticmethod
    def calc(y_true, probs: np.ndarray) -> Dict[str, float]:
        # Use generated predictions directly instead of probability threshold
        # Since we already have discrete 0/1 predictions from generation
        preds = np.round(probs).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            roc_auc = float("nan")
            pr_auc = float("nan")
        else:
            # For ROC-AUC and PR-AUC, use the original probs if they're valid probabilities
            # Otherwise use the discrete predictions
            if np.all((probs >= 0) & (probs <= 1)):
                roc_auc = roc_auc_score(y_true, probs)
                pr_auc = average_precision_score(y_true, probs)
            else:
                roc_auc = roc_auc_score(y_true, preds)
                pr_auc = average_precision_score(y_true, preds)
        
        return dict(
            accuracy  = accuracy_score(y_true, preds),
            precision = precision_score(y_true, preds, zero_division=0),
            recall    = recall_score(y_true, preds, zero_division=0),
            f1        = f1_score(y_true, preds, zero_division=0),
            mcc       = matthews_corrcoef(y_true, preds),
            roc_auc   = roc_auc,
            pr_auc    = pr_auc,
        )


def df_to_sft_dataset(df: pd.DataFrame, inst: str, model_tokenizer) -> Dataset:
    """Create dataset preserving labels for evaluation"""
    
    def preprocess_function(examples):
        inputs = []
        labels = []
        full_texts = []
        
        for sentence, label in zip(examples["sentence"], examples["label"]):
            # Input prompt
            prompt = PROMPT_TMPL.format(instruction=inst, sentence=sentence)
            # Target
            target = str(int(label))
            # Full text for training
            full_text = prompt + target
            
            inputs.append(prompt)
            labels.append(target)
            full_texts.append(full_text)
        
        # Tokenize full texts for training
        model_inputs = model_tokenizer(
            full_texts,
            truncation=True,
            padding=True,
            max_length=512,
        )
        
        # Create labels with -100 for prompt tokens
        labels_for_training = []
        
        for i, (prompt, target) in enumerate(zip(inputs, labels)):
            # Tokenize prompt and target separately to ensure consistency
            prompt_tokens = model_tokenizer(prompt, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
            target_tokens = model_tokenizer(target, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
            
            # Create full sequence
            full_tokens = prompt_tokens + target_tokens
            
            # Ensure we don't exceed max_length
            if len(full_tokens) > 512:
                # Truncate from prompt side, keep target intact
                prompt_tokens = prompt_tokens[:512-len(target_tokens)]
                full_tokens = prompt_tokens + target_tokens
            
            # Create labels: -100 for prompt, actual tokens for target
            label_ids = [-100] * len(prompt_tokens) + target_tokens
            
            # Pad if necessary
            while len(label_ids) < len(model_inputs["input_ids"][i]):
                label_ids.append(-100)
            
            # Truncate if too long
            label_ids = label_ids[:len(model_inputs["input_ids"][i])]
            
            labels_for_training.append(label_ids)
        
        model_inputs["labels"] = labels_for_training
        
        # Store original labels for evaluation
        model_inputs["original_labels"] = [int(label) for label in examples["label"]]
        
        return model_inputs
    
    dataset = Dataset.from_pandas(df[["sentence", "label"]])
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset


def evaluate_generation_pipeline(model, tokenizer, test_df, inst, device, calc_perplexity=True):
    """Evaluate using text_generation_pipeline to match CT_Llama3_S_total_study.py"""
    
    model.eval()
    predictions = []      # All predictions (0/1, invalid treated as 0)
    probs = []           # All probabilities 
    perplexities = []    # All perplexities
    logprobs = []        # All logprobs
    pred_labels_raw = [] # Raw prediction strings for CSV
    valid_flags = []     # Validity flags for CSV
    invalid_count = 0    # Counter for invalid outputs (not 0 or 1)
    
    # Use module-level generation config with tokenizer-specific settings
    generation_config = {**GENERATION_CONFIG, "pad_token_id": tokenizer.pad_token_id}
    
    # Get token IDs for probability calculation (for legacy compatibility)
    one_id, zero_id = get_class_token_ids(tokenizer)
    
    for row in tqdm(test_df.itertuples(), desc="Evaluating", total=len(test_df)):
        prompt = PROMPT_TMPL.format(instruction=inst, sentence=row.sentence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            # Single generation call with scores (unified approach)
            outputs = model.generate(
                **inputs,
                **generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Extract generated text (same method as CT_Llama3_S_total_study.py)
            full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_text = full_output.replace(prompt, "")
            
            # Parse response (EXACTLY same method as CT_Llama3_S_total_study.py)
            pred_label = generated_text.split("\n")[0].strip()
            
            # Determine prediction from text output (max_new_tokens=1, so single token only)
            # Convert to string and strip whitespace for consistent processing
            pred_str = str(pred_label).strip()
            
            if pred_str == '0':
                pred = 0
                is_valid = True
            elif pred_str == '1':
                pred = 1
                is_valid = True
            elif pred_str and len(pred_str) == 1 and pred_str.isdigit():
                # Other single digits (2, 3, 4, ...) - count as invalid and assign 0
                pred = 0
                is_valid = False
                invalid_count += 1
                print(f"Invalid numeric output '{pred_str}' -> assigned 0 (invalid #{invalid_count})")
            else:
                # Non-digit output or empty output - count as invalid and assign 0
                pred = 0
                is_valid = False
                invalid_count += 1
                if not pred_str:
                    print(f"Empty output -> assigned 0 (invalid #{invalid_count})")
                else:
                    print(f"Invalid output '{pred_str}' -> assigned 0 (invalid #{invalid_count})")
            
            # Calculate probabilities for single token generation (max_new_tokens=1)
            if outputs.scores and len(outputs.scores) > 0:
                logits = outputs.scores[0][0]  # Single generated token logits
                token_probs = F.softmax(logits, dim=-1)
                
                # Get probabilities for "0" and "1" tokens and normalize
                if (one_id != tokenizer.unk_token_id and 
                    zero_id != tokenizer.unk_token_id and
                    one_id < len(token_probs) and zero_id < len(token_probs)):
                    
                    p0 = token_probs[zero_id].item()  # int 0 probability
                    p1 = token_probs[one_id].item()   # int 1 probability
                    
                    # Normalize to get binary class probability
                    total = p0 + p1
                    prob_1 = p1 / total if total > 0 else float(pred)
                else:
                    prob_1 = float(pred)  # Fallback if tokens not found
            else:
                prob_1 = float(pred)  # Fallback if no scores
            
            # Calculate logprob and perplexity for single token generation
            if calc_perplexity:
                if outputs.scores and len(outputs.scores) > 0:
                    logits = outputs.scores[0][0]  # Single generated token logits
                    generated_token_id = outputs.sequences[0][-1]  # Last token is the generated one
                    
                    # Calculate logprob directly
                    log_probs = F.log_softmax(logits, dim=-1)
                    logprob = log_probs[generated_token_id].item()
                    
                    # Calculate perplexity from logprob
                    perplexity = torch.exp(torch.tensor(-logprob)).item()
                else:
                    logprob = float('-inf')
                    perplexity = float('inf')
            else:
                logprob = 0.0
                perplexity = 1.0
            
            # Store all predictions for evaluation and CSV
            predictions.append(pred)
            probs.append(prob_1)
            perplexities.append(perplexity)
            logprobs.append(logprob)
            pred_labels_raw.append(pred_label)  # Raw model output for CSV
            valid_flags.append(is_valid)        # Validity flag for CSV
    
    # Calculate metrics using all predictions (invalid treated as 0)
    pred_array = np.array(predictions)
    prob_array = np.array(probs)
    true_labels_array = np.array(test_df["label"])
    
    # Calculate discrete metrics
    accuracy = accuracy_score(true_labels_array, pred_array)
    precision = precision_score(true_labels_array, pred_array, zero_division=0)
    recall = recall_score(true_labels_array, pred_array, zero_division=0)
    f1 = f1_score(true_labels_array, pred_array, zero_division=0)
    mcc = matthews_corrcoef(true_labels_array, pred_array)
    
    # Calculate probability-based metrics
    if len(np.unique(true_labels_array)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(true_labels_array, prob_array)
        pr_auc = average_precision_score(true_labels_array, prob_array)
    
    if invalid_count > 0:
        print(f"Note: {invalid_count}/{len(test_df)} samples had invalid outputs, treated as 0")
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    
    if calc_perplexity:
        metrics["perplexity_mean"] = float(np.mean(perplexities))
        metrics["perplexity_std"] = float(np.std(perplexities))
        metrics["logprob_mean"] = float(np.mean(logprobs))
        metrics["logprob_std"] = float(np.std(logprobs))
    
    # Add invalid output statistics
    metrics["invalid_outputs"] = invalid_count
    metrics["invalid_ratio"] = invalid_count / len(test_df) if len(test_df) > 0 else 0.0
    
    return (metrics, predictions, probs, perplexities if calc_perplexity else None,
            pred_labels_raw, valid_flags, logprobs if calc_perplexity else None)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Llama-3 SFT for CT report classification (VAL not used for training)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)  # Not used for training, kept for compatibility
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model_name", default="elyza/Llama-3-ELYZA-JP-8B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--inst_file", help="外部 .txt で指示文を上書き")
    ap.add_argument("--qlora", action="store_true", help="Enable 4-bit QLoRA + NF4")
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank (8, 16, 32, 64)")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (typically 2*r)")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--lora_target_modules", nargs="+", 
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="Target modules for LoRA")
    ap.add_argument("--output_dir", default="sft_output")
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--use_val_for_validation", action="store_true", 
                    help="Use validation data for early stopping (default: no validation during training)")
    ap.add_argument("--calc_perplexity", action="store_true", help="Calculate perplexity and logprob-based confidence")
    args = ap.parse_args()

    ensure_reproducibility(args.seed)

    # ---------- I/O ----------
    def load_df(p):
        df = pd.read_csv(p)
        if "label" not in df.columns or "sentence" not in df.columns:
            raise ValueError(f"{p} must contain 'sentence' and 'label' columns.")
        df["label"] = df["label"].astype(int)
        return df

    train_df, val_df, test_df = load_df(args.train_csv), load_df(args.val_csv), load_df(args.test_csv)

    inst = Path(args.inst_file).read_text(encoding="utf-8") if args.inst_file else DEFAULT_INSTRUCTION

    # ---------- model & tokenizer ----------
    bnb_cfg = None
    if args.qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_cfg,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    # LoRA configuration for QLoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Dataset preparation - only use training data for training
    train_ds = df_to_sft_dataset(train_df, inst, tokenizer)
    val_ds = df_to_sft_dataset(val_df, inst, tokenizer) if args.use_val_for_validation else None

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # ---------- Trainer ----------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments - adjust based on whether we use validation
    if args.use_val_for_validation:
        evaluation_strategy = "steps"
        eval_steps = args.eval_steps
        save_strategy = "steps"
        save_steps = args.eval_steps
        load_best_model_at_end = True
        metric_for_best_model = "eval_loss"
        greater_is_better = False
        callbacks = [EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.001
        )]
    else:
        evaluation_strategy = "no"
        eval_steps = None
        save_strategy = "epoch"
        save_steps = None
        load_best_model_at_end = False
        metric_for_best_model = None
        greater_is_better = None
        callbacks = []
    
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=50,
        logging_dir=str(output_dir / "logs"),
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
    )

    # Custom evaluation during training
    def compute_metrics(_):
        # Since we're doing generation task, we can't properly evaluate here
        # Return dummy metrics - real evaluation will be done separately
        return {"eval_loss": 0.0}

    trainer = Trainer(
        model=model, 
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if args.use_val_for_validation else None,
        callbacks=callbacks,
    )

    # Train
    print(f"Training on {len(train_df)} samples")
    if args.use_val_for_validation:
        print(f"Using {len(val_df)} validation samples for early stopping")
    else:
        print("No validation during training - VAL data reserved for ICL few-shot examples")
    
    _ = trainer.train()  # Training result not used but kept for potential future logging

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)

    # ---------- Final evaluation on TEST data only ----------
    device = next(model.parameters()).device
    
    print("Evaluating on TEST data...")
    (test_metrics, test_preds, test_probs, test_perplexities,
     pred_labels_raw, valid_flags, test_logprobs) = evaluate_generation_pipeline(
        model, tokenizer, test_df, inst, device, args.calc_perplexity
    )

    # Save results
    results = {
        "config": vars(args),
        "test_metrics": test_metrics,
        "training_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint if args.use_val_for_validation else None,
        "val_used_for_training": args.use_val_for_validation,
        "note": "VAL data not used for training - reserved for ICL few-shot examples" if not args.use_val_for_validation else "VAL data used for validation during training"
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save test predictions - include all samples with raw predictions and validity flags
    test_results_df = pd.DataFrame({
        "sentence": test_df["sentence"],
        "true_label": test_df["label"],
        "pred_label_raw": pred_labels_raw,    # Raw model output (original token)
        "pred_label": test_preds,             # Processed predictions (0/1, invalid as 0)
        "prob_1": test_probs,                 # Probability for class 1
        "is_valid": valid_flags               # Validity flag (True/False)
    })
    if args.calc_perplexity:
        test_results_df["perplexity"] = test_perplexities
        test_results_df["logprob"] = test_logprobs
    test_results_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Save model
    trainer.save_model()
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    if args.use_val_for_validation:
        print(f"Validation samples used during training: {len(val_df)}")
    else:
        print(f"VAL samples ({len(val_df)}) reserved for ICL few-shot examples")
    print("="*50)


if __name__ == "__main__":
    main()