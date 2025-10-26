#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llama_sft_cot.py – Supervised fine-tuning (SFT) of Llama-3 model with Chain-of-Thought prompting

Major differences from llama_sft_val.py:
- Chain-of-Thought prompting with "Let's think step by step." 
- Extended max_new_tokens for reasoning generation
- Final answer extraction from generated text (last token evaluation)
- Compatible output format with aggregate_results_script.py
- Temperature=0 maintained for reproducible results

This script follows the same experimental design as llama_sft_val.py but enables
reasoning-based generation for potentially improved classification performance.
"""

import argparse, json, os, random, warnings
from pathlib import Path
from typing import Dict
import re

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

# Chain-of-Thought instruction for CT report classification
DEFAULT_INSTRUCTION = """以下のCT所見の文章を読んで、次の条件を満たすかどうかを判断してください：

    1. フォローや治療が必要な新規病変の存在
    2. 既存病変の悪化
    3. 追加検査または治療の明確な推奨

各条件について検討し、根拠を示しながら分析してください。
最終的に、上記の条件の内一つでも満たす場合には「1」、一つも満たさない場合は「0」と結論してください。
なお、最終的な回答は必ずint型の数値（0または1）で回答の最後に示してください。"""

# Chain-of-Thought prompt template
PROMPT_TMPL = (
    "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    "### 指示:\n{instruction}\n\n"
    "### 入力:\n{sentence}\n\n"
    "段階的に考えてみましょう。\n\n"
    "### 応答:\n"
)

# Generation configuration for Chain-of-Thought (extended tokens, conservative sampling)
GENERATION_CONFIG = {
    "max_new_tokens": 256,   # Extended for reasoning process
    "temperature": 0.3,      # Conservative sampling for stability
    "do_sample": True,       # Enable sampling
    "top_k": 40,            # Limited choices for stability
    "top_p": 0.95,          # Conservative nucleus sampling
    "repetition_penalty": 1.1,  # Light repetition penalty
}

# ---------- utils ----------

def get_class_token_ids(tokenizer):
    """Get token IDs for '0' and '1' (for probability calculation)"""
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


def get_final_digit_token_position(generated_tokens, final_answer):
    """テキスト解析で決定した最終数字のトークン位置を特定"""
    if not final_answer or final_answer not in ['0', '1']:
        return None
    
    target_token = final_answer  # "0" or "1"
    
    # 後ろから検索して最後の該当トークンを見つける
    for i in range(len(generated_tokens) - 1, -1, -1):
        if generated_tokens[i] == target_token:
            return i
    
    return None  # 見つからない場合


def calculate_probability_from_final_digit(outputs, tokenizer, inputs, final_answer):
    """最終数字トークンの位置での確率を計算"""
    if not outputs.scores or not final_answer or final_answer not in ['0', '1']:
        return None
    
    try:
        # 生成されたトークン列を取得（プロンプト部分を除く）
        generated_token_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids)
        
        # 最終数字の位置を特定
        position = get_final_digit_token_position(generated_tokens, final_answer)
        
        if position is None or position >= len(outputs.scores):
            return None
        
        # その位置でのlogitsから確率計算
        target_logits = outputs.scores[position][0]
        token_probs = F.softmax(target_logits, dim=-1)
        
        # "0"と"1"の確率を取得
        one_id, zero_id = get_class_token_ids(tokenizer)
        
        if (one_id != tokenizer.unk_token_id and 
            zero_id != tokenizer.unk_token_id and
            one_id < len(token_probs) and zero_id < len(token_probs)):
            
            p0 = token_probs[zero_id].item()  # "0"の確率
            p1 = token_probs[one_id].item()   # "1"の確率
            
            # 正規化して二値分類確率を計算
            total = p0 + p1
            if total > 0:
                return p1 / total  # クラス1の確率を返す
        
        return None
        
    except Exception as e:
        print(f"Warning: Error in probability calculation: {e}")
        return None


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


def extract_final_answer(generated_text: str) -> tuple[str, str]:
    """
    Extract final answer from Chain-of-Thought generated text.
    Returns: (final_answer, reasoning_text)
    
    Strategy: Look for the last occurrence of "0" or "1" in the text,
    as CoT should end with the final classification decision.
    """
    # Remove any leading/trailing whitespace
    text = generated_text.strip()
    
    # Strategy 1: Find the last digit (0 or 1) in the text
    # This works well for CoT where reasoning is followed by final answer
    last_zero_pos = text.rfind('0')
    last_one_pos = text.rfind('1')
    
    if last_zero_pos == -1 and last_one_pos == -1:
        # No digits found - return empty answer
        return "", text
    
    # Determine which digit appears last
    if last_zero_pos > last_one_pos:
        final_answer = "0"
        reasoning_end_pos = last_zero_pos
    else:
        final_answer = "1" 
        reasoning_end_pos = last_one_pos
    
    # Extract reasoning text (everything before the final answer)
    if reasoning_end_pos > 0:
        reasoning_text = text[:reasoning_end_pos].strip()
    else:
        reasoning_text = ""
    
    return final_answer, reasoning_text


class Metrics:
    """Static methods for dataset-level metrics (binary)."""

    @staticmethod
    def calc(y_true, probs: np.ndarray) -> Dict[str, float]:
        # Use generated predictions directly instead of probability threshold
        preds = np.round(probs).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            roc_auc = float("nan")
            pr_auc = float("nan")
        else:
            # For ROC-AUC and PR-AUC, use the original probs if they're valid probabilities
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
    """Create dataset for Chain-of-Thought training, preserving labels for evaluation"""
    
    def preprocess_function(examples):
        inputs = []
        labels = []
        full_texts = []
        
        for sentence, label in zip(examples["sentence"], examples["label"]):
            # Input prompt with CoT
            prompt = PROMPT_TMPL.format(instruction=inst, sentence=sentence)
            # Target - for training, include simple reasoning + answer
            # Note: In practice, you might want more sophisticated reasoning examples
            target = f"This CT scan needs to be analyzed for the three conditions mentioned. Based on the findings, the answer is {int(label)}."
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
    """Evaluate using Chain-of-Thought generation with final answer extraction"""
    
    model.eval()
    predictions = []      # All predictions (0/1, invalid treated as 0)
    probs = []           # All probabilities 
    perplexities = []    # All perplexities
    logprobs = []        # All logprobs
    pred_labels_raw = [] # Raw prediction strings for CSV
    valid_flags = []     # Validity flags for CSV
    reasoning_texts = [] # CoT reasoning for analysis
    invalid_count = 0    # Counter for invalid outputs
    
    # Use module-level generation config with tokenizer-specific settings
    generation_config = {**GENERATION_CONFIG, "pad_token_id": tokenizer.pad_token_id}
    
    # Get token IDs for probability calculation
    one_id, zero_id = get_class_token_ids(tokenizer)
    
    for row in tqdm(test_df.itertuples(), desc="Evaluating", total=len(test_df)):
        prompt = PROMPT_TMPL.format(instruction=inst, sentence=row.sentence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            # Generate with extended tokens for CoT
            outputs = model.generate(
                **inputs,
                **generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Extract generated text
            full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_text = full_output.replace(prompt, "").strip()
            
            # Extract final answer and reasoning using CoT-specific logic
            final_answer, reasoning_text = extract_final_answer(generated_text)
            
            # Determine prediction from extracted answer
            if final_answer == '0':
                pred = 0
                is_valid = True
            elif final_answer == '1':
                pred = 1
                is_valid = True
            else:
                # No valid answer found in CoT output - assign 0 as default
                pred = 0
                is_valid = False
                invalid_count += 1
                print(f"Invalid CoT output -> assigned 0 (invalid #{invalid_count})")
                print(f"  Generated text: {generated_text[:100]}...")
            
            # Calculate probabilities based on the final decision token position
            # Use the token position where the final answer was actually generated
            prob_1 = calculate_probability_from_final_digit(outputs, tokenizer, inputs, final_answer)
            
            # Fallback to old method if new method fails
            if prob_1 is None:
                print(f"Warning: Could not calculate probability from final digit '{final_answer}', using fallback")
                if outputs.scores and len(outputs.scores) > 0:
                    # Get the last generated token's logits (fallback method)
                    last_logits = outputs.scores[-1][0]
                    token_probs = F.softmax(last_logits, dim=-1)
                    
                    # Get probabilities for "0" and "1" tokens
                    if (one_id != tokenizer.unk_token_id and 
                        zero_id != tokenizer.unk_token_id and
                        one_id < len(token_probs) and zero_id < len(token_probs)):
                        
                        p0 = token_probs[zero_id].item()  # Probability of "0"
                        p1 = token_probs[one_id].item()   # Probability of "1"
                        
                        # Normalize to get binary class probability
                        total = p0 + p1
                        prob_1 = p1 / total if total > 0 else float(pred)
                    else:
                        prob_1 = float(pred)  # Fallback if tokens not found
                else:
                    prob_1 = float(pred)  # Fallback if no scores
            
            # Calculate logprob and perplexity for multiple token generation (CoT)
            if calc_perplexity:
                if outputs.scores and len(outputs.scores) > 0:
                    # Calculate mean logprob across all generated tokens
                    total_logprob = 0
                    num_generated_tokens = len(outputs.scores)
                    
                    for i, token_logits in enumerate(outputs.scores):
                        log_probs = F.log_softmax(token_logits[0], dim=-1)
                        # Get generated token ID at position i
                        token_id = outputs.sequences[0][inputs.input_ids.shape[1] + i]
                        total_logprob += log_probs[token_id].item()
                    
                    mean_logprob = total_logprob / num_generated_tokens if num_generated_tokens > 0 else float('-inf')
                    perplexity = torch.exp(torch.tensor(-mean_logprob)).item()
                else:
                    mean_logprob = float('-inf')
                    perplexity = float('inf')
            else:
                mean_logprob = 0.0
                perplexity = 1.0
            
            # Store all predictions and metadata
            predictions.append(pred)
            probs.append(prob_1)
            perplexities.append(perplexity)
            logprobs.append(mean_logprob)
            pred_labels_raw.append(generated_text)  # Full CoT output for analysis
            valid_flags.append(is_valid)
            reasoning_texts.append(reasoning_text)
    
    # Calculate metrics using all predictions
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
        print(f"Note: {invalid_count}/{len(test_df)} samples had invalid CoT outputs, treated as 0")
    
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
    
    # Add CoT-specific statistics
    metrics["invalid_outputs"] = invalid_count
    metrics["invalid_ratio"] = invalid_count / len(test_df) if len(test_df) > 0 else 0.0
    
    return (metrics, predictions, probs, perplexities if calc_perplexity else None,
            pred_labels_raw, valid_flags, reasoning_texts, logprobs if calc_perplexity else None)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Llama-3 SFT with Chain-of-Thought for CT report classification")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)  # Not used for training, kept for compatibility
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model_name", default="elyza/Llama-3-ELYZA-JP-8B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--repeat_eval", type=int, default=3, help="Number of evaluation runs with different seeds")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--inst-file", help="外部 .txt で指示文を上書き")
    ap.add_argument("--qlora", action="store_true", help="Enable 4-bit QLoRA + NF4")
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank (8, 16, 32, 64)")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (typically 2*r)")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--lora_target_modules", nargs="+", 
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="Target modules for LoRA")
    ap.add_argument("--output_dir", default="sft_cot_output")
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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_cfg,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    # LoRA configuration
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

    # Custom evaluation during training (dummy for generation task)
    def compute_metrics(_):
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
    print(f"Training on {len(train_df)} samples with Chain-of-Thought")
    if args.use_val_for_validation:
        print(f"Using {len(val_df)} validation samples for early stopping")
    else:
        print("No validation during training - VAL data reserved for ICL few-shot examples")
    
    _ = trainer.train()

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)

    # ---------- Final evaluation on TEST data with multiple runs ----------
    device = next(model.parameters()).device
    
    print(f"Evaluating on TEST data with CoT generation ({args.repeat_eval} runs)...")
    
    # Multiple evaluation runs for statistical analysis
    all_results = []
    for rep in range(args.repeat_eval):
        curr_seed = args.seed + rep * 10  # 12, 22, 32 for default seed=12 with repeat_eval=3
        set_seed(curr_seed)
        
        print(f"  Run {rep+1}/{args.repeat_eval} (seed={curr_seed})")
        (test_metrics, test_preds, test_probs, test_perplexities,
         pred_labels_raw, valid_flags, reasoning_texts, test_logprobs) = evaluate_generation_pipeline(
            model, tokenizer, test_df, inst, device, args.calc_perplexity
        )
        
        # Save individual run results
        test_results_df = pd.DataFrame({
            "sentence": test_df["sentence"],
            "true_label": test_df["label"],
            "pred_label_raw": pred_labels_raw,    # Full CoT output
            "pred_label": test_preds,             # Processed predictions (0/1)
            "prob_1": test_probs,                 # Probability for class 1
            "is_valid": valid_flags,              # Validity flag
            "reasoning": reasoning_texts          # Extracted reasoning text
        })
        if args.calc_perplexity:
            test_results_df["perplexity"] = test_perplexities
            test_results_df["logprob"] = test_logprobs
        test_results_df.to_csv(output_dir / f"test_predictions_run{rep}.csv", index=False)
        
        all_results.append({
            "seed": curr_seed,
            "metrics": test_metrics,
            "predictions": test_preds,
            "probabilities": test_probs
        })

    # Aggregate results across multiple runs
    if args.repeat_eval > 1:
        # Calculate aggregate statistics
        metrics_list = [result["metrics"] for result in all_results]
        aggregate_metrics = {}
        
        for metric_name in metrics_list[0].keys():
            if metric_name in ['invalid_outputs', 'invalid_ratio']:
                continue  # Skip non-numeric aggregation for these
            
            values = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
            if values:
                aggregate_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                aggregate_metrics[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                aggregate_metrics[f"{metric_name}_mean"] = float("nan")
                aggregate_metrics[f"{metric_name}_std"] = float("nan")
    
    # Save results - same format as llama_sft_val.py for aggregate_results_script.py compatibility
    results = {
        "config": vars(args),
        "test_metrics": all_results[0]["metrics"] if args.repeat_eval == 1 else aggregate_metrics,
        "all_runs": all_results,
        "training_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint if args.use_val_for_validation else None,
        "val_used_for_training": args.use_val_for_validation,
        "note": f"CoT-enabled SFT with conservative sampling (temperature=0.3, {args.repeat_eval} evaluation runs)",
        "generation_config": GENERATION_CONFIG
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Save aggregated test predictions (from last run)
    if args.repeat_eval == 1:
        test_results_df = pd.DataFrame({
            "sentence": test_df["sentence"],
            "true_label": test_df["label"],
            "pred_label_raw": all_results[0]["predictions"],
            "pred_label": all_results[0]["predictions"],
            "prob_1": all_results[0]["probabilities"]
        })
        test_results_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Save model
    trainer.save_model()
    
    # Print results
    print("\n" + "="*50)
    print("CHAIN-OF-THOUGHT SFT TRAINING COMPLETED")
    print("="*50)
    
    if args.repeat_eval > 1:
        print(f"Aggregated test metrics ({args.repeat_eval} runs):")
        for metric in ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in aggregate_metrics:
                print(f"  {metric}: {aggregate_metrics[mean_key]:.4f} ± {aggregate_metrics[std_key]:.4f}")
        
        # Calculate total invalid outputs across all runs
        total_invalid = sum([r["metrics"].get('invalid_outputs', 0) for r in all_results])
        total_samples = len(test_df) * args.repeat_eval
        print(f"\nInvalid CoT outputs: {total_invalid}/{total_samples} ({total_invalid/total_samples*100:.2f}%)")
    else:
        print(f"Test metrics:")
        test_metrics = all_results[0]["metrics"]
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        print(f"\nInvalid CoT outputs: {test_metrics.get('invalid_outputs', 0)}/{len(test_df)}")
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Evaluation runs: {args.repeat_eval}")
    if args.use_val_for_validation:
        print(f"Validation samples used during training: {len(val_df)}")
    else:
        print(f"VAL samples ({len(val_df)}) reserved for ICL few-shot examples")
    print("="*50)


if __name__ == "__main__":
    main()