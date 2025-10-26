#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llama_cot_eval.py – Zero-shot Chain-of-Thought evaluation for CT report classification

This script evaluates pre-trained Llama-3 models using Chain-of-Thought prompting
without any fine-tuning. It's designed for comparison with ICL, SFT, and CoT-SFT methods.

Key features:
- Zero-shot evaluation (no training, no few-shot examples)
- Chain-of-Thought prompting with step-by-step reasoning
- Extended token generation for reasoning process
- Final answer extraction from generated reasoning text
- Compatible output format with aggregate_results_script.py
- Multiple evaluation runs with different seeds for statistical analysis
"""

import argparse, json, random, re, warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Chain-of-Thought instruction for Zero-shot evaluation
DEFAULT_INSTRUCTION = """以下のCT所見の文章を読んで、次の条件を満たすかどうかを判断してください：

    1. フォローや治療が必要な新規病変の存在
    2. 既存病変の悪化
    3. 追加検査または治療の明確な推奨

各条件について検討し、根拠を示しながら分析してください。
最終的に、上記の条件の内一つでも満たす場合には「1」、一つも満たさない場合は「0」と結論してください。
なお、最終的な回答は必ずint型の数値（0または1）で回答の最後に示してください。"""

# Chain-of-Thought prompt template for Zero-shot evaluation
PROMPT_TMPL = (
    "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    "### 指示:\n{instruction}\n\n"
    "### 入力:\n{sentence}\n\n"
    "段階的に考えてみましょう。\n\n"
    "### 応答:\n"
)

# Generation configuration for Zero-shot CoT (extended tokens, conservative sampling)
GENERATION_CONFIG = {
    "max_new_tokens": 256,   # Extended for reasoning process in zero-shot setting
    "temperature": 0.3,      # Conservative sampling for stability
    "do_sample": True,       # Enable sampling
    "top_k": 40,            # Limited choices for stability
    "top_p": 0.95,          # Conservative nucleus sampling
    "repetition_penalty": 1.1,  # Light repetition penalty
}

# ---------- utils ----------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_reproducibility(seed: int):
    """Complete reproducibility settings"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# ---------- model ----------
def load_model_and_tokenizer(model_name_or_path: str, gpu: bool):
    """Load model and tokenizer for evaluation"""
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        quantization_config=cfg if device.startswith("cuda") else None,
        device_map=device,
    )
    
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


# ---------- evaluate with confidence ----------
def evaluate_zero_shot_cot(df: pd.DataFrame, model, tokenizer, device: str,
                          instruction: str, calc_perplexity: bool = True) -> Tuple[Dict, List, List, Optional[List], List, List, List, Optional[List]]:
    """
    Evaluate using Zero-shot Chain-of-Thought generation
    Returns: (metrics, predictions, probs, perplexities, raw_outputs, valid_flags, reasoning_texts, logprobs)
    """
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
    
    for row in tqdm(df.itertuples(), desc="Zero-shot CoT evaluation", total=len(df)):
        prompt = PROMPT_TMPL.format(instruction=instruction, sentence=row.sentence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            # Generate with extended tokens for CoT reasoning
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
                print(f"Invalid Zero-shot CoT output -> assigned 0 (invalid #{invalid_count})")
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
    true_labels_array = np.array(df["label"])
    
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
        print(f"Note: {invalid_count}/{len(df)} samples had invalid Zero-shot CoT outputs, treated as 0")
    
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
    
    # Add Zero-shot CoT-specific statistics
    metrics["invalid_outputs"] = invalid_count
    metrics["invalid_ratio"] = invalid_count / len(df) if len(df) > 0 else 0.0
    
    return (metrics, predictions, probs, perplexities if calc_perplexity else None,
            pred_labels_raw, valid_flags, reasoning_texts, logprobs if calc_perplexity else None)


# ---------- Single experiment run function ----------
def run_single_experiment(args):
    ensure_reproducibility(args.seed)

    instruction = Path(args.inst_file).read_text(encoding="utf-8") if args.inst_file else DEFAULT_INSTRUCTION

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Load data
    test_df = pd.read_csv(args.test_csv)
    
    # Ensure label column is int
    test_df["label"] = test_df["label"].astype(int)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_name_or_path, args.gpu)

    results = []
    for rep in range(args.repeat_eval):
        curr_seed = args.seed + rep * 10  # 12, 22, 32 for default seed=12 with repeat_eval=3
        set_seed(curr_seed)

        # Zero-shot CoT evaluation (no few-shot examples needed)
        (test_metrics, test_preds, test_probs, test_perplexities,
         pred_labels_raw, valid_flags, reasoning_texts, test_logprobs) = evaluate_zero_shot_cot(
            test_df, model, tokenizer, device, instruction, args.calc_perplexity
        )

        # Save predictions
        test_pred_df = pd.DataFrame({
            "sentence": test_df["sentence"],
            "label": test_df["label"],
            "pred": test_preds,
            "prob": test_probs,
            "pred_raw": pred_labels_raw,      # Full CoT reasoning
            "is_valid": valid_flags,          # Validity flags
            "reasoning": reasoning_texts      # Extracted reasoning
        })
        if args.calc_perplexity:
            test_pred_df["perplexity"] = test_perplexities
            test_pred_df["logprob"] = test_logprobs
        test_pred_df.to_csv(out / f"run{rep}_test_preds.csv", index=False)

        # Add invalid sample counts to results
        test_metrics['invalid_samples'] = test_metrics['invalid_outputs']
        
        results.append(dict(seed=curr_seed, test=test_metrics, 
                           invalid_counts={'test': test_metrics['invalid_outputs']}))

    # Convert Path objects to strings before saving to JSON
    config_dict = vars(args).copy()
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)

    # Aggregate results
    summary = {"config": config_dict, "results": results}
    
    if args.repeat_eval > 1:
        keys = list(results[0]["test"].keys())
        agg = {}
            
        for k in keys:
            # Skip invalid_samples from aggregation (it's an integer count, not a metric)
            if k in ['invalid_samples', 'invalid_outputs', 'invalid_ratio']:
                continue
                
            arr = [r["test"][k] for r in results if not np.isnan(r["test"][k])]
            if arr:  # Only compute if we have non-NaN values
                agg[f"test_{k}_mean"] = float(np.mean(arr))
                agg[f"test_{k}_sd"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            else:
                agg[f"test_{k}_mean"] = float("nan")
                agg[f"test_{k}_sd"] = float("nan")
        summary["aggregate"] = agg

    # Save summary
    with open(out / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print results
    print("\n" + "="*50)
    print("ZERO-SHOT CHAIN-OF-THOUGHT EVALUATION COMPLETED")
    print("="*50)
    print(f"Zero-shot CoT evaluation performed on TEST data")
    
    # Print invalid sample statistics
    if args.repeat_eval > 1:
        total_invalid_test = sum([r['invalid_counts']['test'] for r in results])
        total_test_samples = len(test_df) * args.repeat_eval
        print(f"Invalid outputs - TEST: {total_invalid_test}/{total_test_samples} ({total_invalid_test/total_test_samples*100:.2f}%)")
    else:
        invalid_test = results[0]['invalid_counts']['test']
        print(f"Invalid outputs - TEST: {invalid_test}/{len(test_df)} ({invalid_test/len(test_df)*100:.2f}%)")
    
    print("="*50)
    
    if args.repeat_eval > 1:
        print("Aggregated test results:")
        for metric in ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]:
            mean_key = f"test_{metric}_mean"
            sd_key = f"test_{metric}_sd"
            if mean_key in agg:
                print(f"  {metric}: {agg[mean_key]:.4f} ± {agg[sd_key]:.4f}")
    else:
        print("Test results:")
        for k, v in results[0]["test"].items():
            if isinstance(v, float) and k not in ['invalid_outputs', 'invalid_ratio']:
                print(f"  {k}: {v:.4f}")
    
    print("="*50)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Zero-shot Chain-of-Thought evaluation with Llama-3")
    ap.add_argument("--test_csv", required=True, help="Test data CSV file")
    ap.add_argument("--model_name_or_path", default="elyza/Llama-3-ELYZA-JP-8B", help="Model name or path")
    ap.add_argument("--repeat_eval", type=int, default=3, help="Number of evaluation runs with different seeds")
    ap.add_argument("--seed", type=int, default=12, help="Base seed for reproducibility")
    ap.add_argument("--inst_file", help="External instruction text file to override default")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")
    ap.add_argument("--output_dir", default="zero_shot_cot_results", help="Output directory")
    ap.add_argument("--calc_perplexity", action="store_true", help="Calculate perplexity-based confidence")
    args = ap.parse_args()

    # Run single experiment (Zero-shot CoT doesn't need pattern variations like ICL)
    run_single_experiment(args)


if __name__ == "__main__":
    main()