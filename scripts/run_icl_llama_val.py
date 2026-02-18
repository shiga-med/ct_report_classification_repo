#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_icl_llama_v3_fixed.py (FULLY FIXED VERSION with VAL data for few-shot)

Major fixes:
- Use VAL data for few-shot examples instead of TRAIN data
- Proper probability calculation using logits
- Added perplexity-based confidence
- Batch processing support
- Better reproducibility
- Fixed JSON serialization issue with Path objects
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

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Common instruction for all experiments
DEFAULT_INSTRUCTION = """以下のCT所見の文章を読んで、次の条件を満たすかどうかを判断してください：

    1. フォローや治療が必要な新規病変の存在
    2. 既存病変の悪化
    3. 追加検査または治療の明確な推奨

もし上記の条件の内一つでも満たす場合には「1」、一つも満たさない場合は「0」と出力してください。
必ず「1」または「0」のどちらかは出力します。
なお、回答となる数値はint型で返し、他には何も含めないことを厳守してください。"""


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_reproducibility(seed: int):
    """Complete reproducibility settings"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- prompt utils ----------
def create_fewshot_pool(val_df: pd.DataFrame, n0:int, n1:int, seed:int)->Tuple[List[str],List[str]]:
    """Create few-shot examples from VAL data instead of TRAIN data"""
    rng = np.random.default_rng(seed)
    l0 = val_df[val_df["label"]==0]["sentence"].tolist()
    l1 = val_df[val_df["label"]==1]["sentence"].tolist()
    if len(l0)<n0 or len(l1)<n1:
        raise ValueError(f"VALデータでfew-shotサンプルが足りません: label0={len(l0)}, label1={len(l1)}, 必要数=({n0}, {n1})")
    s0 = rng.choice(l0, size=n0, replace=False).tolist()
    s1 = rng.choice(l1, size=n1, replace=False).tolist()
    return s0, s1


def order_examples(label0_sents, label1_sents, strategy, rng):
    ex = ([{"sentence":s,"label":0} for s in label0_sents] +\
          [{"sentence":s,"label":1} for s in label1_sents])
    if strategy == "random":
        rng.shuffle(ex)
        return ex
    if strategy == "alternating":
        result = []
        n0, n1 = len(label0_sents), len(label1_sents)
        
        if n0 > n1:
            # label0の方が多い場合：余りのlabel0を先頭に配置
            for i in range(n0 - n1):
                result.append({"sentence": label0_sents[i], "label": 0})
            # 残りを交互に配置（label1から開始）
            for i in range(n1):
                result.append({"sentence": label1_sents[i], "label": 1})
                result.append({"sentence": label0_sents[n0 - n1 + i], "label": 0})
        elif n1 > n0:
            # label1の方が多い場合：余りのlabel1を先頭に配置
            for i in range(n1 - n0):
                result.append({"sentence": label1_sents[i], "label": 1})
            # 残りを交互に配置（label0から開始）
            for i in range(n0):
                result.append({"sentence": label0_sents[i], "label": 0})
                result.append({"sentence": label1_sents[n1 - n0 + i], "label": 1})
        else:
            # 同数の場合：そのまま交互に配置（label0から開始）
            for i in range(n0):
                result.append({"sentence": label0_sents[i], "label": 0})
                result.append({"sentence": label1_sents[i], "label": 1})
        
        return result
    if strategy == "label0_first":
        return ex[:len(label0_sents)] + ex[len(label0_sents):]
    if strategy == "label1_first":
        return ex[len(label0_sents):] + ex[:len(label0_sents)]
    raise ValueError(strategy)


def build_prompt(text:str, few_shots:List[dict], instruction:str)->str:
    prompt = ("以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
              "要求を適切に満たす応答を書きなさい。\n\n")
    prompt += f"### 指示:\n{instruction}\n\n"
    for fs in few_shots:
        prompt += f"### 入力:\n{fs['sentence']}\n\n### 応答:\n{fs['label']}\n\n"
    prompt += f"### 入力:\n{text}\n\n### 応答:\n"
    return prompt


def parse_response(resp:str)->Optional[int]:
    m = re.fullmatch(r"\s*([01])\s*", resp.strip(), re.DOTALL)
    return int(m.group(1)) if m else None


# ---------- model ----------
def load_model_and_tokenizer(model_name_or_path:str, gpu:bool):
    """Load model and tokenizer separately for better control"""
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    
    # Check if path contains PEFT adapter
    is_peft_model = PEFT_AVAILABLE and Path(model_name_or_path).exists() and (Path(model_name_or_path) / "adapter_config.json").exists()
    
    if is_peft_model:
        print(f"Loading PEFT adapter from {model_name_or_path}")
        # Load base model name from adapter config
        import json
        with open(Path(model_name_or_path) / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "tokyotech-llm/Llama-3-Swallow-8B-v0.1")
        
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            quantization_config=cfg if device.startswith("cuda") else None,
            device_map=device,
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
    else:
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


# ---------- classify with confidence ----------
def classify_with_confidence(df: pd.DataFrame, model, tokenizer, device: str,
                            few_shots: List[dict], instruction: str,
                            calc_perplexity: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return predicted labels, probabilities, perplexities, logprobs, and invalid sample count"""
    model.eval()
    preds, probs, perplexities, logprobs = [], [], [], []
    invalid_count = 0  # Counter for invalid outputs (not 0 or 1)
    
    # Get token IDs for probability calculation (still needed for logits)
    one_id = tokenizer.convert_tokens_to_ids("1")
    zero_id = tokenizer.convert_tokens_to_ids("0")
    
    # Handle unknown tokens
    if one_id == tokenizer.unk_token_id:
        one_tokens = tokenizer("1", add_special_tokens=False)["input_ids"]
        one_id = one_tokens[0] if one_tokens else tokenizer.unk_token_id
    
    if zero_id == tokenizer.unk_token_id:
        zero_tokens = tokenizer("0", add_special_tokens=False)["input_ids"]
        zero_id = zero_tokens[0] if zero_tokens else tokenizer.unk_token_id
    
    for text in tqdm(df["sentence"].tolist(), desc="classify"):
        prompt = build_prompt(text, few_shots, instruction)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            # Generate with scores (unified greedy decoding for all models)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=1e-8,      # Very small value instead of 0.0 for ELYZA compatibility
                do_sample=False,       # Greedy decoding
                top_k=1,              # Select only the most probable token
                top_p=0.0,            # Disable nucleus sampling
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Get generated text (same method as CT_Llama3_S_total_study.py)
            full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Extract generated text by removing the prompt (same as CT_Llama3_S_total_study.py)
            generated_text = full_output.replace(prompt, "")
            
            # Parse response: take first line and strip (same as CT_Llama3_S_total_study.py)
            pred_label = generated_text.split("\n")[0].strip()
            
            # Determine prediction from text output (max_new_tokens=1, so single token only)
            # Convert to string and strip whitespace for consistent processing
            pred_str = str(pred_label).strip()
            
            if pred_str == '0':
                pred = 0
            elif pred_str == '1':
                pred = 1
            elif pred_str and len(pred_str) == 1 and pred_str.isdigit():
                # Other single digits (2, 3, 4, ...) - count as invalid and assign 0
                pred = 0
                invalid_count += 1
                print(f"Invalid numeric output '{pred_str}' -> assigned 0 (invalid #{invalid_count})")
            else:
                # Non-digit output or empty output - count as invalid and assign 0
                pred = 0
                invalid_count += 1
                if not pred_str:
                    print(f"Empty output -> assigned 0 (invalid #{invalid_count})")
                else:
                    print(f"Invalid output '{pred_str}' -> assigned 0 (invalid #{invalid_count})")
            
            # Calculate probabilities for single token generation (max_new_tokens=1)
            if outputs.scores and len(outputs.scores) > 0:
                logits = outputs.scores[0][0]  # Single generated token logits
                all_probs = F.softmax(logits, dim=-1)
                
                # Get probabilities for "0" and "1" tokens and normalize
                if (one_id != tokenizer.unk_token_id and 
                    zero_id != tokenizer.unk_token_id and
                    one_id < len(all_probs) and zero_id < len(all_probs)):
                    
                    p0 = all_probs[zero_id].item()  # int 0 probability
                    p1 = all_probs[one_id].item()   # int 1 probability
                    
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
                
                perplexities.append(perplexity)
                logprobs.append(logprob)
            
            preds.append(pred)
            probs.append(prob_1)
    
    return (np.array(preds), np.array(probs), 
            np.array(perplexities) if calc_perplexity else None,
            np.array(logprobs) if calc_perplexity else None, invalid_count)


# ---------- metrics ----------
def compute_metrics(y_true, y_pred, probs):
    """
    Compute metrics using discrete predictions for precision/recall/f1 
    and probabilities for ROC-AUC/PR-AUC (same as llama_sft_val.py)
    """
    # Convert to numpy arrays
    pred_array = np.array(y_pred)
    prob_array = np.array(probs)
    true_array = np.array(y_true)
    
    # Calculate discrete metrics using predictions
    accuracy = accuracy_score(true_array, pred_array)
    precision = precision_score(true_array, pred_array, zero_division=0)
    recall = recall_score(true_array, pred_array, zero_division=0)
    f1 = f1_score(true_array, pred_array, zero_division=0)
    
    # Calculate probability-based metrics
    if len(np.unique(true_array)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(true_array, prob_array)
        pr_auc = average_precision_score(true_array, prob_array)
    
    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=matthews_corrcoef(true_array, pred_array),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )


# ---------- Single experiment run function ----------
def run_single_experiment(args):
    ensure_reproducibility(args.seed)

    instruction = Path(args.inst_file).read_text(encoding="utf-8") if args.inst_file else DEFAULT_INSTRUCTION

    n0 = args.n0 if args.n0 is not None else args.n_fewshot
    n1 = args.n1 if args.n1 is not None else args.n_fewshot

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    
    # Ensure label column is int
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].astype(int)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_name_or_path, args.gpu)

    # Create few-shot pool from VAL data instead of TRAIN data
    if args.resample_examples:
        fixed_pool = None
    else:
        fixed_pool = create_fewshot_pool(val_df, n0, n1, seed=args.seed)

    results = []
    for rep in range(args.repeat_eval):
        curr_seed = args.seed + rep * 10  # 12, 22, 32, 42, 52 for default seed=12
        set_seed(curr_seed)

        if args.resample_examples:
            label0_sents, label1_sents = create_fewshot_pool(val_df, n0, n1, seed=curr_seed)
        else:
            label0_sents, label1_sents = fixed_pool

        rng = random.Random(curr_seed)
        few_shots = order_examples(label0_sents, label1_sents, args.order_strategy, rng)

        # Classify test split only (val is used for few-shot examples)
        # Note: We can still evaluate on val for debugging, but main evaluation is on test
        if args.enable_val_eval:
            y_pred_val, probs_val, perp_val, logprob_val, invalid_val = classify_with_confidence(
                val_df, model, tokenizer, device, few_shots, instruction, args.calc_perplexity
            )
        else:
            y_pred_val, probs_val, perp_val, logprob_val, invalid_val = None, None, None, None, 0
        
        y_pred_test, probs_test, perp_test, logprob_test, invalid_test = classify_with_confidence(
            test_df, model, tokenizer, device, few_shots, instruction, args.calc_perplexity
        )

        # Calculate metrics
        if args.enable_val_eval:
            res_val = compute_metrics(val_df["label"].values, y_pred_val, probs_val)
            # Add perplexity and logprob stats if calculated
            if args.calc_perplexity:
                res_val["perplexity_mean"] = float(np.mean(perp_val))
                res_val["perplexity_std"] = float(np.std(perp_val))
                res_val["logprob_mean"] = float(np.mean(logprob_val))
                res_val["logprob_std"] = float(np.std(logprob_val))
        else:
            res_val = None
            
        res_test = compute_metrics(test_df["label"].values, y_pred_test, probs_test)
        if args.calc_perplexity:
            res_test["perplexity_mean"] = float(np.mean(perp_test))
            res_test["perplexity_std"] = float(np.std(perp_test))
            res_test["logprob_mean"] = float(np.mean(logprob_test))
            res_test["logprob_std"] = float(np.std(logprob_test))

        # Save predictions
        if args.enable_val_eval:
            val_pred_df = pd.DataFrame({
                "sentence": val_df["sentence"],
                "label": val_df["label"],
                "pred": y_pred_val,
                "prob": probs_val
            })
            if args.calc_perplexity:
                val_pred_df["perplexity"] = perp_val
                val_pred_df["logprob"] = logprob_val
            val_pred_df.to_csv(out / f"run{rep}_val_preds.csv", index=False)
        
        test_pred_df = pd.DataFrame({
            "sentence": test_df["sentence"],
            "label": test_df["label"],
            "pred": y_pred_test,
            "prob": probs_test
        })
        if args.calc_perplexity:
            test_pred_df["perplexity"] = perp_test
            test_pred_df["logprob"] = logprob_test
        test_pred_df.to_csv(out / f"run{rep}_test_preds.csv", index=False)

        # Add invalid sample counts to results
        if args.enable_val_eval:
            res_val['invalid_samples'] = invalid_val
        res_test['invalid_samples'] = invalid_test
        
        results.append(dict(seed=curr_seed, val=res_val, test=res_test, 
                           invalid_counts={'val': invalid_val, 'test': invalid_test}))

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
        splits = ["test"]
        if args.enable_val_eval:
            splits.append("val")
            
        for split in splits:
            for k in keys:
                # Skip invalid_samples from aggregation (it's an integer count, not a metric)
                if k == 'invalid_samples':
                    continue
                    
                arr = [r[split][k] for r in results if r[split] is not None and not np.isnan(r[split][k])]
                if arr:  # Only compute if we have non-NaN values
                    agg[f"{split}_{k}_mean"] = float(np.mean(arr))
                    agg[f"{split}_{k}_sd"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                else:
                    agg[f"{split}_{k}_mean"] = float("nan")
                    agg[f"{split}_{k}_sd"] = float("nan")
        summary["aggregate"] = agg

    # Save summary
    with open(out / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print results
    print("\n" + "="*50)
    print("ICL EVALUATION COMPLETED")
    print("="*50)
    resample_status = "resampled each repeat" if args.resample_examples else "fixed examples"
    print(f"Few-shot examples extracted from VAL data (n0={n0}, n1={n1}, {resample_status})")
    if not args.enable_val_eval:
        print(f"Evaluation performed on TEST data only (VAL evaluation skipped)")
    else:
        print(f"Evaluation performed on TEST and VAL data")
    
    # Print invalid sample statistics
    if args.repeat_eval > 1:
        total_invalid_test = sum([r['invalid_counts']['test'] for r in results])
        total_test_samples = len(test_df) * args.repeat_eval
        print(f"Invalid outputs - TEST: {total_invalid_test}/{total_test_samples} ({total_invalid_test/total_test_samples*100:.2f}%)")
        if args.enable_val_eval:
            total_invalid_val = sum([r['invalid_counts']['val'] for r in results])
            total_val_samples = len(val_df) * args.repeat_eval
            print(f"Invalid outputs - VAL: {total_invalid_val}/{total_val_samples} ({total_invalid_val/total_val_samples*100:.2f}%)")
    else:
        invalid_test = results[0]['invalid_counts']['test']
        print(f"Invalid outputs - TEST: {invalid_test}/{len(test_df)} ({invalid_test/len(test_df)*100:.2f}%)")
        if args.enable_val_eval:
            invalid_val = results[0]['invalid_counts']['val']
            print(f"Invalid outputs - VAL: {invalid_val}/{len(val_df)} ({invalid_val/len(val_df)*100:.2f}%)")
    
    print("="*50)
    
    if args.repeat_eval > 1:
        print("Aggregated test results:")
        for metric in ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]:
            mean_key = f"test_{metric}_mean"
            sd_key = f"test_{metric}_sd"
            if mean_key in agg:
                print(f"  {metric}: {agg[mean_key]:.4f} \u00b1 {agg[sd_key]:.4f}")
        
        if args.enable_val_eval:
            print("Aggregated val results:")
            for metric in ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]:
                mean_key = f"val_{metric}_mean"
                sd_key = f"val_{metric}_sd"
                if mean_key in agg:
                    print(f"  {metric}: {agg[mean_key]:.4f} \u00b1 {agg[sd_key]:.4f}")
    else:
        print("Test results:")
        for k, v in results[0]["test"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        
        if args.enable_val_eval:
            print("Val results:")
            for k, v in results[0]["val"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
    
    print("="*50)


# ---------- Experiment Patterns Definition ----------
EXPERIMENT_PATTERNS = [
    # 1. Few-shot例の数による影響 (5回実行で統計的検証)
    {"name": "n_fewshot_0", "n_fewshot": 0, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_0"},
    {"name": "n_fewshot_1_label0", "n0": 1, "n1": 0, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_1_label0"},
    {"name": "n_fewshot_1_label1", "n0": 0, "n1": 1, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_1_label1"},
    {"name": "n_fewshot_2", "n_fewshot": 2, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_2"},
    {"name": "n_fewshot_5", "n_fewshot": 5, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_5"},
    {"name": "n_fewshot_10", "n_fewshot": 10, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_10"},
    {"name": "n_fewshot_15", "n_fewshot": 15, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_15"},
    {"name": "n_fewshot_25", "n_fewshot": 25, "repeat_eval": 3, "resample_examples": True, "output_subdir": "n_fewshot_25"},

    # 2. Few-shot例の並び順による影響 (例題数: 10, 20, 30) - 5回実行で統計的検証
    # n_fewshot=10
    {"name": "order_10_label0_first", "n0": 5, "n1": 5, "n_fewshot": 10, "order_strategy": "label0_first", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_10_label0_first"},
    {"name": "order_10_label1_first", "n0": 5, "n1": 5, "n_fewshot": 10, "order_strategy": "label1_first", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_10_label1_first"},
    {"name": "order_10_alternating", "n0": 5, "n1": 5, "n_fewshot": 10, "order_strategy": "alternating", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_10_alternating"},
    # n_fewshot=20
    {"name": "order_20_label0_first", "n0": 10, "n1": 10, "n_fewshot": 20, "order_strategy": "label0_first", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_20_label0_first"},
    {"name": "order_20_label1_first", "n0":10, "n1": 10, "n_fewshot": 20, "order_strategy": "label1_first", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_20_label1_first"},
    {"name": "order_20_alternating", "n0": 10, "n1": 10, "n_fewshot": 20, "order_strategy": "alternating", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_20_alternating"},
    # n_fewshot=30
    {"name": "order_30_label0_first", "n0": 15, "n1": 15, "n_fewshot": 30, "order_strategy": "label0_first", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_30_label0_first"},
    {"name": "order_30_label1_first", "n0": 15, "n1": 15, "n_fewshot": 30, "order_strategy": "label1_first", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_30_label1_first"},
    {"name": "order_30_alternating", "n0": 15, "n1": 15, "n_fewshot": 30, "order_strategy": "alternating", "repeat_eval": 3, "resample_examples": True, "output_subdir": "order_30_alternating"},


    # 3. クラス不均衡なFew-shot例 (5回実行で統計的検証)
    {"name": "imbalance_1_9", "n0": 1, "n1": 9, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_1_9"},
    {"name": "imbalance_9_1", "n0": 9, "n1": 1, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_9_1"},
    {"name": "imbalance_10_0", "n0": 10, "n1": 0, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_10_0"},
    {"name": "imbalance_0_10", "n0": 0, "n1": 10, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_0_10"},
    {"name": "imbalance_3_7", "n0": 3, "n1": 7, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_3_7"},
    {"name": "imbalance_7_3", "n0": 7, "n1": 3, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_7_3"},
    {"name": "imbalance_5_5", "n0": 5, "n1": 5, "repeat_eval": 5, "resample_examples": True, "order_strategy": "random", "output_subdir": "imbalance_5_5"},
]


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("In-Context Learning with Llama-3 (VAL for few-shot)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model_name_or_path", default="elyza/Llama-3-ELYZA-JP-8B", help="Model name or path (supports PEFT adapters)")
    ap.add_argument("--n_fewshot", type=int, default=5)
    ap.add_argument("--n0", type=int)
    ap.add_argument("--n1", type=int)
    ap.add_argument("--order-strategy", choices=["random","alternating","label0_first","label1_first"], default="alternating")
    ap.add_argument("--repeat-eval", type=int, default=5)
    ap.add_argument("--resample-examples", action="store_true", help="few-shot 抽出を毎回やり直す")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--inst_file", help="指示文を外部 txt から読み込む")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--output_dir", default="icl_nb_prompt")
    ap.add_argument("--calc_perplexity", action="store_true", help="Calculate perplexity and logprob-based confidence")
    ap.add_argument("--enable-val-eval", action="store_true", help="VALデータでの評価を有効化（デフォルトはスキップ）")
    ap.add_argument("--run_patterns", action="store_true", help="定義された複数の実験パターンを実行する")
    args = ap.parse_args()

    if args.run_patterns:
        base_output_dir = Path(args.output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Running predefined experiment patterns into {base_output_dir}...")
        for pattern in EXPERIMENT_PATTERNS:
            print(f"\n=== Running pattern: {pattern['name']} ===")
            
            # Create a new argparse.Namespace for each pattern
            pattern_args = argparse.Namespace(**vars(args))
            
            # Update arguments for the current pattern
            for key, value in pattern.items():
                if key == "name": continue # Skip name
                if key == "output_subdir":
                    pattern_args.output_dir = base_output_dir / value
                else:
                    setattr(pattern_args, key, value)
            
            # Ensure n0/n1 are set correctly if n_fewshot is used and n0/n1 are not explicitly set
            if "n_fewshot" in pattern and "n0" not in pattern and "n1" not in pattern:
                pattern_args.n0 = pattern_args.n_fewshot // 2
                pattern_args.n1 = pattern_args.n_fewshot - pattern_args.n0
            
            # Run the single experiment
            run_single_experiment(pattern_args)
            print(f"=== Pattern {pattern['name']} completed. Results in {pattern_args.output_dir} ===")
            
            # Clear GPU cache after each pattern to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU cache cleared after {pattern['name']}")
        
        print("\nAll predefined patterns completed.")
    else:
        # Run as a single experiment (original behavior)
        run_single_experiment(args)


if __name__ == "__main__":
    main()
