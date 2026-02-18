#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_preprocessing.py – FIXED VERSION

Major improvements:
- Perplexity-based confidence calculation for pseudo-labeling
- Class balance consideration options
- Data quality checks
- Better error handling
- Detailed logging and statistics
"""

import pandas as pd, numpy as np, random, re, argparse, os, json, logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
                          pipeline)
from tqdm import tqdm
import mojimoji

# ---------- constants ----------
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Common instruction for all experiments
DEFAULT_INSTRUCTION = """以下のCT所見の文章を読んで、次の条件を満たすかどうかを判断してください：

    1. フォローや治療が必要な新規病変の存在
    2. 既存病変の悪化
    3. 追加検査または治療の明確な推奨

もし上記の条件の内一つでも満たす場合には「1」、一つも満たさない場合は「0」と出力してください。
必ず「1」または「0」のどちらかは出力します。
なお、回答となる数値はint型で返し、他には何も含めないことを厳守してください。"""

# ---------- text utils ----------
def combine_texts(text1: str, text2: str) -> str:

    # text1 が NaN でない場合、最後に「。」がなければ付与
    if not pd.isna(text1) and not text1.endswith("。"):
        text1 += "。"
    # text2 が NaN でない場合、最後に「。」がなければ付与
    if not pd.isna(text2) and not text2.endswith("。"):
        text2 += "。"
    # text2 が NaN の場合は text1 を返す
    if pd.isna(text2):
        return text1
    # text1 と text2 を結合して返す
    return text1 + text2


def clean_text_advanced(text: str) -> str:
    """Advanced text cleaning with comprehensive special character removal and normalization"""
    if pd.isna(text):
        return ""
    
    # 1. 網羅的な特殊記号の削除
    # 矢印記号類
    text = re.sub(r'[→←↑↓↗↘↙↖↔↕⇒⇐⇑⇓⇔⇕]', '', text)
    # 図形・記号類
    text = re.sub(r'[●○◆◇■□▲△▼▽★☆◎※]', '', text)
    # 数学記号類
    text = re.sub(r'[±×÷≒≠≤≥∞∴∵∝∈∋⊂⊃∩∪]', '', text)
    # その他の記号類
    text = re.sub(r'[§¶†‡°′″‰℃℉]', '', text)
    # 罫線・ボックス描画文字
    text = re.sub(r'[─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬]', '', text)
    # 単位記号
    text = re.sub(r'[㎜㎝㎞㎏㎡㎥℃℉]', '', text)
    
    # 2. 英数字の半角化（数値の半角化を含む）
    text = mojimoji.zen_to_han(text, kana=False)
    
    # 3. 句読点の統一
    text = text.replace(',', '、').replace('.', '。')
    
    # 4. 改行・スペース処理（文境界の統一）
    # 連続する \r や \n を一つにまとめる
    text = re.sub(r'[\r\n]+', '\n', text)
    # \n の前に「、」または「。」がある場合はそのまま消去
    text = re.sub(r'(、|。)\n', r'\1', text)
    # 文間にスペースのみがある場合は「。」に置換（句読点がない文の境界）
    text = re.sub(r'([^。、\s])[ 　]+([あ-んア-ンa-zA-Z一-龯])', r'\1。\2', text)
    # 改行を「。」に置き換える
    text = re.sub(r'\n', '。', text)
    # 残りの連続する半角・全角スペースを１つの全角スペースに統一
    text = re.sub(r'[ 　]+', '　', text)
    
    # 5. 日付・時間に関わる数値のみを０に置換（修正版）
    # より包括的な日付パターンの検出と置換
    
    # 西暦年月日パターン（年が4桁）
    def normalize_date(match):
        return re.sub(r'\d', '0', match.group())
    
    text = re.sub(r'(\d{4})年(\d{1,2})月(\d{1,2})日', normalize_date, text)
    text = re.sub(r'(\d{4})/(\d{1,2})/(\d{1,2})', normalize_date, text)
    text = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', normalize_date, text)
    text = re.sub(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', normalize_date, text)
    
    # 2桁年のパターン（例：23年12月25日）
    text = re.sub(r'(\d{2})年(\d{1,2})月(\d{1,2})日', normalize_date, text)
    
    # 月日のみのパターン
    text = re.sub(r'(\d{1,2})月(\d{1,2})日', normalize_date, text)
    text = re.sub(r'(\d{1,2})/(\d{1,2})', normalize_date, text)  # 10/26形式
    
    # 時刻パターン
    text = re.sub(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', normalize_date, text)
    text = re.sub(r'(\d{1,2})時(\d{1,2})分(?:(\d{1,2})秒)?', normalize_date, text)
    
    # 年齢パターン
    text = re.sub(r'(\d{1,3})歳', normalize_date, text)
    text = re.sub(r'(\d{1,3})才', normalize_date, text)
    
    # 期間パターン
    text = re.sub(r'(\d+)年間', normalize_date, text)
    text = re.sub(r'(\d+)ヶ月', normalize_date, text)
    text = re.sub(r'(\d+)か月', normalize_date, text)
    text = re.sub(r'(\d+)日間', normalize_date, text)
    
    return text.strip()

# ---------- I/O ----------
def detect_encoding(file_path: str) -> str:
    """Detect file encoding, prioritizing common Japanese encodings"""
    try:
        import chardet
    except ImportError:
        print("chardet not available, using utf-8")
        return 'utf-8'
    
    encodings_to_try = ['utf-8', 'shift_jis', 'iso-2022-jp', 'euc-jp', 'cp932']
    
    # Try chardet first
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        confidence = result['confidence']
    
    print(f"Chardet detected: {detected_encoding} (confidence: {confidence:.2f})")
    
    # If confidence is high enough, use detected encoding
    if confidence > 0.8 and detected_encoding:
        return detected_encoding
    
    # Otherwise, try common Japanese encodings
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1000)  # Try reading first 1000 chars
            print(f"Successfully opened with encoding: {encoding}")
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    
    # Fallback to utf-8 with error handling
    print("Warning: Could not detect encoding reliably, using utf-8 with error handling")
    return 'utf-8'

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, bool]:
    """
    Load and preprocess data, returning (dataframe, has_doctor_label)
    - If has_doctor_label=False: regular data for train/val/test split
    - If has_doctor_label=True: test data with doctor labels (no split needed)
    """
    # Detect encoding
    encoding = detect_encoding(file_path)
    
    # Load data with detected encoding
    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        print("Failed with detected encoding, trying with errors='ignore'")
        df = pd.read_csv(file_path, encoding=encoding, errors='ignore')
    
    # Check if doctor label column exists
    has_doctor_label = "医師" in df.columns
    
    if has_doctor_label:
        print("Found doctor label column (医師) - processing as test data with doctor labels")
        # Process with doctor labels for test data only
        required_cols = ["対応必要の有無", "所見", "診断", "医師"]
        df = df[required_cols]
        df = df.rename(columns={
            "対応必要の有無": "label", 
            "所見": "text1", 
            "診断": "text2",
            "医師": "doctor_label"
        })
        
        # Encode doctor labels if they exist and are not already numeric
        if df["doctor_label"].dtype == 'object':
            le = LabelEncoder()
            # Fill NaN values with a placeholder before encoding
            df["doctor_label_filled"] = df["doctor_label"].fillna("Unknown")
            df["doctor_label_encoded"] = le.fit_transform(df["doctor_label_filled"])
            
            # Save label mapping
            label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Doctor label mapping: {label_mapping}")
            
            # Replace with original NaN where applicable
            df.loc[df["doctor_label"].isna(), "doctor_label_encoded"] = np.nan
            df = df.drop("doctor_label_filled", axis=1)
        else:
            df["doctor_label_encoded"] = df["doctor_label"]
    else:
        print("No doctor label column found - processing for train/val/test split")
        # Original processing for regular data
        df = df[["対応必要の有無", "所見", "診断"]]
        df = df.rename(columns={"対応必要の有無":"label", "所見":"text1", "診断":"text2"})
    
    # Common text processing
    df["sentence"] = df.apply(lambda r: combine_texts(r["text1"], r["text2"]), axis=1)
    df["sentence"] = df["sentence"].apply(clean_text_advanced)
    # 全角化（英数字は半角のまま保持、カタカナは全角）
    df["sentence"] = df["sentence"].apply(lambda x: mojimoji.han_to_zen(x, digit=False, ascii=False, kana=True) if not pd.isna(x) else "")
    
    # Drop original text columns
    drop_cols = ["text1", "text2"]
    df = df.drop(drop_cols, axis=1)
    
    return df, has_doctor_label

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality check"""
    report = {
        "total_samples": len(df),
        "labeled_samples": len(df.dropna(subset=["label"])),
        "unlabeled_samples": len(df[df["label"].isna()]),
        "issues": []
    }
    
    # Text length statistics
    if "sentence" in df.columns:
        text_lengths = df["sentence"].str.len()
        report["text_length_stats"] = {
            "mean": float(text_lengths.mean()),
            "std": float(text_lengths.std()),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "median": float(text_lengths.median())
        }
        
        # Check for extreme lengths
        if (text_lengths < 10).any():
            short_count = int((text_lengths < 10).sum())
            report["issues"].append(f"{short_count} samples with very short text (<10 chars)")
        
        if (text_lengths > 2000).any():
            long_count = int((text_lengths > 2000).sum())
            report["issues"].append(f"{long_count} samples with very long text (>2000 chars)")
    
    # Label distribution
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        report["label_distribution"] = {k: int(v) for k, v in label_counts.items()}
        
        # Check for invalid labels
        valid_labels = df["label"].dropna()
        invalid_labels = valid_labels[~valid_labels.isin([0, 1])]
        if len(invalid_labels) > 0:
            report["issues"].append(f"{int(len(invalid_labels))} samples with invalid labels")
    
    # Doctor label distribution (if exists)
    if "doctor_label" in df.columns:
        doctor_counts = df["doctor_label"].value_counts()
        report["doctor_label_distribution"] = {k: int(v) for k, v in doctor_counts.items()}
        
        if "doctor_label_encoded" in df.columns:
            encoded_counts = df["doctor_label_encoded"].value_counts()
            report["doctor_label_encoded_distribution"] = {k: int(v) for k, v in encoded_counts.items()}
    
    # Duplicate detection
    if "sentence" in df.columns:
        duplicates = df[df.duplicated(subset=["sentence"], keep=False)]
        if len(duplicates) > 0:
            report["duplicates"] = {
                "count": int(len(duplicates)),
                "unique_texts": int(df["sentence"].nunique())
            }
            
            # Check for conflicting labels
            for _, group in duplicates.groupby("sentence"):
                if "label" in group.columns:
                    unique_labels = group["label"].dropna().unique()
                    if len(unique_labels) > 1:
                        report["issues"].append(
                            f"Text with conflicting labels: {int(len(unique_labels))} different labels"
                        )
    
    return report

def split_data_improved(df: pd.DataFrame, test_size: float = 0.2, 
                       val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
    """
    Improved data splitting with stratification
    Two-stage split: 7:1:2 ratio
    Stage 1: 80% (train+val) : 20% (test)
    Stage 2: 87.5% (train) : 12.5% (val) of the 80% from stage 1
    Final ratio: Train: 70%, Val: 10%, Test: 20%
    """
    labeled_df = df.dropna(subset=["label"]).reset_index(drop=True)
    unlabeled_df = df[df["label"].isna()].reset_index(drop=True)
    
    # Check class distribution
    label_counts = labeled_df["label"].value_counts()
    print(f"Class distribution: {dict(label_counts)}")
    
    # Check for severe imbalance
    imbalance_ratio = label_counts.max() / label_counts.min()
    if imbalance_ratio > 3:
        print(f"Warning: High class imbalance (ratio: {imbalance_ratio:.2f})")
    
    # Stage 1: Split into train+val (80%) and test (20%)
    try:
        tr, test = train_test_split(
            labeled_df, test_size=test_size, random_state=random_state,
            stratify=labeled_df["label"]
        )
        
        # Stage 2: Split train+val into train (80%) and val (20%)
        # Adjust val_size for the remaining 80%: 0.2 / 0.8 = 0.25
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            tr, test_size=val_size_adjusted, random_state=random_state,
            stratify=tr["label"]
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to random split")
        tr, test = train_test_split(labeled_df, test_size=test_size, random_state=random_state)
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(tr, test_size=val_size_adjusted, random_state=random_state)
    
    # Print split statistics
    print(f"\nSplit results (7:1:2 ratio):")
    print(f"Train: {len(train)} samples ({len(train)/len(labeled_df)*100:.1f}%), distribution: {dict(train['label'].value_counts())}")
    print(f"Val: {len(val)} samples ({len(val)/len(labeled_df)*100:.1f}%), distribution: {dict(val['label'].value_counts())}")
    print(f"Test: {len(test)} samples ({len(test)/len(labeled_df)*100:.1f}%), distribution: {dict(test['label'].value_counts())}")
    print(f"Unlabeled: {len(unlabeled_df)} samples")
    
    return train.reset_index(drop=True), val.reset_index(drop=True), \
           test.reset_index(drop=True), unlabeled_df

# ---------- few-shot helpers ----------
def create_few_shot_examples(train_df: pd.DataFrame, n_examples: int = 5) -> List[dict]:
    label_0_df = train_df[train_df["label"]==0]
    label_1_df = train_df[train_df["label"]==1]

    n0 = min(n_examples, len(label_0_df))
    n1 = min(n_examples, len(label_1_df))
    n  = min(n0, n1)
    
    if n == 0:
        return []

    sampled_0 = label_0_df.sample(n=n, random_state=RANDOM_SEED)
    sampled_1 = label_1_df.sample(n=n, random_state=RANDOM_SEED)

    few_shot = []
    for i in range(n):
        few_shot.append({"sentence":sampled_0.iloc[i]["sentence"], "label":0})
        few_shot.append({"sentence":sampled_1.iloc[i]["sentence"], "label":1})
    
    print("Few-shot sequence:", [ex["label"] for ex in few_shot])
    return few_shot

def create_prompt(text: str, few_shots: List[dict], instruction: str) -> str:
    prompt = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    prompt += f"### 指示:\n{instruction}\n\n"
    for fs in few_shots:
        prompt += f"### 入力:\n{fs['sentence']}\n\n### 応答:\n{fs['label']}\n\n"
    prompt += f"### 入力:\n{text}\n\n### 応答:\n"
    return prompt

def predict_label_with_confidence(text: str, model, tokenizer, few_shots: List[dict], 
                                 instruction: str, device: str) -> Tuple[Optional[int], float, float]:
    """Predict label with logit-based confidence and perplexity"""
    prompt = create_prompt(text, few_shots, instruction)
    
    # Get token IDs
    one_id = tokenizer.convert_tokens_to_ids("1")
    zero_id = tokenizer.convert_tokens_to_ids("0")
    
    # Handle unknown tokens
    if one_id == tokenizer.unk_token_id:
        one_tokens = tokenizer("1", add_special_tokens=False)["input_ids"]
        one_id = one_tokens[0] if one_tokens else tokenizer.unk_token_id
    
    if zero_id == tokenizer.unk_token_id:
        zero_tokens = tokenizer("0", add_special_tokens=False)["input_ids"]
        zero_id = zero_tokens[0] if zero_tokens else tokenizer.unk_token_id
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    
    with torch.no_grad():
        # Generate with scores
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Get generated token
        generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        
        # Determine prediction
        if len(generated_ids) > 0:
            if generated_ids[0] == one_id:
                pred = 1
            elif generated_ids[0] == zero_id:
                pred = 0
            else:
                return None, 0.0, float('inf')
        else:
            return None, 0.0, float('inf')
        
        # Calculate confidence from logits
        confidence = 0.5  # default
        if outputs.scores and len(outputs.scores) > 0:
            logits = outputs.scores[0][0]
            all_probs = F.softmax(logits, dim=-1)
            
            p0 = all_probs[zero_id].item()
            p1 = all_probs[one_id].item()
            
            total = p0 + p1
            if total > 0:
                if pred == 1:
                    confidence = p1 / total
                else:
                    confidence = p0 / total
        
        # Calculate perplexity
        full_text = prompt + str(pred)
        full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        
        outputs = model(**full_inputs)
        
        # Calculate loss for the label token
        if outputs.logits.shape[1] > 1:
            logits = outputs.logits[0, -1, :]
            target = full_inputs.input_ids[0, -1]
            
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            perplexity = torch.exp(loss).item()
        else:
            perplexity = float('inf')
    
    return pred, confidence, perplexity

# ---------- pseudo-labeling ----------
def pseudo_labeling_improved(unlabeled_df: pd.DataFrame, train_df: pd.DataFrame,
                           target_count: int = 320, use_gpu: bool = True,
                           balance_method: str = "equal") -> pd.DataFrame:
    """
    Improved pseudo-labeling with confidence and perplexity
    balance_method: "equal", "original", "adaptive"
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            quantization_config=bnb_cfg if device == "cuda" else None,
            device_map="auto" if device == "cuda" else "cpu"
        )
    except Exception as e:
        print(f"GPU loading failed: {e}")
        print("Falling back to CPU...")
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    few_shots = create_few_shot_examples(train_df)
    instruction = DEFAULT_INSTRUCTION

    # Determine target counts based on balance method
    original_dist = train_df["label"].value_counts(normalize=True).sort_index()
    
    if balance_method == "original":
        total_target = target_count * 2
        target_0 = int(total_target * original_dist[0])
        target_1 = int(total_target * original_dist[1])
    elif balance_method == "equal":
        target_0 = target_count
        target_1 = target_count
    elif balance_method == "adaptive":
        minority_class = original_dist.idxmin()
        if minority_class == 0:
            target_0 = int(target_count * 1.5)
            target_1 = int(target_count * 0.5)
        else:
            target_0 = int(target_count * 0.5)
            target_1 = int(target_count * 1.5)
    
    print(f"Target counts - Label 0: {target_0}, Label 1: {target_1}")

    label0 = label1 = 0
    results = []
    pbar = tqdm(total=target_0 + target_1, desc="pseudo-labeling")

    unlabeled_shuffled = unlabeled_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    for idx, row in unlabeled_shuffled.iterrows():
        if label0 >= target_0 and label1 >= target_1:
            break
        
        sentence = row["sentence"]
        y, confidence, perplexity = predict_label_with_confidence(
            sentence, model, tokenizer, few_shots, instruction, device
        )
        
        if y is None:
            continue
        
        # Accept based on target counts
        if y == 0 and label0 < target_0:
            label0 += 1
            accepted = True
        elif y == 1 and label1 < target_1:
            label1 += 1
            accepted = True
        else:
            accepted = False
        
        if accepted:
            results.append({
                "sentence": sentence,
                "label": y,
                "confidence": confidence,
                "perplexity": perplexity,
                "original_index": idx
            })
            pbar.update(1)
        
        pbar.set_postfix(dict(label0=label0, label1=label1))
    
    pbar.close()
    
    del model
    torch.cuda.empty_cache()
    
    pseudo_df = pd.DataFrame(results)
    
    # Print statistics
    if len(pseudo_df) > 0:
        print(f"\nPseudo-labeling statistics:")
        print(f"Total generated: {len(pseudo_df)}")
        print(f"Label distribution: {dict(pseudo_df['label'].value_counts())}")
        print(f"Confidence - mean: {pseudo_df['confidence'].mean():.3f}, "
              f"std: {pseudo_df['confidence'].std():.3f}")
        print(f"Perplexity - mean: {pseudo_df['perplexity'].mean():.3f}, "
              f"std: {pseudo_df['perplexity'].std():.3f}")
    
    return pseudo_df

# ---------- save helpers ----------
def load_supplementary_data(supplementary_paths: List[str]) -> pd.DataFrame:
    """Load supplementary data from multiple CSV files for label 0 augmentation"""
    supplementary_data = []

    for path in supplementary_paths:
        if not os.path.exists(path):
            print(f"Warning: Supplementary file not found: {path}")
            continue

        try:
            # Detect encoding
            encoding = detect_encoding(path)
            df = pd.read_csv(path, encoding=encoding)

            # Handle different column formats
            if 'sentence' in df.columns and 'label' in df.columns:
                # Format: sentence, label
                clean_df = df[['sentence', 'label']].copy()
            elif '所見' in df.columns and '対応必要の有無' in df.columns:
                # Format: raw_s.csv style
                clean_df = pd.DataFrame()
                clean_df['sentence'] = df.apply(lambda r: combine_texts(r.get('所見', ''), r.get('診断', '')), axis=1)
                clean_df['label'] = df['対応必要の有無']
            else:
                print(f"Warning: Unknown column format in {path}, skipping")
                continue

            # Clean text and filter label 0 only
            clean_df['sentence'] = clean_df['sentence'].apply(clean_text_advanced)
            clean_df['sentence'] = clean_df['sentence'].apply(lambda x: mojimoji.han_to_zen(x, digit=False, ascii=False, kana=True) if not pd.isna(x) else "")

            # Keep only label 0 data and remove NaN labels
            label_0_data = clean_df[(clean_df['label'] == 0) & (clean_df['label'].notna())].copy()

            if len(label_0_data) > 0:
                supplementary_data.append(label_0_data)
                print(f"Loaded {len(label_0_data)} label 0 samples from {path}")

        except Exception as e:
            print(f"Error loading {path}: {e}")

    if supplementary_data:
        combined_supp = pd.concat(supplementary_data, ignore_index=True)
        # Remove duplicates
        combined_supp = combined_supp.drop_duplicates(subset=['sentence'], keep='first')
        print(f"Total supplementary label 0 samples: {len(combined_supp)}")
        return combined_supp
    else:
        return pd.DataFrame(columns=['sentence', 'label'])

def create_balanced_train_datasets(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                                 target_ratios: List[float] = [0.10, 0.05],
                                 supplementary_paths: List[str] = None,
                                 maintain_total_size: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Create training datasets with specific label '1' ratios while keeping val/test unchanged
    Can supplement label 0 data from external sources to maintain dataset size

    Args:
        train, val, test: Original datasets
        target_ratios: List of target ratios for label '1' in training data (e.g., [0.10, 0.05] for 1:10 and 1:20 ratios)
        supplementary_paths: List of paths to CSV files containing additional label 0 data
        maintain_total_size: If True, supplement with external label 0 data to maintain original total size

    Returns:
        Dictionary of balanced training datasets
    """
    balanced_datasets = {}

    # Load supplementary data if provided
    supplementary_label_0 = pd.DataFrame(columns=['sentence', 'label'])
    if supplementary_paths:
        supplementary_label_0 = load_supplementary_data(supplementary_paths)

    # Use training data for resampling
    train_label_1_data = train[train['label'] == 1]
    train_label_0_data = train[train['label'] == 0]

    print(f"Original training data - Label 1: {len(train_label_1_data)}, Label 0: {len(train_label_0_data)}")
    if len(supplementary_label_0) > 0:
        print(f"Supplementary label 0 data available: {len(supplementary_label_0)} samples")

    # Combine original and supplementary label 0 data
    all_label_0_data = pd.concat([train_label_0_data, supplementary_label_0], ignore_index=True)
    all_label_0_data = all_label_0_data.drop_duplicates(subset=['sentence'], keep='first')
    print(f"Total available label 0 data: {len(all_label_0_data)} samples")

    original_total_size = len(train)

    for ratio in target_ratios:
        ratio_name = f"ratio_{int(1/ratio)}"  # e.g., ratio_10 for 1:10 (10%), ratio_20 for 1:20 (5%)

        if maintain_total_size:
            # Maintain original total size: calculate required counts
            target_total_size = original_total_size
            target_label_1 = int(target_total_size * ratio)
            target_label_0 = target_total_size - target_label_1
        else:
            # Use ratio calculation based on available data
            max_label_0 = len(all_label_0_data)
            target_label_1 = int(max_label_0 * ratio / (1 - ratio))
            target_label_0 = max_label_0

        # Ensure we don't exceed available data
        if target_label_1 > len(train_label_1_data):
            target_label_1 = len(train_label_1_data)
            print(f"Warning: Not enough label 1 data for ratio {ratio}. Using all available ({target_label_1}).")

            # Recalculate label 0 needed if maintaining size
            if maintain_total_size:
                target_label_0 = original_total_size - target_label_1

        if target_label_0 > len(all_label_0_data):
            target_label_0 = len(all_label_0_data)
            print(f"Warning: Not enough label 0 data. Using all available ({target_label_0}).")

        # Sample the data
        sampled_label_1 = train_label_1_data.sample(n=target_label_1, random_state=RANDOM_SEED)
        sampled_label_0 = all_label_0_data.sample(n=target_label_0, random_state=RANDOM_SEED)

        # Combine and shuffle
        balanced_train = pd.concat([sampled_label_1, sampled_label_0], ignore_index=True)
        balanced_train = balanced_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        balanced_datasets[f"train_{ratio_name}"] = balanced_train

        # Print statistics
        actual_ratio = balanced_train['label'].value_counts(normalize=True).get(1, 0)
        label_0_from_supp = len(sampled_label_0) - len(sampled_label_0[sampled_label_0.index < len(train_label_0_data)])

        print(f"\n{ratio_name} (target ratio: {ratio:.1%}):")
        print(f"  Train: {len(balanced_train)} samples, actual label 1 ratio: {actual_ratio:.1%}")
        print(f"  Label 1: {len(sampled_label_1)} samples")
        print(f"  Label 0: {len(sampled_label_0)} samples ({len(sampled_label_0) - label_0_from_supp} original + {label_0_from_supp} supplementary)")
        print(f"  Val: {len(val)} samples (unchanged)")
        print(f"  Test: {len(test)} samples (unchanged)")

    return balanced_datasets

def save_datasets_comprehensive(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                               pseudo_df: Optional[pd.DataFrame], output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Save all datasets with metadata and return balanced datasets"""
    # Basic datasets
    train[["sentence", "label"]].to_csv(output_dir / "train.csv", index=False)
    val[["sentence", "label"]].to_csv(output_dir / "val.csv", index=False)
    test[["sentence", "label"]].to_csv(output_dir / "test.csv", index=False)
    
    # # Create balanced training datasets with specific label '1' ratios (1:10 and 1:20)
    # print("\nCreating balanced training datasets with specific label '1' ratios...")
    # balanced_train_datasets = create_balanced_train_datasets(train, val, test, target_ratios=[0.10, 0.05])
    
    # # Save balanced training datasets
    # for name, dataset in balanced_train_datasets.items():
    #     dataset[["sentence", "label"]].to_csv(output_dir / f"{name}.csv", index=False)
    
    # Pseudo-labeled data
    if pseudo_df is not None:
        # Save with confidence and perplexity
        pseudo_df.to_csv(output_dir / "pseudo_labeled.csv", index=False)
        
        # Extended training set
        train_extended = pd.concat([
            train[["sentence", "label"]], 
            pseudo_df[["sentence", "label"]]
        ], ignore_index=True)
        train_extended.to_csv(output_dir / "train_extended.csv", index=False)
        
        # High confidence version
        if "confidence" in pseudo_df.columns:
            high_conf_threshold = pseudo_df["confidence"].quantile(0.75)
            pseudo_high_conf = pseudo_df[pseudo_df["confidence"] >= high_conf_threshold]
            
            train_high_conf = pd.concat([
                train[["sentence", "label"]], 
                pseudo_high_conf[["sentence", "label"]]
            ], ignore_index=True)
            train_high_conf.to_csv(output_dir / "train_high_conf.csv", index=False)
            
            print(f"Created high-confidence training set (threshold={high_conf_threshold:.3f}) "
                  f"with {len(train_high_conf)} samples")
    
    # Metadata
    metadata = {
        "files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
            "train_ratio_10": "train_ratio_10.csv",  # 1:10 ratio (10%) training data
            "train_ratio_20": "train_ratio_20.csv",  # 1:20 ratio (5%) training data
            "train_ratio_50": "train_ratio_50.csv",  # 1:50 ratio (2%) training data
            "pseudo_labeled": "pseudo_labeled.csv" if pseudo_df is not None else None,
            "train_extended": "train_extended.csv" if pseudo_df is not None else None,
            "train_high_conf": "train_high_conf.csv" if pseudo_df is not None else None,
        },
        "preprocessing_steps": [
            "Text cleaning and normalization",
            "Number normalization to '0'",
            "Full-width character conversion",
            "Train/val/test split (70/10/20)",
            "Balanced training dataset creation with specific label ratios (1:10=10%, 1:20=5%, 1:50=2%)",
            "Pseudo-labeling with confidence and perplexity" if pseudo_df is not None else None
        ]
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    # # Return balanced datasets for statistics
    # return balanced_train_datasets

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="CT-report preprocessing with improved pseudo-labeling")
    ap.add_argument("--input", default="NLP_data/repearfile.csv")
    ap.add_argument("--output_dir", default="preprocessed_data")
    ap.add_argument("--pseudo-labeling", action="store_true")
    ap.add_argument("--target-count", type=int, default=700/2)
    ap.add_argument("--balance-method", choices=["equal", "original", "adaptive"], default="equal")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--check-quality", action="store_true")
    ap.add_argument("--remove-duplicates", action="store_true")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--supplementary-data", nargs="+", default=None,
                    help="Paths to CSV files containing additional label 0 data for balanced datasets")
    ap.add_argument("--maintain-size", action="store_true", default=True,
                    help="Maintain original training set size when creating balanced datasets")
    args = ap.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "preprocessing.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting preprocessing with args: {args}")
    
    # Load and preprocess
    df, has_doctor_label = load_and_preprocess_data(args.input)
    logging.info(f"Loaded {len(df)} samples")
    
    # If data has doctor labels, save as test data and exit
    if has_doctor_label:
        print("Processing data with doctor labels as test data...")
        
        # Quality check
        if args.check_quality:
            print("Performing quality check...")
            quality_report = check_data_quality(df)
            print(json.dumps(quality_report, indent=2, ensure_ascii=False))
            
            with open(output_dir / "quality_report.json", "w", encoding="utf-8") as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        # Remove duplicates if requested
        if args.remove_duplicates:
            original_size = len(df)
            df = df.drop_duplicates(subset=["sentence"])
            print(f"Removed {original_size - len(df)} duplicates")
        
        # Save as test data with doctor labels
        df.to_csv(output_dir / "test_with_doctor_labels.csv", index=False)
        print(f"Saved test data with doctor labels: {len(df)} samples")
        
        # Save doctor label mapping if available
        if "doctor_label_encoded" in df.columns:
            mapping_df = df[["doctor_label", "doctor_label_encoded"]].dropna().drop_duplicates()
            mapping_dict = dict(zip(mapping_df["doctor_label"], mapping_df["doctor_label_encoded"]))
            
            with open(output_dir / "doctor_label_mapping.json", "w", encoding="utf-8") as f:
                json.dump(mapping_dict, f, ensure_ascii=False, indent=2)
            print(f"Saved doctor label mapping: {len(mapping_dict)} labels")
        
        print("Processing complete for test data with doctor labels.")
        return
    
    # Quality check
    if args.check_quality:
        quality_report = check_data_quality(df)
        with open(output_dir / "quality_report.json", "w") as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info("Data quality report generated")
        if quality_report["issues"]:
            for issue in quality_report["issues"]:
                 logging.warning(f"Issue: {issue}")

    # Remove duplicates
    if args.remove_duplicates:
        original_len = len(df)
        df_labeled = df.dropna(subset=["label"])
        df_unlabeled = df[df["label"].isna()]
        
        df_labeled_dedup = df_labeled.drop_duplicates(subset=["sentence", "label"], keep="first")
        df_unlabeled_dedup = df_unlabeled.drop_duplicates(subset=["sentence"], keep="first")
        
        df = pd.concat([df_labeled_dedup, df_unlabeled_dedup], ignore_index=True)
        
        removed = original_len - len(df)
        if removed > 0:
            logging.info(f"Removed {removed} duplicate samples")
    
    # Split data
    train_df, val_df, test_df, unlabeled_df = split_data_improved(
        df, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )
    
    # Pseudo-labeling
    pseudo_df = None
    if args.pseudo_labeling and len(unlabeled_df) > 0:
        logging.info(f"Starting pseudo-labeling for {len(unlabeled_df)} unlabeled samples")
        
        pseudo_df = pseudo_labeling_improved(
            unlabeled_df, train_df,
            target_count=args.target_count,
            use_gpu=not args.no_gpu,
            balance_method=args.balance_method
        )
        
        logging.info(f"Generated {len(pseudo_df)} pseudo-labeled samples")
    
    # Save datasets
    save_datasets_comprehensive(train_df, val_df, test_df, pseudo_df, output_dir)
    
    # Create balanced training datasets for statistics
    # Check if supplementary data should be used
    supplementary_paths = []

    # Use command line specified supplementary data if provided
    if args.supplementary_data:
        supplementary_paths.extend(args.supplementary_data)
    else:
        # Default: check if pseudo_labeled_data.csv exists in current directory
        pseudo_labeled_path = "pseudo_labeled_data.csv"
        if os.path.exists(pseudo_labeled_path):
            supplementary_paths.append(pseudo_labeled_path)

    balanced_train_datasets = create_balanced_train_datasets(
        train_df, val_df, test_df,
        target_ratios=[0.1, 0.05, 0.02],
        supplementary_paths=supplementary_paths if supplementary_paths else None,
        maintain_total_size=args.maintain_size
    )
    # Save balanced training datasets
    for name, dataset in balanced_train_datasets.items():
        dataset[["sentence", "label"]].to_csv(output_dir / f"{name}.csv", index=False)
    
    # Save statistics
    stats = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "args": vars(args),
        "data_splits": {
            "train": {"count": len(train_df), "label_dist": dict(train_df["label"].value_counts())},
            "val": {"count": len(val_df), "label_dist": dict(val_df["label"].value_counts())},
            "test": {"count": len(test_df), "label_dist": dict(test_df["label"].value_counts())},
            "unlabeled": {"count": len(unlabeled_df)}
        },
        "balanced_training_datasets": {}
    }
    
    # Add balanced training dataset statistics
    for name, dataset in balanced_train_datasets.items():
        stats["balanced_training_datasets"][name] = {
            "count": len(dataset),
            "label_dist": dict(dataset["label"].value_counts()),
            "label_1_ratio": float(dataset["label"].value_counts(normalize=True).get(1, 0))
        }
    
    if pseudo_df is not None:
        stats["pseudo_labeled"] = {
            "count": len(pseudo_df),
            "label_dist": dict(pseudo_df["label"].value_counts()),
            "confidence_stats": {
                "mean": float(pseudo_df["confidence"].mean()),
                "std": float(pseudo_df["confidence"].std()),
                "min": float(pseudo_df["confidence"].min()),
                "max": float(pseudo_df["confidence"].max())
            },
            "perplexity_stats": {
                "mean": float(pseudo_df["perplexity"].mean()),
                "std": float(pseudo_df["perplexity"].std()),
                "min": float(pseudo_df["perplexity"].min()),
                "max": float(pseudo_df["perplexity"].max())
            }
        }
    
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"Preprocessing completed. Results saved to {output_dir}")
    
    # Summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original train set: {len(train_df)} samples")
    print(f"Original val set: {len(val_df)} samples")
    print(f"Original test set: {len(test_df)} samples")
    
    # Balanced training dataset summary
    print("\nBALANCED TRAINING DATASETS:")
    for name, dataset in balanced_train_datasets.items():
        label_1_ratio = dataset["label"].value_counts(normalize=True).get(1, 0)
        print(f"  {name}: {len(dataset)} samples, label 1 ratio: {label_1_ratio:.1%}")
    
    if pseudo_df is not None:
        print(f"\nPseudo-labeled: {len(pseudo_df)} samples")
        total_train = len(train_df) + len(pseudo_df)
        print(f"Total training data: {total_train} samples")
    print("="*50)


if __name__ == "__main__":
    main()