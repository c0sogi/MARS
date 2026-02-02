import os
import json
import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import f1_score
from library.config import Config
from library.text_utils import (
    build_or_load_tokenizer,
    build_or_load_embedding_matrix,
    segment_sentences,
)
from library.data_loader import get_data_loader
from library.network import SentenceFactorizedModel
from library.train_engine import TrainEngine
from library.predictor import predict_answers


# --- 1. Configuration & Setup ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Override Config for Fast Baseline
Config.NUM_EPOCHS = 3
Config.BATCH_SIZE = 32
TRAIN_SAMPLE_SIZE = 20000
VAL_SAMPLE_SIZE = 5000

set_seed(Config.RANDOM_SEED)

print("--- Starting Runfile ---")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# --- 2. Data Preparation ---
# Tokenizer
tokenizer = build_or_load_tokenizer(load_cached_data=True)

# Embeddings
embedding_matrix = build_or_load_embedding_matrix(tokenizer, load_cached_data=True)

# DataLoaders
# Cite debug_lesson_2: Set load_cached_data=False to force regeneration of data with correct sample size
print("Preparing DataLoaders...")
train_loader = get_data_loader(
    mode="train",
    tokenizer=tokenizer,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    sample_size=TRAIN_SAMPLE_SIZE,
    load_cached_data=False,
)

val_loader = get_data_loader(
    mode="val",
    tokenizer=tokenizer,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    sample_size=VAL_SAMPLE_SIZE,
    load_cached_data=False,
)

# --- 3. Model Initialization ---
print("Initializing Model...")
model = SentenceFactorizedModel(
    vocab_size=tokenizer.vocab_size, embedding_matrix=embedding_matrix
)

# --- 4. Training ---
print("Starting Training...")
engine = TrainEngine(model)
engine.train(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

# --- 5. Validation & Metric Calculation ---
print("--- Starting Detailed Validation ---")


def load_validation_ground_truth(meta_path, raw_path, sample_size):
    """
    Loads ground truth annotations for the validation set.
    """
    meta_df = pd.read_csv(meta_path)
    valid_ids = set(meta_df["example_id"].astype(str))

    ground_truth = {}
    count = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ex_id = str(entry["example_id"])

            if ex_id not in valid_ids:
                continue

            if count >= sample_size:
                break

            # Extract GT
            anns = entry.get("annotations", [])
            gt_long = None
            gt_shorts = []
            gt_yes_no = "NONE"

            if anns:
                ann = anns[0]
                # Long
                la = ann.get("long_answer", {})
                if la.get("candidate_index", -1) != -1:
                    # We need the token span
                    c_idx = la["candidate_index"]
                    candidates = entry.get("long_answer_candidates", [])
                    if c_idx < len(candidates):
                        cand = candidates[c_idx]
                        gt_long = (cand["start_token"], cand["end_token"])

                # Short
                sas = ann.get("short_answers", [])
                for sa in sas:
                    gt_shorts.append((sa["start_token"], sa["end_token"]))

                # Yes/No
                gt_yes_no = ann.get("yes_no_answer", "NONE")

            # Extract Raw Candidates and Sentences for mapping predictions
            candidates = entry.get("long_answer_candidates", [])
            simple_candidates = [
                {"start_token": c["start_token"], "end_token": c["end_token"]}
                for c in candidates
            ]

            doc_text = entry.get("document_text", "")
            sentences = segment_sentences(doc_text)
            if len(sentences) > Config.MAX_SENTS_PER_DOC:
                sentences = sentences[: Config.MAX_SENTS_PER_DOC]

            simple_sentences = [
                {
                    "start_token_idx": s["start_token_idx"],
                    "end_token_idx": s["end_token_idx"],
                }
                for s in sentences
            ]

            ground_truth[ex_id] = {
                "gt_long": gt_long,
                "gt_shorts": gt_shorts,
                "gt_yes_no": gt_yes_no,
                "candidates": simple_candidates,
                "sentences": simple_sentences,
                "doc_len": len(doc_text.split()),  # Metadata for failure analysis
                "q_len": len(entry.get("question_text", "").split()),
            }

            count += 1

    return ground_truth


# Load GT
print("Loading validation ground truth...")
val_gt_data = load_validation_ground_truth(
    Config.VAL_META_PATH, Config.TRAIN_DATA_PATH, VAL_SAMPLE_SIZE
)

# Inference on Validation Set
model.eval()
tp = 0
fp = 0
fn = 0

# For failure analysis
analysis_data = []

yn_map = {0: "NONE", 1: "YES", 2: "NO"}

with torch.no_grad():
    for batch in val_loader:
        questions = batch["questions"].to(engine.device)
        sentences = batch["sentences"].to(engine.device)
        doc_lengths = batch["doc_lengths"]
        example_ids = batch["example_ids"]
        candidate_maps = batch["candidate_maps"]

        scores, yn_logits = model(questions, sentences, doc_lengths)
        scores_split = torch.split(scores, doc_lengths)

        for i, ex_id in enumerate(example_ids):
            if ex_id not in val_gt_data:
                continue

            gt_info = val_gt_data[ex_id]
            doc_scores = scores_split[i]
            doc_cand_map = candidate_maps[i]
            doc_yn_logits = yn_logits[i]

            raw_candidates = gt_info["candidates"]
            raw_sentences = gt_info["sentences"]

            # --- Prediction Logic (Same as Predictor) ---
            best_cand_idx = -1
            best_cand_score = -1.0
            best_sent_idx_in_doc = -1

            for c_idx, sent_indices in enumerate(doc_cand_map):
                if not sent_indices:
                    continue
                valid_indices = [idx for idx in sent_indices if idx < len(doc_scores)]
                if not valid_indices:
                    continue

                cand_sent_scores = doc_scores[valid_indices]
                max_score_val, max_score_arg = torch.max(cand_sent_scores, dim=0)
                max_score = max_score_val.item()

                if max_score > best_cand_score:
                    best_cand_score = max_score
                    best_cand_idx = c_idx
                    best_sent_idx_in_doc = valid_indices[max_score_arg.item()]

            pred_long = None
            pred_short = None  # Can be span tuple or string "YES"/"NO"

            if best_cand_score >= Config.CONFIDENCE_THRESHOLD and best_cand_idx != -1:
                # Long Pred
                c_start = raw_candidates[best_cand_idx]["start_token"]
                c_end = raw_candidates[best_cand_idx]["end_token"]
                pred_long = (c_start, c_end)

                # Short Pred
                yn_pred_idx = torch.argmax(doc_yn_logits).item()
                yn_str = yn_map.get(yn_pred_idx, "NONE")

                if yn_str != "NONE":
                    pred_short = yn_str
                elif best_sent_idx_in_doc != -1:
                    s_start = raw_sentences[best_sent_idx_in_doc]["start_token_idx"]
                    s_end = raw_sentences[best_sent_idx_in_doc]["end_token_idx"]
                    pred_short = (s_start, s_end)

            # --- Evaluation Logic (Micro F1) ---
            # 1. Long Answer Evaluation
            gt_long = gt_info["gt_long"]
            is_long_correct = False

            if pred_long is not None:
                if gt_long is not None and pred_long == gt_long:
                    tp += 1
                    is_long_correct = True
                else:
                    fp += 1
            else:
                if gt_long is not None:
                    fn += 1
                # Else: True Negative (ignored in F1)

            # 2. Short Answer Evaluation
            gt_shorts = gt_info["gt_shorts"]
            gt_yes_no = gt_info["gt_yes_no"]
            is_short_correct = False

            # Determine if GT has short answer
            has_gt_short = (len(gt_shorts) > 0) or (gt_yes_no != "NONE")

            if pred_short is not None:
                match = False
                # Check Yes/No match
                if isinstance(pred_short, str):
                    if pred_short == gt_yes_no:
                        match = True
                # Check Span match
                elif isinstance(pred_short, tuple):
                    if any(pred_short == s for s in gt_shorts):
                        match = True

                if match:
                    tp += 1
                    is_short_correct = True
                else:
                    fp += 1
            else:
                if has_gt_short:
                    fn += 1

            # --- Failure Analysis Data ---
            # Compute local F1 for this example (approximate for correlation)
            # Local F1 = 2*correct / (2*correct + error_count)
            # errors = (1 if long wrong else 0) + (1 if short wrong else 0) ... simplified
            # Let's just use "All Correct" as a binary proxy or average accuracy
            example_acc = (int(is_long_correct) + int(is_short_correct)) / 2.0

            analysis_data.append(
                {
                    "doc_len": gt_info["doc_len"],
                    "q_len": gt_info["q_len"],
                    "accuracy": example_acc,
                    "error": 1.0 - example_acc,
                }
            )

# Calculate Final Metric
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Final Validation Metric: {f1}")

# --- 6. Failure Analysis ---
print("\n--- Failure Analysis ---")
df_analysis = pd.DataFrame(analysis_data)
if not df_analysis.empty:
    corr_doc = df_analysis["error"].corr(df_analysis["doc_len"])
    corr_q = df_analysis["error"].corr(df_analysis["q_len"])

    print("Correlation between Error and Input Features:")
    print(f"  Document Length: {corr_doc:.4f}")
    print(f"  Question Length: {corr_q:.4f}")
else:
    print("No analysis data available.")

# --- 7. Submission ---
print("\n--- Generating Submission ---")
# Predict on Test Set
predict_answers(load_cached_data=True, sample_size=None)
