import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import json
from collections import defaultdict

# Import from provided libraries
from library.utils import (
    ensure_dir,
    load_embeddings,
    load_jsonl_sample,
    get_dataset_partitions,
    CACHE_DIR,
    INPUT_DIR,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_EMBEDDING_DIM,
)
from library.models import CompareAggregateRanker, DilatedConvReader
from library.data_loader import get_tokenizer
from library.trainer import run_training_pipeline
from library.inference import InferencePipeline

# Configuration
SAMPLE_SIZE_TRAIN = 10000  # Limit training data for speed
EPOCHS = 2
BATCH_SIZE = 32
RANKER_THRESHOLD = 0.5
READER_THRESHOLD = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_environment():
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Ensure directories
    ensure_dir("./submission")


def train_models():
    print(">>> Starting Model Training...")
    # This function from library.trainer handles data processing and training
    # It saves 'ranker_best.pth' and 'reader_best.pth' to CACHE_DIR
    run_training_pipeline(
        sample_size=SAMPLE_SIZE_TRAIN,
        ranker_epochs=EPOCHS,
        reader_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        load_cached_data=True,  # Use cached if available to save time
    )
    print(">>> Training Complete.")


def compute_f1(predictions, ground_truths):
    """
    Computes Micro F1 score.
    predictions: list of strings (e.g., "start:end" or "")
    ground_truths: list of lists of strings (e.g., ["s1:e1", "s2:e2"] or [])
    """
    tp = 0
    fp = 0
    fn = 0

    for pred, gts in zip(predictions, ground_truths):
        # Normalize
        pred = pred.strip()
        gts = [g.strip() for g in gts]

        if pred == "":
            if not gts or (len(gts) == 1 and gts[0] == ""):
                # True Negative (Empty prediction, Empty GT)
                pass
            else:
                # GT exists, Pred empty -> FN
                fn += 1
        else:
            if pred in gts:
                tp += 1
            else:
                fp += 1
                # If GT was not empty, we also missed the correct one (FN)
                # However, standard Micro F1 usually counts:
                # P = TP / (TP + FP)
                # R = TP / (TP + FN)
                # If we predicted wrong, it's an FP.
                # If there was a target we missed, it's an FN.
                if gts and not (len(gts) == 1 and gts[0] == ""):
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return f1


def validate_and_analyze():
    print(">>> Starting Validation and Failure Analysis...")

    # 1. Load Resources
    _, val_meta, _ = get_dataset_partitions()

    # Load tokenizer and embeddings
    tokenizer = get_tokenizer(
        val_meta, load_cached_data=True
    )  # Should be cached from training
    embedding_matrix = load_embeddings(tokenizer.word_index, load_cached_data=True)

    # Load Models
    ranker = CompareAggregateRanker(embedding_matrix).to(DEVICE)
    reader = DilatedConvReader(embedding_matrix).to(DEVICE)

    ranker_path = os.path.join(CACHE_DIR, "ranker_best.pth")
    reader_path = os.path.join(CACHE_DIR, "reader_best.pth")

    if os.path.exists(ranker_path):
        ranker.load_state_dict(torch.load(ranker_path, map_location=DEVICE))
    else:
        print("Warning: Ranker model not found, using random weights.")

    if os.path.exists(reader_path):
        reader.load_state_dict(torch.load(reader_path, map_location=DEVICE))
    else:
        print("Warning: Reader model not found, using random weights.")

    ranker.eval()
    reader.eval()

    # 2. Validation Loop
    results = []  # Stores dicts for failure analysis

    all_preds_long = []
    all_gts_long = []
    all_preds_short = []
    all_gts_short = []

    print(f"Validating on {len(val_meta)} examples...")

    count = 0

    with torch.no_grad():
        for _, row in val_meta.iterrows():
            file_path = os.path.join(INPUT_DIR, row["file_path"])
            data = load_jsonl_sample(file_path, row["byte_offset"])
            if not data:
                continue

            # Ground Truth Extraction
            gt_longs = []
            gt_shorts = []
            annotations = data.get("annotations", [])
            for ann in annotations:
                la = ann["long_answer"]
                if la["start_token"] != -1:
                    gt_longs.append(f"{la['start_token']}:{la['end_token']}")

                sas = ann["short_answers"]
                if sas:
                    for sa in sas:
                        gt_shorts.append(f"{sa['start_token']}:{sa['end_token']}")
                elif ann["yes_no_answer"] != "NONE":
                    gt_shorts.append(ann["yes_no_answer"])

            # Prediction Logic
            question_text = data["question_text"]
            document_text = data["document_text"]
            doc_tokens = document_text.split()

            q_seq = tokenizer.texts_to_sequences([question_text])[0]

            candidates = data["long_answer_candidates"]
            cand_seqs = []
            cand_spans = []

            for cand in candidates:
                if cand["top_level"]:
                    start = cand["start_token"]
                    end = cand["end_token"]
                    if start < len(doc_tokens):
                        cand_text = " ".join(doc_tokens[start:end])
                        c_seq = tokenizer.texts_to_sequences([cand_text])[0]
                        cand_seqs.append(c_seq)
                        cand_spans.append((start, end))

            pred_long = ""
            pred_short = ""
            ranker_score = 0.0
            reader_score = 0.0

            if cand_seqs:
                # Batch processing for Ranker candidates
                num_cands = len(cand_seqs)
                max_cand_len = min(max(len(c) for c in cand_seqs), DEFAULT_MAX_SEQ_LEN)

                q_tensor = torch.tensor([q_seq] * num_cands, dtype=torch.long).to(
                    DEVICE
                )
                p_tensor = torch.zeros((num_cands, max_cand_len), dtype=torch.long).to(
                    DEVICE
                )

                for j, c in enumerate(cand_seqs):
                    l = min(len(c), max_cand_len)
                    p_tensor[j, :l] = torch.tensor(c[:l], dtype=torch.long)

                ranker_logits = ranker(q_tensor, p_tensor).squeeze(1)
                ranker_probs = torch.sigmoid(ranker_logits)

                best_idx = torch.argmax(ranker_probs).item()
                ranker_score = ranker_probs[best_idx].item()

                # Reader
                best_p_ids = cand_seqs[best_idx]
                input_seq = q_seq + best_p_ids
                if len(input_seq) > DEFAULT_MAX_SEQ_LEN:
                    input_seq = input_seq[:DEFAULT_MAX_SEQ_LEN]

                input_tensor = torch.tensor([input_seq], dtype=torch.long).to(DEVICE)
                s_logits, e_logits = reader(input_tensor)
                s_probs = torch.softmax(s_logits, dim=1).squeeze(0)
                e_probs = torch.softmax(e_logits, dim=1).squeeze(0)

                # Span Selection
                p_start_offset = len(q_seq)
                p_end_offset = len(input_seq)

                valid_s = s_probs[p_start_offset:p_end_offset]
                valid_e = e_probs[p_start_offset:p_end_offset]

                best_span_score = 0.0
                best_span = None

                if len(valid_s) > 0:
                    top_k = min(5, len(valid_s))
                    top_s = torch.topk(valid_s, k=top_k).indices
                    top_e = torch.topk(valid_e, k=top_k).indices

                    for s_rel in top_s:
                        for e_rel in top_e:
                            if s_rel <= e_rel:
                                score = (valid_s[s_rel] * valid_e[e_rel]).item()
                                if score > best_span_score:
                                    best_span_score = score
                                    best_span = (s_rel.item(), e_rel.item())

                reader_score = best_span_score

                if ranker_score > RANKER_THRESHOLD:
                    la_s, la_e = cand_spans[best_idx]
                    pred_long = f"{la_s}:{la_e}"

                    if best_span and reader_score > READER_THRESHOLD:
                        rel_s, rel_e = best_span
                        sa_s = la_s + rel_s
                        sa_e = la_s + rel_e + 1  # Exclusive end
                        pred_short = f"{sa_s}:{sa_e}"

            # Collect results
            all_preds_long.append(pred_long)
            all_gts_long.append(gt_longs)
            all_preds_short.append(pred_short)
            all_gts_short.append(gt_shorts)

            # Failure Analysis Data
            # Error = 1 if mismatch, 0 if match
            long_match = (pred_long == "" and not gt_longs) or (pred_long in gt_longs)
            long_error = 0.0 if long_match else 1.0

            results.append(
                {
                    "doc_len": len(doc_tokens),
                    "q_len": len(question_text.split()),
                    "ranker_conf": ranker_score,
                    "reader_conf": reader_score,
                    "long_error": long_error,
                }
            )

            count += 1

    # 3. Compute Metrics
    global_preds = all_preds_long + all_preds_short
    global_gts = all_gts_long + all_gts_short

    final_micro_f1 = compute_f1(global_preds, global_gts)

    print(f"Final Validation Metric: {final_micro_f1}")

    # 4. Failure Analysis
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print("\nFailure Analysis (Correlation with Long Answer Error):")
        corrs = df_res.corr()["long_error"].drop("long_error")
        print(corrs)
    else:
        print("No validation results to analyze.")


def generate_submission():
    print(">>> Generating Submission...")
    pipeline = InferencePipeline(submission_dir="./submission")
    pipeline.generate_submission(
        ranker_threshold=RANKER_THRESHOLD,
        reader_threshold=READER_THRESHOLD,
        load_cached_data=True,
    )
    print(">>> Submission Generated.")


def main():
    setup_environment()

    # 1. Train
    train_models()

    # 2. Validate & Analyze
    validate_and_analyze()

    # 3. Submit
    generate_submission()


if __name__ == "__main__":
    main()
