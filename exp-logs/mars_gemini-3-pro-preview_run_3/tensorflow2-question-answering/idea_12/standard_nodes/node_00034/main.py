import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import library modules
from library.config import Config
from library.text_utils import build_or_load_vocab, Tokenizer, parse_candidates
from library.trainer import train_ranker, train_reader
from library.evaluator import predict_submission
from library.ranker_net import InteractionRanker, prepare_ranker_data, RankerDataset
from library.reader_net import UNetReader, prepare_reader_test_data, ReaderDataset


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def validate_and_analyze(vocab):
    """
    Runs the full inference pipeline on the validation set, computes metrics,
    and performs failure analysis.
    """
    print("\n--- Starting Validation & Failure Analysis ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Validation Metadata
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    # Subsample for speed if needed, but requirements say "entire hold-out validation set"
    # We will use the full validation set defined in metadata (approx 55k samples).
    # However, to ensure this runs within 2 hours alongside training, we might need to be careful.
    # The config has VAL_SAMPLE_SIZE. We will respect that if set, but ideally we run on full.
    # Given the constraints, we'll use Config.VAL_SAMPLE_SIZE if it's small, else we might limit it
    # to ensure completion. Let's stick to Config.VAL_SAMPLE_SIZE for safety in this baseline.
    if Config.VAL_SAMPLE_SIZE and len(val_meta) > Config.VAL_SAMPLE_SIZE:
        print(
            f"Subsampling validation set to {Config.VAL_SAMPLE_SIZE} samples for speed."
        )
        val_meta = val_meta.sample(n=Config.VAL_SAMPLE_SIZE, random_state=Config.SEED)

    # 1. Ranker Inference on Validation
    # We need to rank candidates for each validation question
    print("Preparing Ranker Validation Data...")
    # We use prepare_ranker_data with is_train=False to get all candidates, not just pos/neg pairs
    ranker_val_df = prepare_ranker_data(
        val_meta,
        vocab,
        Config.TRAIN_RAW_FILE,  # Read from train file where val data lives
        is_train=False,  # Inference mode to get all candidates
        load_cached_data=False,  # Force regeneration to ensure we have inference format
        cache_path=None,
    )

    if ranker_val_df.empty:
        print("No validation data found.")
        return

    # Load Ranker Model
    ranker_model = InteractionRanker(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_embeddings=vocab.embedding_matrix,
    ).to(device)
    ranker_model.load_state_dict(
        torch.load(Config.RANKER_MODEL_PATH, map_location=device)
    )
    ranker_model.eval()

    # Predict Ranker Scores
    ranker_dataset = RankerDataset(ranker_val_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)
    ranker_loader = DataLoader(
        ranker_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False
    )

    print("Running Ranker Inference...")
    all_scores = []
    with torch.no_grad():
        for batch in ranker_loader:
            q_ids = batch["q_ids"].to(device)
            p_ids = batch["p_ids"].to(device)
            outputs = ranker_model(q_ids, p_ids)
            all_scores.extend(outputs.cpu().tolist())

    ranker_val_df["rank_score"] = all_scores

    # Select Top Candidate per Example
    best_candidates = ranker_val_df.loc[
        ranker_val_df.groupby("example_id")["rank_score"].idxmax()
    ].reset_index(drop=True)

    # 2. Reader Inference on Validation
    print("Preparing Reader Validation Data...")

    # Manually populate since prepare_reader_test_data expects a path usually
    # But we can reuse the logic:
    reader_input_rows = []
    for _, row in best_candidates.iterrows():
        # q_ids and p_ids are lists
        q_ids = list(row["q_ids"])
        p_ids = list(row["p_ids"])
        input_ids = q_ids + p_ids
        reader_input_rows.append(
            {
                "example_id": row["example_id"],
                "input_ids": input_ids,
                "doc_start_token": row["start_token"],
                "q_len": len(q_ids),
                "rank_score": row["rank_score"],
                "candidate_text": row["candidate_text"],
            }
        )
    reader_val_df = pd.DataFrame(reader_input_rows)

    # Add dummy targets for Dataset
    reader_val_df["start_token"] = 0
    reader_val_df["end_token"] = 0

    # Load Reader Model
    reader_model = UNetReader(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_embeddings=vocab.embedding_matrix,
    ).to(device)
    reader_model.load_state_dict(
        torch.load(Config.READER_MODEL_PATH, map_location=device)
    )
    reader_model.eval()

    reader_dataset = ReaderDataset(reader_val_df, Config.Q_MAX_LEN + Config.P_MAX_LEN)
    reader_loader = DataLoader(
        reader_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    print("Running Reader Inference...")

    # Metrics Accumulators
    tp_long, fp_long, fn_long = 0, 0, 0
    tp_short, fp_short, fn_short = 0, 0, 0

    # For Failure Analysis
    analysis_data = []  # Stores (q_len, doc_len, error_bool)

    # Load Ground Truth efficiently
    # We need to look up annotations for each example_id
    # To avoid re-reading the file randomly, we can read the file linearly if sorted,
    # or just seek. Since we have offsets in val_meta, we can use that.
    # Map example_id to offset
    id_to_offset = dict(zip(val_meta["example_id"], val_meta["byte_offset"]))

    batch_idx = 0
    with torch.no_grad():
        for batch in reader_loader:
            input_ids = batch["input_ids"].to(device)
            start_logits, end_logits = reader_model(input_ids)

            start_probs = F.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = F.softmax(end_logits, dim=1).cpu().numpy()

            current_batch_size = input_ids.size(0)
            for i in range(current_batch_size):
                global_idx = batch_idx * Config.BATCH_SIZE + i
                row = reader_val_df.iloc[global_idx]
                eid = row["example_id"]

                # Get Ground Truth
                offset = id_to_offset.get(eid)
                if offset is None:
                    continue

                with open(Config.TRAIN_RAW_FILE, "rb") as f:
                    f.seek(offset)
                    record = json.loads(f.readline().decode("utf-8"))

                annotations = record.get("annotations", [])
                gt_long_spans = []
                gt_short_spans = []  # List of lists of spans
                gt_yes_no = "NONE"

                for ann in annotations:
                    la = ann["long_answer"]
                    if la["start_token"] != -1:
                        gt_long_spans.append((la["start_token"], la["end_token"]))

                    shorts = ann.get("short_answers", [])
                    if shorts:
                        gt_short_spans.append(
                            [(s["start_token"], s["end_token"]) for s in shorts]
                        )

                    if ann["yes_no_answer"] != "NONE":
                        gt_yes_no = ann["yes_no_answer"]

                # Prediction Logic
                s_prob = start_probs[i]
                e_prob = end_probs[i]

                best_score = -1
                best_span = (0, 0)

                # Simple heuristic search
                top_starts = np.argsort(s_prob)[-10:]
                top_ends = np.argsort(e_prob)[-10:]

                for s in top_starts:
                    for e in top_ends:
                        if s <= e and (e - s) < 30:
                            score = s_prob[s] * e_prob[e]
                            if score > best_score:
                                best_score = score
                                best_span = (s, e)

                q_len = row["q_len"]
                doc_offset = row["doc_start_token"]
                rank_score = row["rank_score"]

                pred_long_span = None
                pred_short_span = None
                pred_yes_no = "NONE"  # Not implemented in this baseline reader

                # Thresholding
                has_pred = False

                # Long Answer Logic
                # If ranker score is high enough, we predict the candidate as long answer
                if rank_score > Config.LONG_ANSWER_THRESHOLD:
                    # The candidate text provided by ranker
                    # We need the end token. We have doc_start_token.
                    # We can approximate length from input_ids (minus question)
                    p_len = len(row["input_ids"]) - q_len
                    # Refine: remove padding if any.
                    # Ideally we should have stored exact end token in ranker output.
                    # We will use the length of tokens in p_ids (non-zero)
                    # p_ids was padded in dataset, but here row['input_ids'] is from ranker output which was raw ids?
                    # prepare_reader_test_data concatenates raw lists. So len is accurate.
                    pred_long_span = (doc_offset, doc_offset + p_len)

                # Short Answer Logic
                # If we have a long answer AND reader score is high
                if pred_long_span and best_score > Config.SHORT_ANSWER_THRESHOLD:
                    pred_s, pred_e = best_span
                    if pred_s >= q_len:  # Must be in paragraph
                        abs_s = doc_offset + (pred_s - q_len)
                        abs_e = (
                            doc_offset + (pred_e - q_len) + 1
                        )  # +1 for exclusive? NQ uses byte/token offsets.
                        # Usually annotations are start:end where end is exclusive.
                        pred_short_span = (abs_s, abs_e)
                        has_pred = True

                # --- Evaluation ---

                # Long Answer Eval
                # Match if pred span matches ANY gt long span
                long_match = False
                if pred_long_span:
                    if pred_long_span in gt_long_spans:
                        long_match = True
                        tp_long += 1
                    else:
                        fp_long += 1
                else:
                    if gt_long_spans:
                        fn_long += 1
                    # else: True Negative (not counted in F1)

                # Short Answer Eval
                # Match if pred span matches ANY span in ANY valid short answer group
                # OR if yes/no matches
                short_match = False
                if has_pred:
                    # Check Yes/No (Not implemented prediction, so assumes NONE)
                    # Check Spans
                    if pred_short_span:
                        found = False
                        for group in gt_short_spans:
                            if pred_short_span in group:
                                found = True
                                break
                        if found:
                            short_match = True
                            tp_short += 1
                        else:
                            fp_short += 1
                else:
                    if gt_short_spans or gt_yes_no != "NONE":
                        fn_short += 1

                # Failure Analysis Data
                # Error if either long or short was wrong (when they should have been right)
                # or predicted when shouldn't have.
                is_error = False
                if pred_long_span and not long_match:
                    is_error = True
                if not pred_long_span and gt_long_spans:
                    is_error = True
                if has_pred and not short_match:
                    is_error = True
                if not has_pred and (gt_short_spans or gt_yes_no != "NONE"):
                    is_error = True

                # Features for correlation
                # Question Length (tokens)
                # Document Length (approximate, we don't have full doc len here easily, use candidate len)
                cand_len = len(row["input_ids"]) - q_len

                analysis_data.append(
                    {
                        "q_len": q_len,
                        "cand_len": cand_len,
                        "error": 1 if is_error else 0,
                    }
                )

            batch_idx += 1

    # Calculate Metrics
    f1_long = compute_f1(tp_long, fp_long, fn_long)
    f1_short = compute_f1(tp_short, fp_short, fn_short)

    # Official metric is Micro F1 over all predictions (Long + Short + YesNo)
    # We treat Long and Short predictions as separate instances for Micro F1 calculation logic in NQ?
    # Usually NQ metric is average of F1 per example, or micro over all.
    # The Task Description says "Metric: Micro F1".
    # We sum TPs, FPs, FNs across both tasks.

    total_tp = tp_long + tp_short
    total_fp = fp_long + fp_short
    total_fn = fn_long + fn_short

    final_f1 = compute_f1(total_tp, total_fp, total_fn)

    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    analysis_df = pd.DataFrame(analysis_data)
    if not analysis_df.empty:
        corr_q = analysis_df["q_len"].corr(analysis_df["error"])
        corr_c = analysis_df["cand_len"].corr(analysis_df["error"])

        print("Correlation between Error and Input Features:")
        print(f"Question Length: {corr_q:.4f}")
        print(f"Candidate Length: {corr_c:.4f}")

        if abs(corr_q) > 0.1:
            print(
                "Observation: Question length has a notable correlation with error rate."
            )
        if abs(corr_c) > 0.1:
            print(
                "Observation: Candidate/Paragraph length has a notable correlation with error rate."
            )
    else:
        print("No analysis data available.")


def main():
    # 1. Setup
    Config.setup_directories()
    set_seed(Config.SEED)

    # 2. Build/Load Vocab
    print("Initializing Vocabulary...")
    # Load train metadata to build vocab if needed
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    vocab = build_or_load_vocab(train_meta, load_cached_data=True)

    # 3. Train Ranker
    # Use small sample size for baseline speed
    print("\n--- Training Ranker ---")
    train_ranker(
        epochs=1,  # Baseline speed
        batch_size=Config.BATCH_SIZE,
        train_sample_size=5000,  # Fast training
        val_sample_size=1000,
    )

    # 4. Train Reader
    print("\n--- Training Reader ---")
    train_reader(
        epochs=1,  # Baseline speed
        batch_size=Config.BATCH_SIZE,
        train_sample_size=5000,  # Fast training
        val_sample_size=1000,
    )

    # 5. Validation & Failure Analysis
    validate_and_analyze(vocab)

    # 6. Submission
    print("\n--- Generating Submission ---")
    predict_submission(
        load_cached_data=False
    )  # Force fresh prediction using trained models


if __name__ == "__main__":
    main()
