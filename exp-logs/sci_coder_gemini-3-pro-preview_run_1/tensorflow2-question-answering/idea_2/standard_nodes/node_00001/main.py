import os
import json
import torch
import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr

from library.config import Config
from library.utils import (
    load_glove_embeddings,
    compute_micro_f1,
    parse_annotation_record,
    tokenize,
    format_submission,
)
from library.data_loader import (
    build_vocab,
    get_long_answer_loader,
    process_short_answer_data,
)
from library.modeling import (
    DEConvNet,
    WindowLogisticRegressor,
    extract_short_answer_inference,
)
from library.training import train_ranker, train_extractor
from library.inference import generate_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_validation_and_analysis(la_model, sa_model, val_meta, vocab, device):
    """
    Runs inference on the validation set, computes metrics, and performs failure analysis.
    """
    print("\n--- Starting Validation ---")
    la_model.eval()
    sa_model.eval()

    # Create Validation Loader
    # We use the loader to get batches of (Q, Candidate) pairs efficiently
    val_loader = get_long_answer_loader(
        val_meta,
        Config.TRAIN_DATA_FILE,  # Validation data is in the train file
        vocab,
        split="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=True,
    )

    # 1. Rank Long Answers
    # Store best candidate per example: example_id -> (score, candidate_index)
    best_candidates = {}

    # Store features for failure analysis: example_id -> {feat_name: value}
    analysis_features = {}

    with torch.no_grad():
        for batch in val_loader:
            q = batch["question"].to(device)
            c = batch["candidate"].to(device)
            e_ids = batch["example_id"]
            c_idxs = batch["candidate_index"]

            # Forward pass
            scores = la_model(q, c).cpu().numpy()

            # Process batch
            for i, eid in enumerate(e_ids):
                score = float(scores[i])
                c_idx = int(c_idxs[i])

                # Initialize tracking for this example if new
                if eid not in best_candidates:
                    best_candidates[eid] = (-1.0, -1)
                    # Simple features: Question length
                    # Note: q[i] is padded, count non-pad tokens
                    q_len = (q[i] != 0).sum().item()
                    analysis_features[eid] = {
                        "q_len": q_len,
                        "num_candidates": 0,  # Will update
                    }

                # Update candidate count
                analysis_features[eid]["num_candidates"] += 1

                # Keep best score
                if score > best_candidates[eid][0]:
                    best_candidates[eid] = (score, c_idx)

    # 2. Extract Short Answers and Format Predictions
    val_predictions = {}

    # We need to read text for Short Answer extraction.
    # Iterate metadata to minimize file seeks.
    with open(Config.TRAIN_DATA_FILE, "rb") as f:
        for _, row in val_meta.iterrows():
            eid = row["example_id"]

            # Default empty
            val_predictions[eid] = {"long": "", "short": ""}

            if eid not in best_candidates:
                continue

            score, cand_idx = best_candidates[eid]

            # Thresholding
            if score < Config.LONG_ANSWER_THRESHOLD or cand_idx == -1:
                continue

            # Read JSON line
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line)
            except:
                continue

            candidates = data.get("long_answer_candidates", [])
            if cand_idx >= len(candidates):
                continue

            cand = candidates[cand_idx]
            start_token = cand["start_token"]
            end_token = cand["end_token"]

            # Long Answer String
            long_ans_str = f"{start_token}:{end_token}"

            # Short Answer Extraction
            doc_text = data.get("document_text", "")
            doc_tokens = tokenize(doc_text)
            q_text = data.get("question_text", "")
            q_tokens_set = set(tokenize(q_text.lower()))

            short_ans_str = extract_short_answer_inference(
                doc_tokens, q_tokens_set, start_token, end_token, sa_model
            )

            val_predictions[eid] = {"long": long_ans_str, "short": short_ans_str}

    # 3. Compute Metrics
    metrics = compute_micro_f1(val_predictions, val_meta)
    print(f"Final Validation Metric: {metrics['overall_f1']}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate per-instance error (1 - F1)
    # We need to compute F1 per instance. The compute_micro_f1 function aggregates.
    # We will do a simplified per-instance check:
    # Error = 1.0 if Long Answer F1 < 1.0 else 0.0 (Binary error for simplicity in correlation)
    # Or better: Error magnitude.

    errors = []
    feat_q_len = []
    feat_num_cand = []

    # Parse ground truth once
    gt_map = {}
    for _, row in val_meta.iterrows():
        gt_map[row["example_id"]] = parse_annotation_record(row["annotations"])

    for eid, pred in val_predictions.items():
        if eid not in gt_map or eid not in analysis_features:
            continue

        truth = gt_map[eid]
        p_long = pred["long"]
        t_long_set = truth["long"]

        # Calculate Instance F1 for Long Answer
        # Logic: If prediction matches ANY ground truth -> F1=1.0, else 0.0 (simplified for ranking)
        # If both empty -> 1.0
        if not p_long and not t_long_set:
            inst_f1 = 1.0
        elif p_long and t_long_set and p_long in t_long_set:
            inst_f1 = 1.0
        else:
            inst_f1 = 0.0

        error = 1.0 - inst_f1

        errors.append(error)
        feat_q_len.append(analysis_features[eid]["q_len"])
        feat_num_cand.append(analysis_features[eid]["num_candidates"])

    if len(errors) > 1:
        corr_q, _ = pearsonr(errors, feat_q_len)
        corr_c, _ = pearsonr(errors, feat_num_cand)

        print("Correlation between Error and Input Features:")
        print(f"  Question Length: {corr_q:.4f}")
        print(f"  Number of Candidates: {corr_c:.4f}")
    else:
        print("Not enough data for correlation analysis.")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override Config for fast baseline execution
    Config.TRAIN_SAMPLE_SIZE = 50000  # Limit training samples
    Config.NUM_EPOCHS = 3  # Reduce epochs
    Config.BATCH_SIZE = 32  # Safe batch size

    # 2. Data Loading
    print("Loading Metadata...")
    train_meta = pd.read_parquet(Config.TRAIN_META_FILE)
    val_meta = pd.read_parquet(Config.VAL_META_FILE)

    print("Building Vocabulary...")
    vocab = build_vocab(train_meta, load_cached_data=True)

    print("Loading Embeddings...")
    embedding_matrix = load_glove_embeddings(
        vocab, Config.EMBEDDING_DIM, load_cached_data=True
    )

    # 3. Training
    # --- Long Answer Model ---
    print("\n--- Training Long Answer Ranker ---")
    train_loader = get_long_answer_loader(
        train_meta,
        Config.TRAIN_DATA_FILE,
        vocab,
        split="train",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
    )
    # For validation during training, we use a subset to speed up epoch loops
    val_loader_subset = get_long_answer_loader(
        val_meta.head(5000),
        Config.TRAIN_DATA_FILE,
        vocab,
        split="val",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
    )

    la_model = DEConvNet(embedding_matrix)
    la_model = train_ranker(
        la_model,
        train_loader,
        val_loader_subset,
        val_meta.head(5000),
        epochs=Config.NUM_EPOCHS,
        device=device,
    )

    # --- Short Answer Model ---
    print("\n--- Training Short Answer Extractor ---")
    X, y = process_short_answer_data(
        train_meta, Config.TRAIN_DATA_FILE, vocab, load_cached_data=True
    )
    sa_model = train_extractor(
        X, y, epochs=10, device=device
    )  # Reduced epochs for speed

    # 4. Validation & Failure Analysis
    run_validation_and_analysis(la_model, sa_model, val_meta, vocab, device)

    # 5. Submission Inference
    print("\n--- Generating Submission ---")
    # generate_predictions handles loading models from disk, but we have them in memory.
    # However, to strictly follow the requirement of using library functions and
    # ensuring the pipeline works from saved artifacts, we call the library function.
    # The training functions saved the models to disk.
    generate_predictions(load_cached_data=True, batch_size=Config.BATCH_SIZE)


if __name__ == "__main__":
    main()
