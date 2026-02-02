import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import sys

# Import from provided library
from library.config import Config
from library.preprocessing import Tokenizer, build_embedding_matrix
from library.dataset import NQDataset
from library.model import InteractionGridCNN
from library.engine import Trainer, set_seed
from library.utils import load_ground_truth_data, compute_f1_score, load_jsonl


def generate_submission_dataframe(model, dataset, device, threshold=0.5):
    """
    Runs inference on an expanded dataset (Q, Candidate pairs), aggregates results
    by example_id, selects the best candidate, and formats the submission strings.
    """
    model.eval()
    # num_workers=0 to avoid multiprocessing overhead in short runs, or 2 for speed
    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Structure: { example_id: [ {score, c_start, c_end, rel_s, rel_e, yn}, ... ] }
    grouped_preds = {}

    with torch.no_grad():
        for batch in loader:
            q_ids = batch["q_ids"].to(device)
            c_ids = batch["c_ids"].to(device)

            outputs = model(q_ids, c_ids)

            # Get probabilities
            rank_probs = torch.sigmoid(outputs["rank_logits"]).cpu().numpy()
            start_probs = torch.softmax(outputs["start_logits"], dim=1).cpu().numpy()
            end_probs = torch.softmax(outputs["end_logits"], dim=1).cpu().numpy()
            yn_probs = torch.softmax(outputs["yn_logits"], dim=1).cpu().numpy()

            eids = batch["example_id"]
            c_starts = batch["token_start"].numpy()
            c_ends = batch["token_end"].numpy()

            for i, eid in enumerate(eids):
                if eid not in grouped_preds:
                    grouped_preds[eid] = []

                # Get best span indices relative to candidate
                s_idx = np.argmax(start_probs[i])
                e_idx = np.argmax(end_probs[i])

                # Get Yes/No prediction
                yn_idx = np.argmax(yn_probs[i])
                yn_str = ["NONE", "YES", "NO"][yn_idx]

                grouped_preds[eid].append(
                    {
                        "score": rank_probs[i],
                        "c_start": c_starts[i],
                        "c_end": c_ends[i],
                        "rel_s": s_idx,
                        "rel_e": e_idx,
                        "yn": yn_str,
                    }
                )

    # Formulate final predictions
    submission_rows = []

    # Iterate over all unique IDs found
    for eid, candidates in grouped_preds.items():
        # Sort candidates by ranking score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        long_pred_str = ""
        short_pred_str = ""

        # Apply threshold for Long Answer
        if best["score"] > threshold:
            long_pred_str = f"{best['c_start']}:{best['c_end']}"

            # Determine Short Answer
            if best["yn"] in ["YES", "NO"]:
                short_pred_str = best["yn"]
            else:
                # Calculate absolute span indices
                abs_s = best["c_start"] + best["rel_s"]
                abs_e = best["c_start"] + best["rel_e"]

                # Validate span logic (start < end and within candidate bounds)
                if abs_s < abs_e and abs_e <= best["c_end"]:
                    short_pred_str = f"{abs_s}:{abs_e}"

        submission_rows.append(
            {"example_id": f"{eid}_long", "PredictionString": long_pred_str}
        )
        submission_rows.append(
            {"example_id": f"{eid}_short", "PredictionString": short_pred_str}
        )

    return pd.DataFrame(submission_rows)


def failure_analysis(val_preds_df, ground_truth_df):
    """
    Analyzes model failures by correlating error with document and question lengths.
    """
    print("\n--- Failure Analysis ---")

    # Create a lookup for predictions
    pred_map = dict(zip(val_preds_df["example_id"], val_preds_df["PredictionString"]))

    # Calculate correctness per example
    scores = []
    val_ids = set()

    for _, row in ground_truth_df.iterrows():
        eid = str(row["example_id"])
        val_ids.add(eid)

        # Check Long Answer
        long_pred = pred_map.get(f"{eid}_long", "")
        # Correct if pred in valid set OR (pred is empty AND valid set is empty)
        long_correct = (
            (long_pred in row["valid_long"])
            if len(row["valid_long"]) > 0
            else (long_pred == "")
        )

        # Check Short Answer
        short_pred = pred_map.get(f"{eid}_short", "")
        short_correct = (
            (short_pred in row["valid_short"])
            if len(row["valid_short"]) > 0
            else (short_pred == "")
        )

        # Define "Correct" as getting both right (strict)
        is_correct = 1.0 if (long_correct and short_correct) else 0.0
        scores.append({"example_id": eid, "is_correct": is_correct})

    score_df = pd.DataFrame(scores)

    # Extract features (Doc Length, Question Length) from source JSONL
    # We iterate the train file since validation is a subset of it
    features = []
    print("Extracting features from source file...")

    # We scan the file efficiently
    for entry in load_jsonl(Config.TRAIN_FILE):
        eid = str(entry["example_id"])
        if eid in val_ids:
            doc_len = len(entry.get("document_text", "").split())
            q_len = len(entry.get("question_text", "").split())
            features.append({"example_id": eid, "doc_len": doc_len, "q_len": q_len})

            # Optimization: stop if we found all
            if len(features) >= len(val_ids):
                break

    feat_df = pd.DataFrame(features)

    # Merge scores and features
    analysis_df = pd.merge(score_df, feat_df, on="example_id", how="inner")

    if analysis_df.empty:
        print("Warning: No matching data for failure analysis.")
        return

    # Calculate Error (1 - Accuracy)
    analysis_df["error"] = 1.0 - analysis_df["is_correct"]

    # Compute Correlations
    corr_doc = analysis_df["error"].corr(analysis_df["doc_len"])
    corr_q = analysis_df["error"].corr(analysis_df["q_len"])

    print(f"Correlation between Error and Document Length: {corr_doc:.4f}")
    print(f"Correlation between Error and Question Length: {corr_q:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Preprocessing
    print("Initializing Tokenizer and Embeddings...")
    tokenizer = Tokenizer()
    # Fit tokenizer on training data (using cache if available)
    tokenizer.fit(Config.TRAIN_FILE, load_cached_data=True)

    # Build embedding matrix
    embedding_matrix = build_embedding_matrix(tokenizer, load_cached_data=True)

    # 3. Data Loading
    print("Preparing Datasets...")
    # Train set: Limited to 20,000 samples for fast baseline training
    train_dataset = NQDataset(
        mode="train", tokenizer=tokenizer, sample_size=20000, load_cached_data=True
    )

    # Validation set (Training loop): Limited subset for quick monitoring
    val_dataset_train = NQDataset(
        mode="val", tokenizer=tokenizer, sample_size=2000, load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader_train = DataLoader(
        val_dataset_train, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 4. Model Initialization
    print("Initializing InteractionGridCNN Model...")
    model = InteractionGridCNN(embedding_matrix)

    # 5. Training
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader_train)
    # Train for 2 epochs to ensure quick execution
    trainer.train(epochs=2)

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # 6. Validation Assessment (Full Set)
    print("Running Inference on Full Validation Set...")
    # Expand candidates=True is required for evaluation to rank all options
    val_dataset_eval = NQDataset(
        mode="val", tokenizer=tokenizer, expand_candidates=True, load_cached_data=True
    )

    val_preds_df = generate_submission_dataframe(
        model, val_dataset_eval, device, threshold=Config.LONG_ANSWER_THRESHOLD
    )

    # Load Ground Truth for Validation
    # load_ground_truth_data loads for the whole file, we filter by validation IDs
    print("Loading Ground Truth...")
    gt_df_full = load_ground_truth_data(Config.TRAIN_FILE, load_cached_data=True)

    # Filter GT to validation set
    val_ids = set([eid.split("_")[0] for eid in val_preds_df["example_id"]])
    gt_df_val = gt_df_full[gt_df_full["example_id"].astype(str).isin(val_ids)]

    # Compute Metric
    precision, recall, f1 = compute_f1_score(val_preds_df, gt_df_val)
    print(f"Final Validation Metric: {f1}")

    # 7. Failure Analysis
    failure_analysis(val_preds_df, gt_df_val)

    # 8. Test Submission
    print("Generating Test Submission...")
    test_dataset = NQDataset(
        mode="test", tokenizer=tokenizer, expand_candidates=True, load_cached_data=True
    )

    submission_df = generate_submission_dataframe(
        model, test_dataset, device, threshold=Config.LONG_ANSWER_THRESHOLD
    )

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
