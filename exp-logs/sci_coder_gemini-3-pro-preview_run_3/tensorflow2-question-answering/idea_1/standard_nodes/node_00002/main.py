import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pointbiserialr

from library.config import Config
from library.trainer import Trainer
from library.inference import NQPipeline
from library.data_loader import InferenceDataset, preprocess_annotations
from library.text_processing import tokenize_text, split_document_by_html


def setup_fast_config():
    """Overrides Config defaults for a fast baseline run."""
    # Limit data for speed
    Config.MAX_TRAIN_SAMPLES = 50000

    # Training parameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32  # Safe size for 12GB+ GPU

    # Inference threshold
    Config.CONFIDENCE_THRESHOLD = 0.1

    print("Configuration updated for fast baseline execution.")


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def run_validation_and_analysis(trainer):
    print("\n--- Starting Validation & Failure Analysis ---")

    # 1. Load Validation Metadata and Ground Truth
    val_meta_path = Config.VAL_METADATA_PATH
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found.")
        return

    val_meta_df = pd.read_csv(val_meta_path)

    # Load processed annotations (Ground Truth)
    # This file contains only samples with valid answers.
    # We need to handle samples without answers as well (GT = NULL).
    gt_df = preprocess_annotations(val_meta_df, load_cached_data=True)

    # Create a lookup for ground truth: example_id -> {long: (s, e), short: (s, e)}
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        eid = str(row["example_id"])

        # Get global offsets from the processed record
        # Note: preprocess_annotations saves candidate index. We need to map back to global if we want exact string match,
        # but our pipeline predicts token indices.
        # Actually, preprocess_annotations logic in data_loader.py doesn't explicitly save global start/end in the dataframe
        # for the final output, it saves local offsets.
        # However, we can infer existence.
        # To get exact global coordinates for metric calculation, we need to re-derive them or trust the pipeline logic.
        # For the purpose of this script, we will rely on the fact that we need to match the *valid* answer.
        # Let's re-read the file logic or simply use the fact that we have the file path and can re-parse if needed.
        # Optimization: The InferenceDataset returns candidate metadata. We can match based on that.

        # Construct GT entry
        # We need to know WHICH candidate is positive and the offsets.
        # The dataframe has 'pos_cand_idx'.
        # We will store: {'cand_idx': int, 'short_local': (s, e) or None}
        entry = {
            "pos_cand_idx": row["pos_cand_idx"],
            "has_short": row["has_short_answer"],
        }
        if row["has_short_answer"]:
            entry["short_local"] = (row["short_start_local"], row["short_end_local"])

        gt_lookup[eid] = entry

    # 2. Prepare Inference Dataset for Validation
    val_dataset = InferenceDataset(val_meta_df, trainer.vocab_encoder)
    # Use a custom collate_fn similar to the one in data_loader
    from library.data_loader import inference_collate_fn

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=inference_collate_fn,
    )

    # 3. Inference Loop
    trainer.ranker.eval()
    trainer.reader.eval()
    device = trainer.device
    sep_token_id = trainer.vocab_encoder.unk_idx
    pad_token_id = trainer.vocab_encoder.pad_idx

    tp = 0
    fp = 0
    fn = 0

    # For Failure Analysis
    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            example_id, q_ids, cand_tensors, cand_meta = batch
            example_id = str(example_id)

            # Features for analysis
            q_len = (q_ids != pad_token_id).sum().item()
            num_candidates = len(cand_meta)

            # Move to device
            q_ids = q_ids.to(device).unsqueeze(0)
            cand_tensors = cand_tensors.to(device).unsqueeze(0)

            # --- Prediction ---
            scores = trainer.ranker(q_ids, cand_tensors).squeeze(0)  # (K,)

            pred_long = None
            pred_short = None

            if scores.numel() > 0:
                best_score, best_idx = torch.max(scores, dim=0)
                best_idx = best_idx.item()

                if best_score.item() >= Config.CONFIDENCE_THRESHOLD:
                    # Long Answer Prediction
                    pred_long = best_idx  # Candidate index

                    # Short Answer Prediction
                    q_raw = q_ids[0]
                    ctx_raw = cand_tensors[0, best_idx]
                    q_valid = q_raw[q_raw != pad_token_id]
                    ctx_valid = ctx_raw[ctx_raw != pad_token_id]

                    sep_tensor = torch.tensor([sep_token_id], device=device)
                    reader_input = torch.cat(
                        [q_valid, sep_tensor, ctx_valid]
                    ).unsqueeze(0)

                    start_logits, end_logits = trainer.reader(reader_input)
                    s_pred = torch.argmax(start_logits, dim=1).item()
                    e_pred = torch.argmax(end_logits, dim=1).item()

                    offset = len(q_valid) + 1
                    local_s = s_pred - offset
                    local_e = e_pred - offset

                    if local_s >= 0 and local_e >= local_s and local_e < len(ctx_valid):
                        pred_short = (local_s, local_e)

            # --- Evaluation ---
            gt = gt_lookup.get(example_id)

            # Long Answer Eval
            is_long_error = False
            if gt is None:
                # GT is Null
                if pred_long is not None:
                    fp += 1
                    is_long_error = True
                # else TN (ignore)
            else:
                # GT exists
                if pred_long is None:
                    fn += 1
                    is_long_error = True
                elif pred_long == gt["pos_cand_idx"]:
                    tp += 1
                else:
                    # Wrong candidate
                    fp += 1
                    fn += 1
                    is_long_error = True

            # Short Answer Eval
            # Logic: If GT Short exists, we must match it.
            # If GT Short doesn't exist (Long only), pred should be null.
            is_short_error = False
            if gt is None or not gt.get("has_short", False):
                # GT Short is Null
                if pred_short is not None:
                    fp += 1
                    is_short_error = True
            else:
                # GT Short exists
                if pred_short is None:
                    fn += 1
                    is_short_error = True
                elif (pred_long == gt["pos_cand_idx"]) and (
                    pred_short == gt["short_local"]
                ):
                    # Must match candidate AND span
                    tp += 1
                else:
                    fp += 1
                    fn += 1
                    is_short_error = True

            # Record for analysis (1 if any error, 0 otherwise)
            error_flag = 1 if (is_long_error or is_short_error) else 0
            analysis_data.append(
                {
                    "q_len": q_len,
                    "num_candidates": num_candidates,
                    "is_error": error_flag,
                }
            )

    # 4. Compute Metric
    final_f1 = compute_f1(tp, fp, fn)
    print(f"Final Validation Metric: {final_f1}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(analysis_data)

    if len(df_analysis) > 0 and df_analysis["is_error"].std() > 0:
        # Point-biserial correlation: Binary Error vs Continuous Feature
        corr_q, _ = pointbiserialr(df_analysis["is_error"], df_analysis["q_len"])
        corr_c, _ = pointbiserialr(
            df_analysis["is_error"], df_analysis["num_candidates"]
        )

        print("Correlation between Error and Input Features:")
        print(f"  Question Length: {corr_q:.4f}")
        print(f"  Document Complexity (Num Candidates): {corr_c:.4f}")

        # Interpretation
        if abs(corr_c) > 0.1:
            print("  -> Significant correlation with document complexity.")
        if abs(corr_q) > 0.1:
            print("  -> Significant correlation with question length.")
    else:
        print("Insufficient variance in errors for correlation analysis.")


def main():
    # 1. Setup
    setup_fast_config()

    # 2. Training
    # Initialize trainer with debug sample size if needed, or full (limited by Config)
    trainer = Trainer()
    trainer.run_training()

    # 3. Validation & Analysis
    run_validation_and_analysis(trainer)

    # 4. Submission
    # We can reuse the trained models via NQPipeline or just call the method on trainer if we implemented it there.
    # The provided Trainer class has generate_submission, but NQPipeline is the designated inference class.
    # NQPipeline loads from disk. Since Trainer saved 'best' models to disk, this works.

    # Clear GPU memory before inference pipeline to be safe
    del trainer
    torch.cuda.empty_cache()

    pipeline = NQPipeline()
    pipeline.generate_predictions()


if __name__ == "__main__":
    main()
