import os
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
import sys

# Import from provided library files
from library.config import Config
from library.trainers import Trainer
from library.inference import InferencePipeline


def validate_and_analyze(pipeline):
    """
    Runs inference on the validation set, computes Micro F1,
    and performs failure analysis.
    """
    print("Starting Validation on Hold-out Set...")

    if not os.path.exists(Config.VAL_METADATA_PATH):
        print(f"Error: Validation metadata not found at {Config.VAL_METADATA_PATH}")
        return

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    source_file = os.path.join(Config.INPUT_DIR, "simplified-nq-train.jsonl")

    # Metrics counters for Micro F1
    tp = 0
    fp = 0
    fn = 0

    # Data for failure analysis
    errors = []
    doc_lengths = []
    q_lengths = []

    print(f"Validating on {len(val_df)} examples...")

    with open(source_file, "rb") as f:
        for idx, row in val_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            # --- 1. Extract Ground Truth ---
            annotations = data.get("annotations", [])

            # Ground Truth Long Answers (Set of strings "start:end")
            gt_long = set()
            for ann in annotations:
                la = ann["long_answer"]
                if la["start_token"] != -1:
                    gt_long.add(f"{la['start_token']}:{la['end_token']}")

            # Ground Truth Short Answers (Set of strings "start:end" or "YES"/"NO")
            gt_short = set()
            for ann in annotations:
                # Yes/No
                if ann["yes_no_answer"] != "NONE":
                    gt_short.add(ann["yes_no_answer"])

                # Spans
                for sa in ann["short_answers"]:
                    gt_short.add(f"{sa['start_token']}:{sa['end_token']}")

            # --- 2. Generate Prediction ---
            try:
                # predict_single returns strings "start:end" or ""
                pred_long, pred_short = pipeline.predict_single(data)
            except Exception:
                pred_long, pred_short = "", ""

            # --- 3. Evaluate Long Answer ---
            has_gt_long = len(gt_long) > 0
            has_pred_long = len(pred_long) > 0

            l_tp = 0
            l_fp = 0
            l_fn = 0

            if has_pred_long:
                if pred_long in gt_long:
                    l_tp = 1
                else:
                    l_fp = 1  # Predicted something, but it was wrong

            if has_gt_long:
                if not has_pred_long:
                    l_fn = 1  # Missed the answer
                elif pred_long not in gt_long:
                    l_fn = 1  # Missed the correct answer (already counted as FP above, but also FN for recall)

            # --- 4. Evaluate Short Answer ---
            has_gt_short = len(gt_short) > 0
            has_pred_short = len(pred_short) > 0

            s_tp = 0
            s_fp = 0
            s_fn = 0

            if has_pred_short:
                if pred_short in gt_short:
                    s_tp = 1
                else:
                    s_fp = 1

            if has_gt_short:
                if not has_pred_short:
                    s_fn = 1
                elif pred_short not in gt_short:
                    s_fn = 1

            # Update Global Counters
            tp += l_tp + s_tp
            fp += l_fp + s_fp
            fn += l_fn + s_fn

            # --- 5. Failure Analysis Data Collection ---
            # We define an error as any mismatch (FP or FN) in either long or short
            is_error = 1 if (l_fp + l_fn + s_fp + s_fn) > 0 else 0

            doc_text = data.get("document_text", "")
            q_text = data.get("question_text", "")

            errors.append(is_error)
            doc_lengths.append(len(doc_text.split()))
            q_lengths.append(len(q_text.split()))

    # --- 6. Compute and Print Metric ---
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Final Validation Metric: {f1}")

    # --- 7. Perform Failure Analysis ---
    print("Performing Failure Analysis...")
    if len(errors) > 1:
        # Correlation with Document Length
        corr_doc, _ = pearsonr(errors, doc_lengths)
        print(f"Correlation between Error and Document Length: {corr_doc}")

        # Correlation with Question Length
        corr_q, _ = pearsonr(errors, q_lengths)
        print(f"Correlation between Error and Question Length: {corr_q}")
    else:
        print("Insufficient data for failure analysis.")


def main():
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # 1. Train Models
    # The Trainer handles data loading and model training.
    # We use load_cached_data=True to use the parquet files generated in ./working if available.
    print("--- Starting Training Phase ---")
    trainer = Trainer()
    trainer.train_ranker(load_cached_data=True)
    trainer.train_reader(load_cached_data=True)

    # 2. Validation and Analysis
    # We use the InferencePipeline to load the best models and run prediction.
    print("\n--- Starting Validation Phase ---")
    pipeline = InferencePipeline(load_cached_data=True)
    validate_and_analyze(pipeline)

    # 3. Submission
    # Generate predictions for the test set.
    print("\n--- Starting Submission Phase ---")
    pipeline.run_inference(load_cached_data=True)
    print("Run completed.")


if __name__ == "__main__":
    main()
