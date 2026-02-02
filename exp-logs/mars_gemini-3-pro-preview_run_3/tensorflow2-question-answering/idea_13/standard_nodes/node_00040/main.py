import os
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.trainer import ModelTrainer
from library.inference import PredictionPipeline
from library.text_utils import TextProcessor


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 3
    Config.SAMPLE_SIZE = 20000  # Limit training data size
    Config.DEBUG = True  # Enable sampling in data loaders
    Config.BATCH_SIZE = 128  # Increase batch size for speed

    Config.setup()

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("\n=== Starting Training Phase ===")
    trainer = ModelTrainer()
    # load_cached_data=False forces regeneration of data with the new SAMPLE_SIZE
    # However, if data already exists from previous runs, we might want to use it.
    # Given requirements to "Call data loading functions with load_cached_data=True",
    # we will try to use cached data if available, but for a fresh run in this env,
    # it will likely generate it.
    trainer.run(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n=== Starting Validation Phase ===")

    # Initialize Pipeline
    pipeline = PredictionPipeline()
    pipeline.load_resources(load_cached_data=True)

    # Run Inference on Validation Set
    print("Running inference on validation set...")
    val_preds_df = pipeline.run_inference(test_metadata_path=Config.VAL_METADATA)

    # Create a lookup for predictions
    preds_map = val_preds_df.set_index("example_id").to_dict("index")

    # Load Validation Metadata to get Ground Truth
    val_meta_df = pd.read_csv(Config.VAL_METADATA)

    tp = 0
    fp = 0
    fn = 0

    # For Failure Analysis
    errors = []
    q_lengths = []
    doc_lengths = []  # We'll use document text length as proxy

    print("Computing metrics and extracting features...")
    # Group by file to minimize open/close operations
    for file_name, group in val_meta_df.groupby("file_path"):
        full_path = os.path.join(Config.INPUT_DIR, file_name)

        with open(full_path, "rb") as f:
            for _, row in group.iterrows():
                eid = row["example_id"]
                offset = row["byte_offset"]

                # Read GT
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    entry = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                # Extract Features for Failure Analysis
                q_text = entry.get("question_text", "")
                doc_text = entry.get("document_text", "")
                q_len = len(q_text.split())
                doc_len = len(doc_text.split())

                # Extract Ground Truth
                annotations = entry.get("annotations", [])
                gt_longs = set()
                gt_shorts = set()

                for ann in annotations:
                    # Long Answer
                    la = ann["long_answer"]
                    if la["start_token"] != -1:
                        gt_longs.add(f"{la['start_token']}:{la['end_token']}")

                    # Short Answer
                    if ann["short_answers"]:
                        for sa in ann["short_answers"]:
                            gt_shorts.add(f"{sa['start_token']}:{sa['end_token']}")
                    elif ann["yes_no_answer"] != "NONE":
                        # Pipeline doesn't predict YES/NO, but we record GT for correctness
                        gt_shorts.add(ann["yes_no_answer"])

                # Get Prediction
                pred_row = preds_map.get(eid, {"long_answer": "", "short_answer": ""})
                pred_long = str(pred_row["long_answer"]).strip()
                pred_short = str(pred_row["short_answer"]).strip()

                # --- Compute Stats for Long Answer ---
                match_long = False
                if pred_long:
                    if pred_long in gt_longs:
                        tp += 1
                        match_long = True
                    else:
                        fp += 1

                if not match_long and len(gt_longs) > 0:
                    fn += 1

                # --- Compute Stats for Short Answer ---
                match_short = False
                if pred_short:
                    if pred_short in gt_shorts:
                        tp += 1
                        match_short = True
                    else:
                        fp += 1

                if not match_short and len(gt_shorts) > 0:
                    fn += 1

                # --- Failure Analysis Data ---
                # Calculate F1 for this specific example (averaged over long/short tasks)
                # Local Precision/Recall
                l_tp = (1 if match_long else 0) + (1 if match_short else 0)
                l_fp = (1 if pred_long and not match_long else 0) + (
                    1 if pred_short and not match_short else 0
                )
                l_fn = (1 if not match_long and gt_longs else 0) + (
                    1 if not match_short and gt_shorts else 0
                )

                l_prec = l_tp / (l_tp + l_fp) if (l_tp + l_fp) > 0 else 0.0
                l_rec = (
                    l_tp / (l_tp + l_fn) if (l_tp + l_fn) > 0 else 0.0
                )  # If no GT, Recall is 1.0? No, undefined.
                # If no GT and no Pred: Perfect. F1=1.
                # If no GT and Pred: F1=0.

                if (l_tp + l_fp + l_fn) == 0:
                    # No GT, No Pred -> Perfect
                    example_f1 = 1.0
                elif (l_tp + l_fn) == 0:
                    # No GT, but Pred exists -> Precision 0, Recall undef -> F1 0
                    example_f1 = 0.0
                else:
                    if (l_prec + l_rec) > 0:
                        example_f1 = 2 * l_prec * l_rec / (l_prec + l_rec)
                    else:
                        example_f1 = 0.0

                errors.append(1.0 - example_f1)
                q_lengths.append(q_len)
                doc_lengths.append(doc_len)

    # Compute Global Micro F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Final Validation Metric: {f1}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    if len(errors) > 1:
        corr_q, _ = pearsonr(errors, q_lengths)
        corr_doc, _ = pearsonr(errors, doc_lengths)

        print(f"Correlation (Error vs Question Length): {corr_q}")
        print(f"Correlation (Error vs Document Length): {corr_doc}")

        if abs(corr_q) > abs(corr_doc):
            print(
                "Conclusion: Question length has a stronger association with model error."
            )
        else:
            print(
                "Conclusion: Document length has a stronger association with model error."
            )
    else:
        print("Insufficient data for correlation analysis.")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    print("\n=== Generating Submission ===")
    # The pipeline handles inference on Test set (Config.TEST_METADATA) and saving to CSV
    results_df = pipeline.run_inference(test_metadata_path=Config.TEST_METADATA)
    pipeline.generate_submission(results_df)
    print("Submission generation complete.")


if __name__ == "__main__":
    main()
