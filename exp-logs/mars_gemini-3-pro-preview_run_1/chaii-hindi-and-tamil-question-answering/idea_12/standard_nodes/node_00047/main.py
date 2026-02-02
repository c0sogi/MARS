import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.data import prepare_train_features, prepare_test_features, QADataset
from library.trainer import train_model
from library.inference import predict_and_aggregate, post_processing
from library.utils import set_seed, jaccard


def main():
    # 1. Configuration
    # We use 5 epochs to ensure convergence (Cite Lesson 34, Lesson 40)
    # and 5 seeds for robustness (Cite Lesson 46).
    cfg = Config(debug=False, epochs=5, batch_size=4)
    set_seed(42)

    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 2. Data Loading
    # We manually load train and val metadata to keep them separate for evaluation.
    print("Loading Metadata...")
    train_meta_path = os.path.join(cfg.metadata_dir, "train.csv")
    val_meta_path = os.path.join(cfg.metadata_dir, "val.csv")

    train_df = pd.read_csv(train_meta_path)
    val_df = pd.read_csv(val_meta_path)

    # 3. Prepare Training Data
    print("Processing Training Features...")
    # Generate training features with sliding windows and negative sampling
    train_features = prepare_train_features(cfg, train_df, tokenizer)
    train_dataset = QADataset(train_features, mode="train")

    # 4. Train Model
    print("Starting Training...")
    # Trains models for all seeds defined in Config and saves them to output_dir
    train_model(cfg, train_dataset)

    # 5. Validation Inference
    print("Processing Validation Features for Inference...")
    # Process validation data using the test-time pipeline (sliding windows, no labels)
    # to accurately simulate the inference process.
    val_features = prepare_test_features(cfg, val_df, tokenizer)
    val_dataset = QADataset(val_features, mode="test")

    print("Running Inference on Validation Set...")
    # Run ensemble inference
    val_preds_map = predict_and_aggregate(cfg, val_dataset, val_features)
    # Post-process logits to extract best answer strings
    val_pred_df = post_processing(cfg, val_preds_map, val_features)

    # 6. Compute Metric
    print("Computing Validation Metrics...")
    # Merge predictions with ground truth
    val_merged = pd.merge(val_df, val_pred_df, on="id", how="left")

    # Handle missing predictions
    val_merged["PredictionString"] = val_merged["PredictionString"].fillna("")

    # Calculate Jaccard score for each sample
    val_merged["jaccard"] = val_merged.apply(
        lambda row: jaccard(str(row["answer_text"]), str(row["PredictionString"])),
        axis=1,
    )

    final_metric = val_merged["jaccard"].mean()

    # Print metric in required format
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Error magnitude is the complement of the Jaccard score
    val_merged["error"] = 1.0 - val_merged["jaccard"]

    # Compute basic text features
    val_merged["context_len"] = val_merged["context"].apply(len)
    val_merged["question_len"] = val_merged["question"].apply(len)

    # Calculate correlations
    if len(val_merged) > 1:
        corr_ctx, _ = pearsonr(val_merged["error"], val_merged["context_len"])
        corr_q, _ = pearsonr(val_merged["error"], val_merged["question_len"])

        print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
        print(f"Correlation (Error vs Question Length): {corr_q:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 8. Conditional Submission
    THRESHOLD = 0.6075833333333334

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        test_meta_path = os.path.join(cfg.metadata_dir, "test.csv")
        test_df = pd.read_csv(test_meta_path)

        # Process Test Data
        test_features = prepare_test_features(cfg, test_df, tokenizer)
        test_dataset = QADataset(test_features, mode="test")

        # Run Inference on Test
        test_preds_map = predict_and_aggregate(cfg, test_dataset, test_features)
        submission_df = post_processing(cfg, test_preds_map, test_features)

        # Save Submission
        save_path = os.path.join(cfg.submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(submission_df.head())
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
