import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_score, jaccard
from library.data import QADataset, load_and_process_data, prepare_test_features
from library.model import QAModel
from library.engine import train_loop, predict_fn, post_process_predictions


def main():
    # 1. Setup Configuration and Environment
    config = Config()
    seed_everything(config.seed)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading and Preparation
    print("Loading and processing data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    # Load the main datasets (cached if available)
    # train_features contains the training data with 'fold' columns (for CV)
    # test_features contains the submission test data
    train_features, _, test_features = load_and_process_data(
        config, tokenizer, load_cached_data=True
    )

    # Load Hold-out Validation Set for Final Scoring
    # We need to process this specifically for inference (with offset mapping)
    # to reconstruct the answer strings for Jaccard calculation.
    # The default load_and_process_data processes validation data for loss calculation (targets),
    # but for the final metric we need it processed as 'test' data (reconstruction).
    if os.path.exists(config.val_data_path):
        val_df_raw = pd.read_csv(config.val_data_path)
        val_features_inference = prepare_test_features(val_df_raw, tokenizer, config)

        val_inference_ds = QADataset(val_features_inference, is_test=True)
        val_inference_loader = DataLoader(
            val_inference_ds, batch_size=config.batch_size, shuffle=False
        )
    else:
        print("Error: Validation data path not found.")
        return

    # Prepare Test Loader for Submission
    test_ds = QADataset(test_features, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # Initialize Accumulators for Ensemble Averaging
    # Dimensions: [num_samples, max_seq_length]
    n_val_samples = len(val_features_inference)
    n_test_samples = len(test_features)

    val_start_accum = np.zeros((n_val_samples, config.max_length))
    val_end_accum = np.zeros((n_val_samples, config.max_length))

    test_start_accum = np.zeros((n_test_samples, config.max_length))
    test_end_accum = np.zeros((n_test_samples, config.max_length))

    # 3. 5-Fold Cross-Validation Training Loop
    print(f"Starting {config.n_folds}-Fold Cross-Validation...")

    for fold in range(config.n_folds):
        print(f"\n=== Fold {fold + 1}/{config.n_folds} ===")

        # Split data for this fold
        fold_train_df = train_features[train_features["fold"] != fold].reset_index(
            drop=True
        )
        fold_val_df = train_features[train_features["fold"] == fold].reset_index(
            drop=True
        )

        # Create DataLoaders
        train_ds = QADataset(fold_train_df, is_test=False)
        fold_val_ds = QADataset(fold_val_df, is_test=False)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        fold_val_loader = DataLoader(
            fold_val_ds, batch_size=config.batch_size, shuffle=False
        )

        # Initialize Model
        model = QAModel(config).to(device)

        # Train (using engine.train_loop)
        # Note: train_loop handles optimizer, scheduler, and early stopping
        model = train_loop(model, train_loader, fold_val_loader, config, device)

        # Inference on Hold-out Validation Set (Accumulate Logits)
        print("Inferencing on validation set...")
        s_logits, e_logits = predict_fn(model, val_inference_loader, device)
        val_start_accum += s_logits
        val_end_accum += e_logits

        # Inference on Test Set (Accumulate Logits)
        print("Inferencing on test set...")
        s_logits_test, e_logits_test = predict_fn(model, test_loader, device)
        test_start_accum += s_logits_test
        test_end_accum += e_logits_test

        # Cleanup to save memory
        del model, train_loader, fold_val_loader
        torch.cuda.empty_cache()

    # 4. Ensemble Aggregation
    print("\nAggregating Ensemble Predictions...")
    val_start_avg = val_start_accum / config.n_folds
    val_end_avg = val_end_accum / config.n_folds

    test_start_avg = test_start_accum / config.n_folds
    test_end_avg = test_end_accum / config.n_folds

    # 5. Validation Evaluation
    # We need to trick post_process_predictions into reading the validation CSV
    # to extract context strings. It uses Config.test_data_path internally.
    original_test_path = Config.test_data_path
    Config.test_data_path = config.val_data_path

    print("Post-processing validation predictions...")
    val_preds_map = post_process_predictions(
        val_features_inference, val_start_avg, val_end_avg
    )

    # Restore config path for test set processing later
    Config.test_data_path = original_test_path

    # Compute Metric
    # Ground truth
    val_ground_truth = dict(zip(val_df_raw["id"], val_df_raw["answer_text"]))

    y_true = []
    y_pred = []

    # Ensure alignment by ID
    for uid, gt_text in val_ground_truth.items():
        pred_text = val_preds_map.get(uid, "")
        y_true.append(str(gt_text))
        y_pred.append(str(pred_text))

    final_metric = compute_score(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample error
    sample_scores = [jaccard(t, p) for t, p in zip(y_true, y_pred)]
    errors = [1.0 - s for s in sample_scores]

    # Build analysis dataframe
    analysis_df = pd.DataFrame({"id": list(val_ground_truth.keys()), "error": errors})

    # Merge with raw features
    analysis_df = analysis_df.merge(val_df_raw, on="id", how="left")

    # Compute meta-features
    analysis_df["context_len"] = analysis_df["context"].astype(str).apply(len)
    analysis_df["question_len"] = analysis_df["question"].astype(str).apply(len)

    # Compute correlations
    corr_ctx = analysis_df["error"].corr(analysis_df["context_len"])
    corr_que = analysis_df["error"].corr(analysis_df["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx}")
    print(f"Correlation (Error vs Question Length): {corr_que}")

    # 7. Submission Generation
    threshold = 0.4804191919191919

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Post-process test predictions
        # Config.test_data_path is already correct (restored above)
        test_preds_map = post_process_predictions(
            test_features, test_start_avg, test_end_avg
        )

        # Create submission file
        sample_sub = pd.read_csv(config.sample_submission_path)

        final_preds_list = []
        for pid in sample_sub["id"]:
            final_preds_list.append(test_preds_map.get(pid, ""))

        sample_sub["PredictionString"] = final_preds_list

        # Ensure directory exists
        os.makedirs(os.path.dirname(config.submission_file), exist_ok=True)

        sample_sub.to_csv(config.submission_file, index=False)
        print(f"Submission saved to {config.submission_file}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
