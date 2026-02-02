import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from library.config import Config
from library.model import SharedBottomRoBERTa
from library.engine import run_training
from library.dataset import get_dataloaders, load_data
from library.utils import seed_everything, compute_spearmanr


def inference(model, loader, device):
    """
    Runs inference on a dataloader and returns predictions and labels (if available).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)

            # Store IDs if needed for submission alignment
            if "qa_id" in batch:
                all_ids.extend(batch["qa_id"].numpy())

            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)

    labels = None
    if len(all_labels) > 0:
        labels = np.concatenate(all_labels, axis=0)

    return predictions, labels, all_ids


def main():
    # 1. Configuration and Setup
    config = Config()
    # Ensure reproducibility
    seed_everything(config.SEED)

    print("Initializing configuration...")
    config.print_config()

    # 2. Training
    # run_training handles the training loop, validation per epoch, and saving the best model.
    print("\n--- Starting Training ---")
    run_training(config)

    # 3. Load Best Model for Final Evaluation
    print("\n--- Loading Best Model ---")
    model = SharedBottomRoBERTa(config)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}")

    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.device)
    )
    model.to(config.device)
    model.eval()

    # 4. Validation Assessment
    print("\n--- Performing Validation Assessment ---")
    loaders = get_dataloaders(config, load_cached_data=True)
    val_loader = loaders["val"]

    val_preds, val_labels, _ = inference(model, val_loader, config.device)

    # Compute Metric
    final_metric = compute_spearmanr(val_preds, val_labels)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Performing Failure Analysis ---")
    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples, N_targets) -> mean over targets -> (N_samples,)
    mae_per_sample = np.mean(np.abs(val_preds - val_labels), axis=1)

    # Load validation dataframe to get text features
    val_df = load_data("val", config, load_cached_data=True)

    # Ensure alignment (loaders might drop last if configured, but val loader usually doesn't)
    # The dataloader and dataframe should be aligned if shuffle=False (which it is for val)
    if len(val_df) != len(mae_per_sample):
        print(
            f"Warning: Size mismatch in failure analysis. DF: {len(val_df)}, Preds: {len(mae_per_sample)}"
        )
        # Truncate to match (assumes order is preserved)
        min_len = min(len(val_df), len(mae_per_sample))
        val_df = val_df.iloc[:min_len]
        mae_per_sample = mae_per_sample[:min_len]

    # Compute meta-features
    val_df["q_title_len"] = val_df["question_title"].fillna("").str.len()
    val_df["q_body_len"] = val_df["question_body"].fillna("").str.len()
    val_df["answer_len"] = val_df["answer"].fillna("").str.len()

    # Correlate Error with Lengths
    features_to_check = ["q_title_len", "q_body_len", "answer_len"]
    print("Correlation between Mean Absolute Error and Input Features:")
    for feat in features_to_check:
        corr, _ = spearmanr(mae_per_sample, val_df[feat])
        print(f"  MAE vs {feat}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.4118214482019393

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        if "test" not in loaders:
            # Should not happen if test.csv exists
            print("Test loader not found. Attempting to recreate.")
            loaders = get_dataloaders(config, load_cached_data=True)

        test_loader = loaders["test"]
        test_preds, _, test_ids = inference(model, test_loader, config.device)

        # Load sample submission to get column names
        sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)
        target_cols = [col for col in sample_sub.columns if col != "qa_id"]

        # Create submission DataFrame
        submission_df = pd.DataFrame(test_preds, columns=target_cols)
        submission_df.insert(0, "qa_id", test_ids)

        # Save
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
