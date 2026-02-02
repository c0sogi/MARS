import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.train import run_training
from library.model import SiameseDeberta
from library.data import get_dataloaders
from library.utils import seed_everything, compute_log_loss


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    print("Initializing Fast Baseline Run...")

    # 2. Prepare Training Data
    # Training on full dataset as per Lesson 00019 (Data Volume > Model Scale)
    print("Using full training data...")
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    print(
        f"Training configuration: Samples={len(full_train_df)}, Epochs={Config.EPOCHS}"
    )

    # 3. Run Training
    # We use debug=False so that get_dataloaders does not truncate the Validation and Test sets.
    # The Training set is 'truncated' effectively because we changed the source file in Config.
    run_training(debug=False, load_cached_data=True)

    # 4. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")
    device = Config.DEVICE

    # Load Best Model
    model = SiameseDeberta()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get DataLoaders (re-fetch to get full validation/test sets)
    # Note: Train loader will be the subsampled one, but we don't need it here.
    _, val_loader, test_loader = get_dataloaders(debug=False, load_cached_data=True)

    val_preds = []
    val_targets = []
    val_features = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            features = batch["features"].to(device)
            targets = batch["target"].to(device)

            logits = model(
                input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, features
            )
            probs = torch.softmax(logits, dim=1)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_features.append(features.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_features = np.concatenate(val_features, axis=0)

    # Compute Metric
    val_score = compute_log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis: Correlation between Error and Features
    # Calculate per-sample Log Loss
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    # Cross Entropy: -sum(y_true * log(y_pred))
    sample_losses = -np.sum(val_targets * np.log(val_preds_clipped), axis=1)

    # Feature names corresponding to library.data.extract_scalar_features
    # [len_a_char, len_b_char, len_a_word, len_b_word, newline_a, newline_b, ratio, diff]
    feature_names = [
        "char_len_a",
        "char_len_b",
        "word_len_a",
        "word_len_b",
        "newline_a",
        "newline_b",
        "len_ratio",
        "len_diff",
    ]

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    for i, name in enumerate(feature_names):
        if i < val_features.shape[1]:
            feat_values = val_features[:, i]
            # Handle potential constant features (std=0) to avoid NaN correlation
            if np.std(feat_values) > 0 and np.std(sample_losses) > 0:
                corr = np.corrcoef(sample_losses, feat_values)[0, 1]
                print(f"  {name}: {corr:.4f}")
            else:
                print(f"  {name}: NaN (Constant feature or error)")

    # 5. Submission
    THRESHOLD = 1.0115036312379488

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score:.5f}) meets threshold ({THRESHOLD:.5f}). Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids_a = batch["input_ids_a"].to(device)
                attention_mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                attention_mask_b = batch["attention_mask_b"].to(device)
                features = batch["features"].to(device)

                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    features,
                )
                probs = torch.softmax(logits, dim=1)
                test_preds.append(probs.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Load Test Metadata to get IDs
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        submission_df = pd.DataFrame(
            {
                "id": test_df["id"],
                "winner_model_a": test_preds[:, 0],
                "winner_model_b": test_preds[:, 1],
                "winner_tie": test_preds[:, 2],
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({val_score:.5f}) did not meet threshold ({THRESHOLD:.5f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
