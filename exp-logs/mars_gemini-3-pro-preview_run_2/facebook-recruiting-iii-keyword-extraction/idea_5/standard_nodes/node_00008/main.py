import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score

from library.config import Config
from library.data_processing import prepare_loaders
from library.model import LinearTaggingModel, FocalLoss
from library.engine import train_model, validate
from library.inference import find_best_threshold, generate_submission
from library.utils import set_seed


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for a fast baseline execution
    # A linear model on A100 is very fast, but we reduce epochs to be safe within time limits
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 8192

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n=== Data Loading ===")
    # Load data using the library function
    # This handles caching automatically
    train_loader, val_loader, test_loader, encoder, test_ids = prepare_loaders(
        load_cached_data=True
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n=== Model Initialization ===")
    model = LinearTaggingModel(input_dim=Config.INPUT_DIM, output_dim=Config.OUTPUT_DIM)

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("\n=== Training ===")
    # train_model handles the training loop, validation monitoring, and early stopping
    model = train_model(model, train_loader, val_loader)

    # ---------------------------------------------------------
    # 5. Validation Assessment
    # ---------------------------------------------------------
    print("\n=== Validation Assessment ===")
    device = torch.device(Config.DEVICE)
    # Use FocalLoss for consistency, though we just need probabilities here
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Run validation to get final probabilities and targets
    val_loss, val_probs, val_targets = validate(model, val_loader, criterion, device)

    # Find optimal threshold using the library function
    best_threshold = find_best_threshold(val_targets, val_probs)

    # Compute Final Validation Metric (Mean F1-Score)
    val_preds = (val_probs > best_threshold).astype(int)
    final_f1 = f1_score(val_targets, val_preds, average="samples", zero_division=0)

    # REQUIRED: Print the final validation metric in the specified format
    print(f"Final Validation Metric: {final_f1}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    try:
        # Load validation metadata to retrieve text features
        # We assume the order matches val_loader (shuffle=False in prepare_loaders)
        val_df = pd.read_csv(Config.VAL_PATH)

        # If DEBUG mode was used in prepare_loaders, we must slice val_df to match
        if len(val_df) != len(val_targets):
            print(
                f"Adjusting validation dataframe length from {len(val_df)} to {len(val_targets)}..."
            )
            val_df = val_df.iloc[: len(val_targets)]

        # Calculate Input Features
        # Title Length
        val_df["title_len"] = val_df["Title"].fillna("").astype(str).apply(len)
        # Body Length
        val_df["body_len"] = val_df["Body"].fillna("").astype(str).apply(len)

        # Calculate Error Magnitude
        # We use the sum of absolute errors per sample: sum(|prob - target|)
        # This represents how 'far' the probability distribution is from the ideal 0/1 vector
        error_magnitude = np.sum(np.abs(val_probs - val_targets), axis=1)

        # Calculate Correlations
        corr_title = np.corrcoef(val_df["title_len"], error_magnitude)[0, 1]
        corr_body = np.corrcoef(val_df["body_len"], error_magnitude)[0, 1]

        print(f"Correlation between Error Magnitude and Title Length: {corr_title}")
        print(f"Correlation between Error Magnitude and Body Length: {corr_body}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\n=== Submission Generation ===")
    TARGET_METRIC = 0.0542101508997596

    if final_f1 > TARGET_METRIC:
        print(f"Metric {final_f1} > {TARGET_METRIC}. Proceeding with submission.")
        generate_submission(model, test_loader, test_ids, encoder, best_threshold)
    else:
        print(f"Metric {final_f1} <= {TARGET_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()
