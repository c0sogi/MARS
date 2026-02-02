import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import GatedWideMLP
from library.engine import train_model, evaluate, predict


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_continuous = []
    all_targets = []
    all_preds = []

    # Collect data without gradients
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            tokens = batch["tokens"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(continuous, tokens)

            # Move to CPU for analysis
            all_continuous.append(continuous.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    # Concatenate
    X_cont = np.vstack(all_continuous)
    y_true = np.vstack(all_targets).flatten()
    y_pred = np.vstack(all_preds).flatten()

    # Calculate Error Magnitude
    error_magnitude = np.abs(y_true - y_pred)

    # Calculate correlations with continuous features
    # Config.NUM_COLS contains the feature names corresponding to columns in X_cont
    feature_names = Config.NUM_COLS

    correlations = []
    for i, feature_name in enumerate(feature_names):
        if i < X_cont.shape[1]:
            feat_values = X_cont[:, i]
            # Handle potential constant columns to avoid NaN correlation
            if np.std(feat_values) == 0 or np.std(error_magnitude) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feat_values, error_magnitude)[0, 1]
            correlations.append((feature_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features (Top 10):")
    print(f"{'Feature':<10} {'Correlation':<12}")
    for name, corr in correlations[:10]:
        print(f"{name:<10} {corr:.6f}")


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print(
        f"Initializing GatedWideMLP with vocab_size={vocab_size} on {Config.DEVICE}..."
    )
    model = GatedWideMLP(vocab_size=vocab_size).to(Config.DEVICE)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training
    # We use the epochs defined in Config, but early stopping in train_model handles efficiency.
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        Config.DEVICE,
        Config.EPOCHS,
        Config.PATIENCE,
        Config.MODEL_PATH,
    )

    # 6. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    criterion = nn.BCELoss()
    _, val_auc = evaluate(model, val_loader, criterion, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, Config.DEVICE)

    # 8. Submission
    THRESHOLD = 0.9948596381822921

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions
        preds = predict(model, test_loader, Config.DEVICE)

        # Load test metadata for IDs
        test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Verify length alignment
        if len(preds) != len(test_meta):
            print(
                f"Warning: Prediction length {len(preds)} does not match metadata length {len(test_meta)}"
            )

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_meta["id"], "target": preds})

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
