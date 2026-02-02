import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.trainer import get_scaled_data, ContactDataset
from library.model import MMWIN, train_model, optimize_threshold, predict


def main():
    # 1. Configuration and Seeding
    seed_everything(Config.SEED)

    # Configure for Fast Baseline Execution
    # We keep USE_ALL_DATA = True to ensure we have the full validation set for accurate metrics,
    # but we will subsample the training data in memory.
    Config.EPOCHS = 5  # Reduce epochs for speed

    # 2. Data Loading
    print("Loading data...")
    # This loads full datasets (cached if available)
    X_train, y_train, X_val, y_val, X_test, test_ids = get_scaled_data(
        load_cached_data=True
    )

    # Training on full data (Cite Lesson 00023)
    # The simple MLP architecture allows for efficient training on the large dataset.
    print(f"Training on full dataset: {len(X_train)} samples.")

    # Create Datasets
    train_dataset = ContactDataset(X_train, y_train)
    val_dataset = ContactDataset(X_val, y_val)
    test_dataset = ContactDataset(X_test)

    # Create DataLoaders
    # Using num_workers=4 for efficient data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    print(f"Initializing model on {device} with input dimension {input_dim}...")

    model = MMWIN(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    )

    # 4. Training
    print("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
    )

    # 5. Validation & Threshold Optimization
    print("Optimizing decision threshold on validation set...")
    best_threshold = optimize_threshold(model, val_loader, device)

    # Compute Final Validation Metric on the ENTIRE hold-out set
    print("Computing final validation metrics...")
    model.eval()
    val_probs = []

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_preds = (val_probs > best_threshold).astype(int)

    final_val_mcc = compute_mcc(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_mcc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude: |y_true - y_prob|
    errors = np.abs(y_val - val_probs.flatten())

    # Calculate correlation between features and error magnitude
    # X_val is a numpy array, so we reference features by index
    feature_correlations = []
    num_features = X_val.shape[1]

    for i in range(num_features):
        feature_values = X_val[:, i]
        # Avoid correlation calculation for constant features
        if np.std(feature_values) > 1e-9:
            corr, _ = pearsonr(errors, feature_values)
            feature_correlations.append((i, corr))
        else:
            feature_correlations.append((i, 0.0))

    # Sort by absolute correlation (descending)
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(
        "Top 5 Features correlated with Error Magnitude (Feature Index, Correlation):"
    )
    for idx, corr in feature_correlations[:5]:
        print(f"Feature {idx}: {corr:.6f}")

    # 7. Submission Generation
    TARGET_METRIC = 0.62458462731896

    if final_val_mcc > TARGET_METRIC:
        print(
            f"\nValidation metric ({final_val_mcc}) exceeds threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Predict on Test Set
        test_preds_bin = predict(model, test_loader, device, threshold=best_threshold)

        # Create Submission DataFrame
        submission = pd.DataFrame({"contact_id": test_ids, "contact": test_preds_bin})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation metric ({final_val_mcc}) did not exceed threshold ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
