import torch
import numpy as np
import pandas as pd
import sys
import os

from library.config import (
    SEED,
    DEVICE,
    HIDDEN_DIM,
    NUM_CLASSES,
    DROPOUT,
    EPOCHS,
    RAW_CONTINUOUS_FEATURES,
    DERIVED_FEATURES,
    RAW_BINARY_FEATURES,
)
from library.utils import seed_everything
from library.data_loader import get_dataloaders, process_data
from library.model import ParallelDCNResNet
from library.trainer import Trainer, generate_predictions


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Data Loading
    # Using load_cached_data=False to force regeneration of the full dataset
    # Cite debug_lesson_6: Verify Cache Integrity Against Runtime Configuration
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Infer input dimension from a single batch
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]

    # 3. Model Initialization
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )
    model.to(DEVICE)

    # 4. Training
    # We use the config EPOCHS (60). On A100 with tabular data, this is very fast (<15 mins).
    # This ensures we have the best chance to hit the high threshold.
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer.fit(epochs=EPOCHS)

    # 5. Validation & Metric Reporting
    # The trainer loads the best model state at the end of fit()
    val_loss, val_acc = trainer.validate()
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load raw numpy arrays for correlation analysis
    # process_data returns: X_train, y_train, X_val, y_val, X_test, test_ids
    _, _, X_val, y_val, _, _ = process_data(load_cached_data=True)

    # Generate predictions on validation set to identify errors
    model.eval()
    val_preds = []
    with torch.no_grad():
        # val_loader is shuffle=False, so order matches X_val/y_val
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())

    val_preds = np.array(val_preds)

    # Calculate Error Vector (1 = Wrong, 0 = Correct)
    # y_val is 0-indexed integer labels
    errors = (val_preds != y_val).astype(int)

    # Construct Feature Names list to match column order
    # Order defined in data_loader.py: Continuous (Raw + Derived) then Binary
    feature_names = RAW_CONTINUOUS_FEATURES + DERIVED_FEATURES + RAW_BINARY_FEATURES

    # Calculate Correlations
    correlations = []
    # Ensure we don't go out of bounds if dimensions mismatch (safety check)
    num_features = min(len(feature_names), X_val.shape[1])

    for i in range(num_features):
        feat_name = feature_names[i]
        feat_values = X_val[:, i]

        # Pearson correlation between Feature and Error
        # Handling potential constant features (std=0) which produce NaN correlation
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        correlations.append((feat_name, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Model Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.9625222222222222
    if val_acc > THRESHOLD:
        generate_predictions(model, test_loader, test_ids, DEVICE)
    else:
        print(
            f"Validation metric {val_acc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
