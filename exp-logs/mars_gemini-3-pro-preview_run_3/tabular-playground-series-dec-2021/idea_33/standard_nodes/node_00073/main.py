import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, set_performance_mode
from library.data import get_dataloaders
from library.model import ZeroInitDeepAsymmetricNet
from library.train import Trainer


def main():
    # 1. Setup & Configuration
    # Ensure reproducibility
    seed_everything(Config.SEED)
    # Optimize for A100
    set_performance_mode(deterministic=False, benchmark=True)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use the full dataset to maximize performance to hit the threshold.
    # Batch size is large (4096) for A100 efficiency.
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        num_workers=4,
        pin_memory=True,
    )

    # Determine input dimension from a dummy batch
    dummy_x, _ = next(iter(train_loader))
    input_dim = dummy_x.shape[1]
    print(f"Input dimension: {input_dim}")

    # 3. Model Initialization
    print("Initializing model...")
    model = ZeroInitDeepAsymmetricNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.RESNET_BLOCKS,
        dcn_layers=Config.DCN_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Training
    # We use the Trainer class from library.train
    print("Starting training...")
    trainer = Trainer(model, device, Config)

    # We run for the configured epochs to ensure we hit the high threshold.
    # The A100 is fast enough to handle 60 epochs on 3M rows within 2 hours.
    best_acc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 5. Final Validation & Metric Reporting
    print("Performing final validation...")
    # We re-run validation on the best model (loaded by trainer.fit) to get exact metrics and data for failure analysis
    model.eval()

    val_preds = []
    val_targets = []
    val_inputs = []

    # Collect all validation data for analysis
    # We disable gradients for inference speed
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            val_preds.append(predicted.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
            # Store inputs for failure analysis (move to CPU to save GPU memory)
            val_inputs.append(X_batch.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_inputs = np.concatenate(val_inputs)

    # Calculate Accuracy
    accuracy = (val_preds == val_targets).mean()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Error vector: 1 if incorrect, 0 if correct
    errors = (val_preds != val_targets).astype(int)

    # Calculate correlation between Error and each Feature
    # val_inputs shape: (N, Features)
    n_features = val_inputs.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = val_inputs[:, i]
        # Handle constant columns to avoid warning/nan
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_col, errors)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.6f}")

    # 7. Conditional Submission
    THRESHOLD = 0.9626291666666666

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy} > {THRESHOLD}. Generating submission..."
        )

        test_preds = trainer.predict(test_loader)

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: test_preds})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation accuracy {accuracy} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
