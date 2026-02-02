import os
import sys
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_dataloaders
from library.model import ResNetMLP
from library.train import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Overrides
    # --------------------------------------------------------------------------
    # Optimize for A100 GPU and ensure fast execution within time limits
    Config.BATCH_SIZE = 4096
    Config.EPOCHS = 20  # Reduced from 30 to ensure timely completion

    print(
        f"Initializing pipeline with Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}"
    )
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Load full dataset using cached numpy arrays if available
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Determine input dimensions dynamically
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    print(f"Detected {input_dim} total features.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)
    model = ResNetMLP(input_dim=input_dim)
    model.to(device)

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_ids=test_ids,
        device=device,
    )

    # Execute training loop with early stopping
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nRunning Validation and Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []

    # Perform inference on validation set
    # We accumulate results on CPU to avoid GPU OOM during analysis
    with torch.no_grad():
        for x, target in val_loader:
            x = x.to(device)
            target = target.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_inputs.append(x.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Calculate and Print Final Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy:.16f}")

    # --- Failure Analysis ---
    # Calculate error vector (1 if prediction is wrong, 0 if correct)
    errors = (all_preds != all_targets).astype(int)

    # Reconstruct feature names for meaningful reporting
    # Since we now use all features (cont + binary), we need a generic way to name them
    # or just use indices if it's too complex to reconstruct perfectly in runfile.
    feature_names = [f"Feature_{i}" for i in range(input_dim)]

    # Compute correlation between each feature and the error magnitude
    print("\nFailure Analysis - Correlation with Error Magnitude:")
    correlations = []
    for i in range(all_inputs.shape[1]):
        feat_vals = all_inputs[:, i]
        # Handle potential constant features (avoid division by zero)
        if np.std(feat_vals) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<30} {'Correlation':<10}")
    print("-" * 45)
    for name, corr in correlations[:10]:
        print(f"{name[:29]:<30} {corr:.6f}")

    # --------------------------------------------------------------------------
    # 6. Submission Logic
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9622416666666667

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy:.6f} > {THRESHOLD:.6f}. Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation accuracy {accuracy:.6f} <= {THRESHOLD:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
