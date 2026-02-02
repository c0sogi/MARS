import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import (
    EPOCHS,
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    SUBMISSION_PATH,
    SEED,
    FINAL_CONTINUOUS_FEATURES,
    FINAL_BINARY_FEATURES,
)
from library.utils import seed_everything, get_device, accuracy_score
from library.data import get_dataloaders
from library.model import DeepVectorDCNResNet
from library.train import Trainer


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True allows using the preprocessed numpy files in ./working
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DeepVectorDCNResNet(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES
    )

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    # Using the config epochs. With A100 and batch size 4096, this is fast.
    trainer.fit(epochs=EPOCHS, patience=15)

    # 5. Validation & Metric Calculation
    print("Performing final validation...")
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            _, preds = torch.max(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            # Store inputs for failure analysis (move to CPU to save GPU mem)
            all_inputs.append(batch_X.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_inputs = np.concatenate(all_inputs)

    # Calculate Metric
    val_accuracy = np.mean(all_preds == all_targets)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_accuracy}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    errors = (all_preds != all_targets).astype(int)

    # Reconstruct feature names list
    # The data pipeline stacks continuous then binary features
    feature_names = FINAL_CONTINUOUS_FEATURES + FINAL_BINARY_FEATURES

    correlations = []
    # Calculate correlation for each feature against the error vector
    # Using numpy for speed
    for i, feature_name in enumerate(feature_names):
        feat_values = all_inputs[:, i]

        # Check for constant features to avoid division by zero in correlation
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            # Pearson correlation
            corr, _ = pearsonr(feat_values, errors)
            if np.isnan(corr):
                corr = 0.0

        correlations.append((feature_name, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Error:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.9625041666666667
    if val_accuracy > THRESHOLD:
        print(
            f"\nValidation metric ({val_accuracy}) > threshold ({THRESHOLD}). Generating submission..."
        )
        df_submission = trainer.predict(test_loader, test_ids)

        # Ensure directory exists (handled by config but good to be safe)
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        df_submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({val_accuracy}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
