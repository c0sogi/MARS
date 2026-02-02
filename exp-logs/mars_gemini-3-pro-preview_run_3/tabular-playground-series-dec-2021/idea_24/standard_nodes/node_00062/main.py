import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import process_data
from library.model import ParallelDCNResNet
from library.train import Trainer, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Get computation device
    device = get_device()

    # Fast Baseline Configuration overrides
    # We limit epochs and training samples to ensure quick execution
    FAST_EPOCHS = 10
    FAST_TRAIN_SAMPLES = 200000

    print(
        f"Running Fast Baseline with {FAST_EPOCHS} epochs and {FAST_TRAIN_SAMPLES} training samples."
    )

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load processed data (utilizing cache if available)
    # process_data returns: X_train, y_train, X_val, y_val, X_test, test_ids
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=True
    )

    # Subsample Training Data for speed
    if len(X_train) > FAST_TRAIN_SAMPLES:
        # Use random choice with fixed seed (handled by seed_everything)
        indices = np.random.choice(len(X_train), FAST_TRAIN_SAMPLES, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Create TensorDatasets
    # Note: We convert to tensors here. For very large data, this might consume RAM,
    # but 200k samples is manageable.
    train_dataset = TensorDataset(torch.tensor(X_train_sub), torch.tensor(y_train_sub))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test))

    # Create DataLoaders
    # We use the batch size from Config (4096) which is efficient for GPU
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

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    input_dim = X_train.shape[1]

    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    # Initialize Trainer
    trainer = Trainer(model, device, train_loader, val_loader)

    # Run training
    # fit() returns the model loaded with the best weights found during training
    best_model = trainer.fit(epochs=FAST_EPOCHS)

    # --------------------------------------------------------------------------
    # 5. Validation Assessment
    # --------------------------------------------------------------------------
    print("Performing final validation...")
    best_model.eval()

    correct = 0
    total = 0
    all_preds = []

    # Inference loop: No gradients, GPU usage
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = best_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_preds.append(predicted.cpu().numpy())

    final_acc = correct / total

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Concatenate all predictions
    y_pred_val = np.concatenate(all_preds)

    # Calculate error vector (1 if error, 0 if correct)
    errors = (y_pred_val != y_val).astype(int)

    # Calculate correlation between errors and input features
    # We iterate over columns of X_val
    correlations = []
    num_features = X_val.shape[1]

    for i in range(num_features):
        feat_col = X_val[:, i]
        # Check for constant columns to avoid division by zero in correlation
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append(corr)

    # Map to feature names
    feature_names = Config.CONTINUOUS_FEATURES + Config.BINARY_FEATURES

    # Create list of (name, correlation)
    if len(feature_names) == num_features:
        corr_pairs = list(zip(feature_names, correlations))
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top features correlated with prediction error:")
        for name, corr in corr_pairs[:5]:
            print(f"  {name}: {corr:.6f}")
    else:
        print(
            f"Warning: Number of features in data ({num_features}) does not match config names ({len(feature_names)}). Skipping detailed name mapping."
        )

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9625041666666667

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric ({final_acc:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        predict(best_model, test_loader, test_ids)
    else:
        print(
            f"\nValidation metric ({final_acc:.6f}) does not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
