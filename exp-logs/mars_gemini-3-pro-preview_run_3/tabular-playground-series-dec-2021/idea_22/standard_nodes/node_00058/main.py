import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders
from library.model_arch import HybridModel
from library.train_utils import Trainer, set_seed


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Data
    # We use load_cached_data=True to leverage the preprocessed files in ./working
    print("Loading data...")
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Initialize Model
    print("Initializing HybridModel...")
    model = HybridModel(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        resnet_depth=Config.RESNET_DEPTH,
        resnet_width=Config.RESNET_WIDTH,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Setup Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Using parameters from Config for the scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Fit the model (handles training loop and early stopping)
    # The trainer automatically loads the best model weights upon completion
    model = trainer.fit(
        epochs=Config.EPOCHS, early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )

    # 6. Final Validation & Failure Analysis
    print("Performing final validation and failure analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []

    correct = 0
    total = 0

    # Disable gradients for inference
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Store for failure analysis
            # Move to CPU to save GPU memory, though 40GB is plenty for batches
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_inputs.append(X.cpu().numpy())

    final_acc = correct / total
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis: Correlation between Error and Features
    print("\n--- Failure Analysis ---")
    try:
        # Concatenate all batches
        X_val_full = np.vstack(all_inputs)
        y_val_full = np.concatenate(all_targets)
        preds_val_full = np.concatenate(all_preds)

        # Calculate Error Vector (1 if wrong, 0 if correct)
        errors = (preds_val_full != y_val_full).astype(int)

        # Calculate correlation for each feature
        n_features = X_val_full.shape[1]
        correlations = []

        for i in range(n_features):
            # Calculate point-biserial correlation roughly equivalent to Pearson here
            # Check for zero variance to avoid division by zero
            feat_col = X_val_full[:, i]
            if np.std(feat_col) > 1e-9 and np.std(errors) > 1e-9:
                corr = np.corrcoef(feat_col, errors)[0, 1]
                correlations.append((i, corr))
            else:
                correlations.append((i, 0.0))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 5 Features correlated with Error:")
        for idx, corr in correlations[:5]:
            print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # 7. Submission
    threshold = 0.9625041666666667
    if final_acc > threshold:
        print(
            f"\nValidation metric ({final_acc}) > threshold ({threshold}). Generating submission..."
        )

        predictions = []
        model.eval()

        with torch.no_grad():
            for X in test_loader:
                X = X.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                # Map 0-6 back to 1-7 (add 1)
                preds = predicted.cpu().numpy() + 1
                predictions.extend(preds)

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_acc}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
