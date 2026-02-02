import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import warnings

# Ensure library imports work
sys.path.append(os.getcwd())

from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import ParallelDCN_ResNet
from library.train import train_epoch, validate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between features and error probability.
    """
    model.eval()
    all_inputs = []
    all_targets = []
    all_preds = []

    # Collect data without gradients
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Move to CPU for analysis
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    # Concatenate all batches
    X_val = np.vstack(all_inputs)
    y_val = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    # Identify errors (1 for error, 0 for correct)
    errors = (y_val != y_pred).astype(int)

    print("\n==== Failure Analysis ====")
    print(f"Total Validation Samples: {len(errors)}")
    print(f"Total Errors: {errors.sum()}")
    print(f"Error Rate: {errors.mean():.6f}")

    # Calculate correlations between each feature and the error flag
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = X_val[:, i]
        # Handle constant features to avoid NaN correlations
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features correlated with Error:")
    print(f"{'Feature Index':<15} {'Correlation':<15}")
    print("-" * 30)
    for idx, corr in correlations[:10]:
        print(f"{idx:<15} {corr:.6f}")
    print("==========================\n")


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # Configuration
    BATCH_SIZE = 4096
    EPOCHS = 50  # Increased epochs for standard ResNet convergence
    THRESHOLD = 0.9625041666666667

    # 2. Data Loading
    # quick_run=False is required to use the full dataset and meet the high accuracy threshold
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE,
        load_cached_data=True,
        cache_dir="./working/idea_11/",
        quick_run=False,
    )

    # Determine input dimensions
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    num_classes = 7  # Mapped to 0-6 internally

    # 3. Model Initialization
    model = ParallelDCN_ResNet(input_dim, num_classes).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 8
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
        )

        scheduler.step(val_acc)

        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Final Metric & Analysis
    print(f"Final Validation Metric: {best_acc}")

    # Load best weights for analysis and inference
    model.load_state_dict(best_model_wts)

    # Run Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    if best_acc > THRESHOLD:
        print(f"Validation accuracy {best_acc} > {THRESHOLD}. Generating submission...")

        model.eval()
        preds = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())

        # Remap 0-6 back to 1-7
        final_preds = np.array(preds) + 1

        os.makedirs("./submission", exist_ok=True)
        sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path} with {len(sub_df)} rows.")

    else:
        print(
            f"Validation accuracy {best_acc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
