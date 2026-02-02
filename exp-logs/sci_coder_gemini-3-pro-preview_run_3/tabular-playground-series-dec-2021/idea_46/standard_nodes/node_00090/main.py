import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import from provided library files
from library.utils import seed_everything, get_device, save_submission
from library.data_processing import get_dataloaders
from library.model import AsymmetricParallelNet, predict
from library.train import run_training, validate


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()

    # 2. Data Loading
    # Using cached data for speed and efficiency
    # num_workers=2 ensures fast data feeding without excessive overhead
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=4096, load_cached_data=True, num_workers=2
    )

    # 3. Model Initialization
    # Get input dimension from a sample batch to initialize the model correctly
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    num_classes = 7  # Internal class mapping 0-6

    model = AsymmetricParallelNet(input_dim, num_classes).to(device)

    # 4. Training
    # Fast baseline execution: 5 epochs are estimated to take ~100 seconds on A100,
    # fitting well within the 3-minute limit while allowing convergence.
    model = run_training(
        model, train_loader, val_loader, device, epochs=5, lr=1e-3, patience=3
    )

    # 5. Validation Assessment
    # Calculate final metric on the full validation set
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Required Output Format
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()

    all_inputs = []
    all_preds = []
    all_labels = []

    # Collect data for analysis
    # Disable gradient calculation to save memory and computation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_inputs.append(inputs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    X_val = np.concatenate(all_inputs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Error vector: 1 if prediction is wrong, 0 if correct
    errors = (y_pred != y_true).astype(int)

    # Calculate correlations between each feature and the error vector
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Compute Pearson correlation
        # Handle constant features to avoid division by zero
        if np.std(X_val[:, i]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X_val[:, i], errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")

    # 7. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 0.9626291666666666

    if val_acc > THRESHOLD:
        print(f"Validation metric {val_acc} > {THRESHOLD}. Generating submission...")
        predictions = predict(model, test_loader, device)
        save_submission(predictions, test_ids, "./submission/submission.csv")
    else:
        print(f"Validation metric {val_acc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
