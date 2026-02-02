import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.data_processor import DataProcessor
from library.dataset import ManufacturingDataset
from library.model import GUTClassifier, set_seed
from library.engine import train_model, validate, generate_submission


def main():
    # -------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------
    # Using debug=False to use the full dataset and achieve the target AUC.
    # The A100 GPU is sufficient to run this within the time limit.
    config = Config(debug=False)

    # Set seeds for reproducibility
    set_seed(config.seed)

    # Device configuration
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------
    print("Initializing DataProcessor...")
    processor = DataProcessor(config)

    # Load data (uses cache if available)
    data = processor.process_data(load_cached_data=True)

    # -------------------------------------------------------------------
    # 3. Dataset & DataLoader Creation
    # -------------------------------------------------------------------
    print("Creating DataLoaders...")
    train_dataset = ManufacturingDataset(
        data["X_num_train"], data["X_seq_train"], data["y_train"]
    )
    val_dataset = ManufacturingDataset(
        data["X_num_val"], data["X_seq_val"], data["y_val"]
    )
    test_dataset = ManufacturingDataset(data["X_num_test"], data["X_seq_test"], None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------
    print("Initializing Model...")
    model = GUTClassifier(config).to(device)

    # -------------------------------------------------------------------
    # 5. Optimization Setup
    # -------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Scheduler setup
    total_steps = config.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )

    # -------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------
    best_auc = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device, config
    )

    # -------------------------------------------------------------------
    # 7. Final Validation & Failure Analysis
    # -------------------------------------------------------------------
    print("\nLoading best model for final validation and analysis...")
    best_model_path = os.path.join(config.working_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    model.eval()

    # Compute final metrics and get predictions for analysis
    print("Computing final validation metrics...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)
            target = batch["target"].to(device)

            logits = model(x_num, x_seq)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    final_val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = np.abs(all_targets - all_preds)

    # Calculate correlation between error magnitude and numerical features
    X_val_num = data["X_num_val"]
    feature_names = config.numerical_features

    correlations = []
    for i, feat_name in enumerate(feature_names):
        # Extract column i from validation data
        feat_values = X_val_num[:, i]

        # Compute correlation
        if np.std(feat_values) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        else:
            corr = 0.0
        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------
    threshold = 0.9961695560998911

    if final_val_auc > threshold:
        print(
            f"\nValidation AUC ({final_val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, data["ids_test"], device, config)
    else:
        print(
            f"\nValidation AUC ({final_val_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
