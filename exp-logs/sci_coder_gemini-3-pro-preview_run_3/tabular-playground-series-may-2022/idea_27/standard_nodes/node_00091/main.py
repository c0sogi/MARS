import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import TRAIN_PARAMS, STREAM_CONFIGS, CACHE_DIR
from library.utils import seed_everything, compute_roc_auc
from library.preprocessing import process_data
from library.dataset import ManufacturingDataset
from library.model import PIFEModel
from library.engine import train_model, generate_submission, predict


def run():
    # 1. Setup
    seed_everything(TRAIN_PARAMS["seed"])
    device = torch.device(TRAIN_PARAMS["device"])

    # 2. Data Loading
    # Using load_cached_data=True to utilize preprocessed parquet files
    train_df, val_df, test_df, metadata = process_data(load_cached_data=True)

    # Dataset creation
    # We use the full dataset (max_samples=None) to ensure we can hit the high AUC target.
    train_ds = ManufacturingDataset(
        train_df, metadata["cont_cols"], metadata["cat_cols"], target_col="target"
    )
    val_ds = ManufacturingDataset(
        val_df, metadata["cont_cols"], metadata["cat_cols"], target_col="target"
    )
    test_ds = ManufacturingDataset(
        test_df, metadata["cont_cols"], metadata["cat_cols"], target_col=None
    )

    # DataLoader creation
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )

    # 3. Model Initialization
    model = PIFEModel(
        num_cont=len(metadata["cont_cols"]),
        cat_cardinalities=metadata["cat_cardinalities"],
        stream_configs=STREAM_CONFIGS,
    ).to(device)

    # 4. Training Configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_PARAMS["learning_rate"],
        weight_decay=TRAIN_PARAMS["weight_decay"],
    )

    # Cite {lesson_node_00080}: Extended schedule to 50 epochs for full convergence of regularized ensemble.
    epochs = TRAIN_PARAMS["epochs"]

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_PARAMS["learning_rate"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()
    save_path = os.path.join(CACHE_DIR, "best_model_runfile.pth")

    # 5. Training Execution
    model, best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=epochs,
        patience=5,
        save_path=save_path,
    )

    # 6. Final Validation & Metric
    # Ensure we are using the best model state
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Get predictions on validation set for final metric and failure analysis
    val_probs = predict(model, val_loader, device)
    val_targets = val_ds.target

    final_val_auc = compute_roc_auc(val_targets, val_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_targets - val_probs)

    # Create a DataFrame with features and error for correlation analysis
    analysis_df = val_df[metadata["cont_cols"] + metadata["cat_cols"]].copy()
    analysis_df["error"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Top 5 features correlated with error:")
    print(correlations.head(5))

    # 8. Submission
    threshold = 0.9975746465492954

    if final_val_auc > threshold:
        print(
            f"Validation metric {final_val_auc} exceeds threshold {threshold}. Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        test_ids = test_df["id"].values
        generate_submission(model, test_loader, test_ids, device, submission_path)
        print("Submission saved to ./submission/submission.csv")
    else:
        print(
            f"Validation metric {final_val_auc} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
