import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import load_and_preprocess_data
from library.model import AVPFE
from library.train_utils import train_epoch, validate


def run():
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading and preprocessing data...")
    # Load datasets (uses cache if available)
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        load_cached_data=True
    )

    # Configure DataLoaders
    # Pin memory enables faster host-to-device transfer
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

    # 3. Model Initialization
    print("Initializing model...")
    # Determine categorical cardinalities by inspecting the max index in all datasets
    # This ensures embeddings cover the full vocabulary range
    max_cat_train = train_dataset.x_cat.max(dim=0).values
    max_cat_val = val_dataset.x_cat.max(dim=0).values
    max_cat_test = test_dataset.x_cat.max(dim=0).values

    global_max = (
        torch.stack([max_cat_train, max_cat_val, max_cat_test]).max(dim=0).values
    )
    cat_cardinalities = (global_max + 1).tolist()

    n_cont = train_dataset.x_cont.shape[1]

    print(
        f"Architecture: {len(cat_cardinalities)} categorical embeddings, {n_cont} continuous inputs."
    )

    model = AVPFE(cat_cardinalities, n_cont).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # OneCycleLR setup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_auc = validate(model, val_loader, device, criterion)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val AUC: {best_auc:.6f}")

    # 6. Final Validation & Failure Analysis
    print("Loading best model for detailed analysis...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions, targets, and inputs for analysis
    all_preds = []
    all_targets = []
    all_inputs_cont = []

    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            x_cat = x_cat.to(device)
            x_cont_gpu = x_cont.to(device)
            y = y.to(device)

            # Inference
            outputs = model(x_cat, x_cont_gpu)
            # Ensemble average of probabilities
            probs = torch.sigmoid(outputs).mean(dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy().flatten())
            # Store continuous inputs for correlation analysis
            all_inputs_cont.append(x_cont.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_inputs_cont = np.concatenate(all_inputs_cont, axis=0)

    # Compute and print Final Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc:.20f}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(all_targets - all_preds)

    # Reconstruct feature names to map correlations correctly
    try:
        # Load metadata schema
        df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH, nrows=1)
        # Simulate the feature engineering step to match column order
        df_meta["unique_char_count"] = 0

        # Define categorical columns to exclude from continuous list
        cat_cols = [
            f"f_27_{i}" for i in range(Config.N_F27_CHARS)
        ] + Config.DISCRETE_FEATURES
        cat_cols = sorted(cat_cols)
        exclude_cols = ["id", "target", "source_path", "f_27"] + cat_cols

        # Identify continuous columns and sort (matching data_utils logic)
        cont_cols = [c for c in df_meta.columns if c not in exclude_cols]
        cont_cols = sorted(cont_cols)

        if len(cont_cols) != all_inputs_cont.shape[1]:
            print("Warning: Feature name mismatch. Using generic indices.")
            cont_cols = [f"feat_{i}" for i in range(all_inputs_cont.shape[1])]
    except Exception as e:
        print(f"Could not derive feature names: {e}. Using generic indices.")
        cont_cols = [f"feat_{i}" for i in range(all_inputs_cont.shape[1])]

    # Compute correlations
    analysis_df = pd.DataFrame(all_inputs_cont, columns=cont_cols)
    analysis_df["error"] = errors
    correlations = (
        analysis_df.corr()["error"].drop("error").abs().sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Prediction Error:")
    print(correlations.head(5))

    # 7. Conditional Submission
    threshold = 0.9975746465492954
    if final_auc > threshold:
        print(
            f"\nValidation Metric ({final_auc:.6f}) exceeds threshold ({threshold}). Generating submission..."
        )

        test_preds = []
        model.eval()
        with torch.no_grad():
            for x_cat, x_cont in test_loader:
                x_cat = x_cat.to(device)
                x_cont = x_cont.to(device)

                outputs = model(x_cat, x_cont)
                probs = torch.sigmoid(outputs).mean(dim=1)
                test_preds.extend(probs.cpu().numpy())

        # Create submission dataframe
        try:
            df_test_ids = pd.read_csv(Config.TEST_METADATA_PATH, usecols=["id"])
            df_test_ids["target"] = test_preds

            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            df_test_ids.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
        except Exception as e:
            print(f"Error saving submission: {e}")
    else:
        print(
            f"\nValidation Metric ({final_auc:.6f}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
