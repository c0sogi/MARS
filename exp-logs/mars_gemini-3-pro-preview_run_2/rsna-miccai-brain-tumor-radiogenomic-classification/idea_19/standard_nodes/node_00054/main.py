import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from provided library files
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train_eval import run_training, generate_submission, validate


def perform_failure_analysis(model, val_loader, device, val_df):
    """
    Analyzes model errors on the validation set and correlates them with metadata features.
    """
    print("Performing failure analysis...")
    model.eval()

    all_preds = []
    all_targets = []

    # 1. Generate predictions for validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)

            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate absolute error
    errors = np.abs(all_preds - all_targets)

    # 2. Extract metadata features (Slice Counts) for correlation analysis
    # We use the order from val_df which matches val_loader (shuffle=False)
    input_dir = "./input"
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    print("Error Correlations with Metadata Features:")

    for mod in modalities:
        col_path = f"path_{mod}"
        counts = []

        # Extract slice count for each subject in validation set
        for rel_path in val_df[col_path]:
            full_path = os.path.join(input_dir, rel_path)
            try:
                # Fast count of files
                if os.path.exists(full_path):
                    n_slices = len(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                    )
                else:
                    n_slices = 0
            except Exception:
                n_slices = 0
            counts.append(n_slices)

        counts = np.array(counts)

        # Calculate Pearson correlation coefficient
        if np.std(counts) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(counts, errors)[0, 1]
        else:
            corr = 0.0

        print(f"{mod}_slice_count: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()

    # Define paths
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    # 2. Train Model
    # We use 15 epochs to ensure a good baseline within the time limit.
    # The dataset is small (~500), so this is fast.
    print("Starting training pipeline...")
    run_training(
        epochs=15,
        batch_size=32,
        save_path=model_save_path,
        patience=5,
        debug_limit=None,  # Use full dataset
    )

    # 3. Validation & Metrics
    # Reload the best model for evaluation
    model = AsymmetricEfficientNet(pretrained=False).to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("Error: Model file not found after training.")
        return

    # Get validation dataloader
    # Note: get_dataloaders handles caching, so this is efficient
    _, val_loader, _ = get_dataloaders(batch_size=32, load_cached_data=True)

    # Compute final metric
    criterion = nn.BCEWithLogitsLoss()
    _, final_auc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    # Access the dataframe underlying the validation loader to map features
    val_df = val_loader.dataset.df
    perform_failure_analysis(model, val_loader, device, val_df)

    # 5. Submission Generation
    # Threshold defined in task description
    threshold = 0.6254545454545455

    if final_auc > threshold:
        print(f"Validation metric {final_auc} > {threshold}. Generating submission...")
        generate_submission(
            model_path=model_save_path, output_path=submission_path, batch_size=32
        )
    else:
        print(f"Validation metric {final_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
