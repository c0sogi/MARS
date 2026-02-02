import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch

# Append current directory to system path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import get_dataloaders
from library.model import CatheterModel
from library.train import run_training
from library.predict import inference_fn


def main():
    # --- 1. Setup & Configuration ---
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set random seeds for reproducibility
    seed_everything(Config.seed)

    print("==================================================")
    print("Running Catheter Detection Pipeline")
    print("==================================================")

    # --- 2. Training ---
    # We execute the training routine using the full dataset as per the Idea.
    # The Config defines 6 epochs, which fits within the time limit.
    print("\n[Step 1/4] Starting Model Training...")

    # run_training returns the best AUC achieved during training (non-TTA)
    # We ensure debug is False to use the full dataset.
    _ = run_training(
        debug=False,
        num_epochs=Config.num_epochs,
        batch_size=Config.batch_size,
        model_save_path=Config.model_save_path,
    )

    # --- 3. Validation & Metric Calculation ---
    print("\n[Step 2/4] Performing Final Validation...")

    # Load metadata files
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    # We need to recreate the validation loader.
    # We pass the dataframes to get_dataloaders, but we only use val_loader.
    _, val_loader, _ = get_dataloaders(
        train_df, val_df, test_df, batch_size=Config.batch_size
    )

    # Load the best saved model
    device = Config.device
    model = CatheterModel(
        model_name=Config.model_name,
        pretrained=False,  # Loading custom weights
        num_classes=Config.num_classes,
        in_channels=Config.in_channels,
    )

    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError(f"Model file not found: {Config.model_save_path}")

    print(f"Loading model weights from {Config.model_save_path}...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    y_true_list = []
    y_pred_list = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Calculate Final Metric
    final_metric = get_score(y_true, y_pred)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # --- 4. Failure Analysis ---
    print("\n[Step 3/4] Performing Failure Analysis...")

    # Calculate absolute error per sample (mean across all classes)
    # Shape: (N_samples, N_classes) -> (N_samples,)
    abs_errors = np.abs(y_true - y_pred)
    mean_error_per_sample = np.mean(abs_errors, axis=1)

    print("Correlation between Error Magnitude and Target Presence:")
    # We calculate the correlation between the presence of each label and the error magnitude.
    # This tells us if specific catheter conditions are harder to predict.
    correlations = []
    for idx, col_name in enumerate(Config.target_cols):
        # Get the binary labels for this column
        col_labels = y_true[:, idx]

        # Calculate correlation if there is variance in the labels
        if np.std(col_labels) > 0:
            corr = np.corrcoef(col_labels, mean_error_per_sample)[0, 1]
            correlations.append((col_name, corr))
        else:
            correlations.append((col_name, 0.0))

    # Sort by correlation (descending) to show strongest associations first
    correlations.sort(key=lambda x: x[1], reverse=True)

    for name, corr in correlations:
        print(f"  {name}: {corr:.6f}")

    # --- 5. Submission ---
    print("\n[Step 4/4] Checking Submission Threshold...")

    THRESHOLD = 0.9529163070786033
    SUBMISSION_DIR = "./submission"

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensure output directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Run inference on test set
        inference_fn(
            test_metadata_path=Config.test_metadata_path,
            model_path=Config.model_save_path,
            submission_output_dir=SUBMISSION_DIR,
            batch_size=Config.batch_size,
            debug=False,
            device=device,
        )
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
