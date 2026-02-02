import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging
import sys

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib
import library.predict as predict_lib


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    # Suppress library logs to meet "Only print the required information" constraint
    logging.getLogger("data_loader").setLevel(logging.ERROR)
    logging.getLogger("model").setLevel(logging.ERROR)
    logging.getLogger("train").setLevel(logging.ERROR)
    logging.getLogger("predict").setLevel(logging.ERROR)

    # 2. Data Loading
    # We use the full dataset (debug=False) because the dataset size is small (approx 500 samples),
    # allowing for fast training (<10 mins) while maximizing the chance to pass the AUC threshold.
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(debug=False)

    # 3. Model Initialization
    model = model_lib.AsymmetricEfficientNet().to(device)

    # 4. Training
    # Initialize Trainer with the model and loaders
    # We use the default configuration from config.py (20 epochs), which is optimal for a fast baseline.
    trainer = train_lib.Trainer(
        model, train_loader, val_loader, device, corruption_threshold=0.01
    )

    # Execute training loop
    try:
        trainer.fit()
    except RuntimeError as e:
        print(f"Training aborted due to error: {e}")
        return

    # 5. Validation Assessment
    # Load the best model saved during training
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_targets = []
    all_preds = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds).flatten()

    # Calculate and print Final Validation Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Extract Feature: FLAIR Slice Count (Volume Depth)
    # We iterate through the validation dataframe to get file paths and count slices
    val_df = val_loader.dataset.df
    flair_slice_counts = []

    for _, row in val_df.iterrows():
        flair_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])
        count = 0
        if os.path.exists(flair_path):
            try:
                # Fast directory listing
                count = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
            except Exception:
                count = 0
        flair_slice_counts.append(count)

    flair_slice_counts = np.array(flair_slice_counts)

    # Calculate Correlation
    if len(errors) == len(flair_slice_counts) and len(errors) > 1:
        # Pearson correlation
        correlation = np.corrcoef(errors, flair_slice_counts)[0, 1]
        print(
            f"Correlation between Error Magnitude and FLAIR_slices: {correlation:.6f}"
        )
    else:
        print(
            "Correlation between Error Magnitude and FLAIR_slices: NaN (Insufficient data)"
        )

    # 7. Submission Generation
    # Only generate submission if metric threshold is met
    THRESHOLD = 0.6321818181818182

    if val_auc > THRESHOLD:
        # Call the provided prediction routine which handles TTA and saving
        predict_lib.generate_submission(debug=False)


if __name__ == "__main__":
    main()
