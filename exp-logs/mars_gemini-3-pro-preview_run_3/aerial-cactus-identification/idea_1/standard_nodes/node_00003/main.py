import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import functions and classes from the provided library files
from library.utils import seed_everything, get_device
from library.model import SimpleCNN, predict, validate
from library.dataset import create_dataloaders
from library.train import fit_model


def perform_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set by correlating
    prediction errors with input image features (brightness and contrast).
    """
    print("\n==== FAILURE ANALYSIS ====")

    # 1. Get model predictions on the validation set
    # predict() returns a flattened numpy array of probabilities
    val_probs = predict(model, val_loader, device)

    # 2. Get ground truth labels and raw images from the dataset
    # Accessing the underlying arrays directly from the CactusDataset
    val_images = val_loader.dataset.images  # Shape: (N, 32, 32, 3), dtype: uint8
    val_labels = val_loader.dataset.labels  # Shape: (N,), dtype: float32

    # Ensure alignment
    if len(val_probs) != len(val_labels):
        print(
            f"Error: Mismatch between predictions ({len(val_probs)}) and labels ({len(val_labels)})."
        )
        return

    # 3. Calculate Error Magnitude
    # Error is the absolute difference between the predicted probability and the true label
    errors = np.abs(val_probs - val_labels)

    # 4. Extract Image Meta-features
    # Flatten images to (N, Pixels) to easily compute stats per image
    flat_images = val_images.reshape(val_images.shape[0], -1)

    # Brightness: Mean pixel intensity
    brightness = flat_images.mean(axis=1)

    # Contrast: Standard deviation of pixel intensity
    contrast = flat_images.std(axis=1)

    # 5. Calculate Correlations
    # We use Pearson correlation to see if error scales linearly with these features
    corr_brightness, _ = pearsonr(errors, brightness)
    corr_contrast, _ = pearsonr(errors, contrast)

    print(f"Correlation between Error and Brightness: {corr_brightness}")
    print(f"Correlation between Error and Contrast: {corr_contrast}")


def main():
    # 1. Setup Environment
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    SUBMISSION_DIR = "./submission"

    # Hyperparameters for a fast baseline
    BATCH_SIZE = 64
    EPOCHS = 10
    PATIENCE = 3
    LEARNING_RATE = 1e-3

    # 2. Load Data
    print("Loading data...")
    # create_dataloaders handles metadata loading and caching
    train_loader, val_loader, test_loader, test_ids = create_dataloaders(
        batch_size=BATCH_SIZE,
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        load_cached_data=True,
        num_workers=2,
    )

    # 3. Initialize Model, Loss, and Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train Model
    # fit_model manages the training loop, validation, and early stopping
    model = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    # 5. Final Validation
    # Evaluate the best model on the validation set
    _, final_auc = validate(model, val_loader, criterion, device)
    # Print the metric in the exact required format
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Generate Submission
    print("Generating predictions for test set...")
    test_probs = predict(model, test_loader, device)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": test_probs})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
