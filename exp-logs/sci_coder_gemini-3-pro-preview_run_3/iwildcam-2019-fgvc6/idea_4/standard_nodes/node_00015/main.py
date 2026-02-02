import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.predict import run_inference
from library.model import AnimalModel
from library.dataset import AnimalDataset, get_transforms
from library.utils import seed_everything


def perform_failure_analysis(device):
    """
    Performs failure analysis on the validation set by correlating
    prediction error magnitude with input features (Category, Class Frequency).
    """
    print("\n--- Performing Failure Analysis ---")

    # Load Validation Metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found.")
        return

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Load Train Metadata to calculate class frequency
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    class_counts = train_df["Category"].value_counts().to_dict()

    # Prepare Dataset & Loader
    # We use 'valid' transforms (resize + normalize)
    val_dataset = AnimalDataset(
        val_df, transforms=get_transforms("valid"), mode="valid"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = AnimalModel(pretrained=False)
    checkpoint_path = Config.MODEL_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found, skipping failure analysis.")
        return

    print(f"Loading weights for analysis from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_errors = []
    all_categories = []

    # Inference loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)

            # Get the probability assigned to the ground truth class
            # gather requires index tensor to have same dims as input except at dim
            true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()

            # Define Error Magnitude:
            # 0.0 means model was 100% confident in the correct class.
            # 1.0 means model was 0% confident in the correct class.
            batch_errors = 1.0 - true_probs.cpu().numpy()

            all_errors.extend(batch_errors)
            all_categories.extend(labels.cpu().numpy())

    # Create Analysis DataFrame
    df_analysis = pd.DataFrame(
        {"ErrorMagnitude": all_errors, "Category": all_categories}
    )

    # Map class frequency
    df_analysis["ClassFrequency"] = df_analysis["Category"].map(class_counts)

    # Calculate Correlations
    # Correlation between Error and Category ID (nominal, but requested)
    corr_cat = df_analysis["ErrorMagnitude"].corr(df_analysis["Category"])

    # Correlation between Error and Class Frequency (tests if rare classes have higher error)
    corr_freq = df_analysis["ErrorMagnitude"].corr(df_analysis["ClassFrequency"])

    print(f"Correlation between Error Magnitude and Category: {corr_cat}")
    print(f"Correlation between Error Magnitude and Class Frequency: {corr_freq}")

    # Summary of worst classes
    print("\nTop 5 Classes with highest mean error magnitude:")
    mean_errors = (
        df_analysis.groupby("Category")["ErrorMagnitude"]
        .mean()
        .sort_values(ascending=False)
    )
    print(mean_errors.head(5))


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Train the model
    # We use debug=False to train on the full dataset to achieve the high F1 score required.
    # The A100 GPU can handle 10 epochs on this dataset size within the time limit.
    print("Starting Training Pipeline...")
    best_f1 = run_training(debug=False, epochs=Config.EPOCHS)

    # 2. Print the Final Validation Metric in the required format
    print(f"Final Validation Metric: {best_f1}")

    # 3. Perform Failure Analysis
    device = torch.device(Config.DEVICE)
    perform_failure_analysis(device)

    # 4. Generate Submission if Threshold is Met
    THRESHOLD = 0.9314033053214555

    if best_f1 > THRESHOLD:
        print(
            f"\nValidation F1 ({best_f1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nValidation F1 ({best_f1}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
