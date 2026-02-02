import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

# Import from provided library
from library.config import Config
from library.train import run_training, generate_submission
from library.model import SpeciesModel
from library.dataset import get_loaders
from library.utils import set_seed


def main():
    # 1. Configure for Full Training
    # We use the full dataset to ensure frequent classes are not under-represented.
    # Cite solution_lesson_node_00001: The previous lesson showed that capping samples per class
    # (via stratified sampling) led to higher error rates on frequent classes (positive correlation).
    # To fix this, we use the full dataset, allowing the model to learn from the natural frequency of head classes.
    Config.NUM_EPOCHS = 12
    Config.DEBUG_SAMPLE_SIZE = None
    Config.BATCH_SIZE = 128

    print("--- Starting Full Training ---")
    # run_training handles the training loop, validation, and checkpointing
    run_training(num_epochs=Config.NUM_EPOCHS)

    # 2. Prepare for Full Validation and Submission
    print("--- Loading Full Datasets for Final Evaluation ---")

    # Re-initialize loaders (though they are already full, this ensures consistency)
    train_loader, val_loader, test_loader = get_loaders()

    # Load the best model trained in step 1
    device = torch.device(Config.DEVICE)
    model = SpeciesModel()
    model = model.to(device)

    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Error: Checkpoint not found! Cannot proceed with evaluation.")
        return

    model.eval()

    # 3. Final Validation on Entire Hold-out Set
    print("--- Performing Final Validation ---")
    # We need to collect predictions and targets for failure analysis
    all_preds = []
    all_targets = []

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            correct_count += (preds == targets).sum().item()
            total_count += targets.size(0)

    # Calculate Top-1 Error
    # Metric is Error Rate (0.0 to 1.0)
    final_error = 1.0 - (correct_count / total_count)

    print(f"Final Validation Metric: {final_error:.16f}")

    # 4. Failure Analysis
    print("--- Performing Failure Analysis ---")
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    errors = (all_preds != all_targets).astype(int)  # 1 if error, 0 if correct

    # Analysis: Correlation with Class Frequency
    # Load train metadata to get class counts
    # We need to reconstruct the mapping logic used in dataset.py to ensure indices match
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    unique_categories = sorted(train_df["category_id"].unique())
    cat_to_idx = {cat: i for i, cat in enumerate(unique_categories)}

    # Calculate frequency per internal index
    train_df["label_idx"] = train_df["category_id"].map(cat_to_idx)
    class_counts = train_df["label_idx"].value_counts().to_dict()

    # Map counts to validation samples based on their target label
    # This tells us how many training samples existed for the class of the validation image
    val_sample_counts = np.array([class_counts.get(t, 0) for t in all_targets])

    # Calculate Point Biserial Correlation
    # Correlation between binary variable 'errors' and continuous variable 'val_sample_counts'
    if len(np.unique(errors)) > 1:
        corr, p_value = pointbiserialr(errors, val_sample_counts)
        print(
            f"Correlation between Error and Class Frequency: {corr:.6f} (p-value: {p_value:.6f})"
        )
        if corr < 0:
            print(
                "Observation: Negative correlation implies rare classes have higher error rates."
            )
        else:
            print(
                "Observation: Positive correlation implies frequent classes have higher error rates."
            )
    else:
        print("Correlation could not be computed (insufficient variance in errors).")

    # 5. Generate Submission on Full Test Set
    THRESHOLD = 0.3978755364806867
    if final_error < THRESHOLD:
        print("--- Generating Final Submission ---")
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE_PATH)
    else:
        print(
            f"Final metric {final_error:.6f} is not lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
