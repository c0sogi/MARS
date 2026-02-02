import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.engine import train_model, predict


def main():
    # 1. Setup and Configuration
    # Set random seed for reproducibility
    seed_everything(Config.SEED)

    # Initialize configuration
    cfg = Config()

    # Adjust configuration for a fast baseline execution
    # Using Config defaults (15 epochs) for ResNet50

    print(
        f"Configuration: Model={cfg.MODEL_NAME}, Epochs={cfg.NUM_EPOCHS}, Device={cfg.DEVICE}"
    )

    # 2. Training
    print("\n=== Starting Training ===")
    # train_model handles the training loop, validation, and saving the best model
    trained_model = train_model(cfg)

    # 3. Validation Assessment
    print("\n=== Starting Validation Assessment ===")
    device = torch.device(cfg.DEVICE)

    # Get dataloaders (we only need val_loader here)
    _, val_loader, _ = get_dataloaders(cfg)

    # Load validation metadata to align with file paths later
    val_df = pd.read_csv(cfg.VAL_CSV)
    if cfg.DEBUG:
        val_df = val_df.head(cfg.DEBUG_SAMPLE_SIZE)

    # Ensure model is in eval mode
    trained_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # Probability of the true class

    # Inference loop on validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = trained_model(images)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Gather probability assigned to the true label for error magnitude calculation
            # gather expects index to have same dim as input, so we view labels
            true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()
            # Handle case where batch size is 1 (squeeze might remove too many dims)
            if true_probs.ndim == 0:
                true_probs = true_probs.unsqueeze(0)
            all_probs.extend(true_probs.cpu().numpy())

    # Calculate Accuracy
    val_accuracy = accuracy_score(all_labels, all_preds)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {val_accuracy}")

    # 4. Failure Analysis
    print("\n=== Starting Failure Analysis ===")

    # Calculate Error Magnitude: 1.0 - Probability(True Class)
    # High error magnitude means the model assigned low probability to the correct class.
    error_magnitudes = 1.0 - np.array(all_probs)

    # Extract Input Features: File Size
    # Since image dimensions are constant (800x600), file size is a proxy for complexity/compression.
    file_sizes = []
    print("Extracting file sizes for validation set...")
    for _, row in val_df.iterrows():
        full_path = os.path.join(cfg.INPUT_DIR, row["file_path"])
        try:
            size = os.path.getsize(full_path)
        except Exception:
            size = 0
        file_sizes.append(size)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {"error_magnitude": error_magnitudes, "file_size": file_sizes}
    )

    # Calculate Correlation
    correlation = analysis_df["error_magnitude"].corr(analysis_df["file_size"])

    print(f"Correlation between Error Magnitude and File Size: {correlation:.10f}")
    if abs(correlation) < 0.1:
        print(
            "Observation: Little to no linear relationship between image file size and model error."
        )
    elif correlation > 0:
        print(
            "Observation: Larger file sizes (potentially more complex images) are associated with higher error."
        )
    else:
        print("Observation: Smaller file sizes are associated with higher error.")

    # 5. Submission
    print("\n=== Generating Submission ===")
    # The predict function loads the best model from disk and generates submission.csv
    if val_accuracy > 0.859012016:
        print(
            f"Validation accuracy {val_accuracy} > 0.859012016. Generating submission..."
        )
        predict(cfg)
    else:
        print(
            f"Validation accuracy {val_accuracy} <= 0.859012016. Skipping submission."
        )

    print("\nRunfile execution complete.")


if __name__ == "__main__":
    main()
