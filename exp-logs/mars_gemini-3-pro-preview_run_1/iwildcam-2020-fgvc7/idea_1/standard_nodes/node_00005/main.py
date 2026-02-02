import torch
import numpy as np
import pandas as pd
import os
import warnings
from library.trainer import Trainer
from library.utils import set_seed
from library.config import SEED

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration for Full Training
    # Using full dataset and more epochs for better performance
    MAX_SAMPLES = None
    EPOCHS = 15

    # Set reproducibility
    set_seed(SEED)

    # 2. Initialize Trainer
    # load_cached_data=True allows using pre-computed bounding boxes if available
    print(
        f"Initializing pipeline with MAX_SAMPLES={MAX_SAMPLES} and EPOCHS={EPOCHS}..."
    )
    trainer = Trainer(load_cached_data=True, max_samples=MAX_SAMPLES)

    # 3. Train the model
    trainer.fit(epochs=EPOCHS)

    # 4. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model weights for evaluation
    if os.path.exists(trainer.best_model_path):
        print(f"Loading best model from {trainer.best_model_path}...")
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: Best model not found. Using current model state.")

    # Ensure model is in eval mode
    trainer.model.eval()
    device = trainer.device
    val_loader = trainer.val_loader

    all_preds = []
    all_labels = []

    # Run inference on validation set
    # We disable gradients for speed and memory efficiency
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = trainer.model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute Final Validation Metric (Accuracy)
    accuracy = (all_preds == all_labels).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    # We correlate error with:
    # A) Whether the image is empty (Category 0)
    # B) Class frequency in training data

    # Get the validation dataframe corresponding to the processed samples
    # Since shuffle=False in val_loader, the order matches the dataset dataframe
    val_df = trainer.val_dataset.df.copy()

    # Safety check for length alignment (in case of dropped batches, though unlikely with default settings)
    if len(val_df) != len(all_preds):
        min_len = min(len(val_df), len(all_preds))
        val_df = val_df.iloc[:min_len]
        all_preds = all_preds[:min_len]
        all_labels = all_labels[:min_len]

    val_df["pred"] = all_preds
    val_df["label"] = all_labels
    val_df["error"] = (val_df["pred"] != val_df["label"]).astype(int)

    # Feature 1: Is Empty (Category 0)
    val_df["is_empty"] = (val_df["label"] == 0).astype(int)

    # Feature 2: Class Frequency
    # Map class IDs to their frequency in the training set
    train_df = trainer.train_dataset.df
    class_counts = train_df["category_id"].value_counts().to_dict()
    val_df["class_freq"] = val_df["label"].map(lambda x: class_counts.get(x, 0))

    # Calculate Correlations
    # Handle edge cases where standard deviation is 0 (e.g., if subset only has one class)
    if val_df["is_empty"].std() > 1e-9:
        corr_empty = val_df["error"].corr(val_df["is_empty"])
    else:
        corr_empty = 0.0

    if val_df["class_freq"].std() > 1e-9:
        corr_freq = val_df["error"].corr(val_df["class_freq"])
    else:
        corr_freq = 0.0

    print("-" * 30)
    print("Failure Analysis Results:")
    print(f"Correlation (Error vs Is_Empty): {corr_empty}")
    print(f"Correlation (Error vs Class_Freq): {corr_freq}")
    print("-" * 30)

    # 5. Generate Submission
    # Only submit if we improved over the baseline
    if accuracy > 0.7259454817265181:
        print(f"Validation accuracy {accuracy:.4f} > 0.7259. Generating submission...")
        trainer.generate_submission()
    else:
        print(f"Validation accuracy {accuracy:.4f} <= 0.7259. Skipping submission.")


if __name__ == "__main__":
    main()
