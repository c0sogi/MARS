import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config, set_seed
from library.trainer import Trainer
from library.inference import generate_submission
from library.dataset import SETIDataset
from library.model import ShallowCNN


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Enforce fast baseline constraints
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10000  # Train on a subset for speed
    Config.EPOCHS = 5  # Limit epochs to ensure completion within time

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Configuration:")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    trainer = Trainer()
    trainer.fit()

    # --------------------------------------------------------------------------
    # 3. Full Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Full Validation & Failure Analysis ---")

    # Load the best model
    model_path = trainer.best_model_path
    if not os.path.exists(model_path):
        print("Warning: Best model not found. Using current model state.")
        model = trainer.model
    else:
        model = ShallowCNN()
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

    model.eval()

    # Load Full Validation Set (Ignoring DEBUG flag for this step to get true metric)
    val_dataset = SETIDataset(metadata_path=Config.VAL_METADATA)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_targets = []
    all_probs = []

    # Features for failure analysis
    feature_means = []
    feature_stds = []
    feature_maxs = []

    print(f"Validating on {len(val_dataset)} samples...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

            # Extract simple features for failure analysis
            # Compute stats per image in the batch
            # images: (B, C, H, W)
            imgs_np = images.cpu().numpy()
            # Flatten spatial dims: (B, C*H*W)
            imgs_flat = imgs_np.reshape(imgs_np.shape[0], -1)

            feature_means.extend(np.mean(imgs_flat, axis=1))
            feature_stds.extend(np.std(imgs_flat, axis=1))
            feature_maxs.extend(np.max(imgs_flat, axis=1))

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate and Print Final Metric
    final_auc = roc_auc_score(all_targets, all_probs)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Correlation between error magnitude and input features
    errors = np.abs(all_targets - all_probs)

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": feature_means,
            "std_intensity": feature_stds,
            "max_intensity": feature_maxs,
        }
    )

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    generate_submission(model_path=model_path)


if __name__ == "__main__":
    main()
