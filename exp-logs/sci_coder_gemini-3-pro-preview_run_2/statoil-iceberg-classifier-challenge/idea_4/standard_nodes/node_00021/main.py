import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.train_eval import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Running on device: {Config.DEVICE}")

    # 2. Data Loading
    # We use the full dataset as it is small (approx 1600 samples).
    # Limiting samples further would prevent reaching the target metric.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Training
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # 4. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model weights for accurate evaluation
    if os.path.exists(Config.MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    trainer.model.eval()

    val_preds = []
    val_targets = []
    val_inc_angles = []
    val_img_means = []

    # Inference loop without gradient calculation
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE)
            inc_angles = batch["inc_angle"].to(Config.DEVICE)
            labels = batch["label"].cpu().numpy()

            # Forward pass
            outputs = trainer.model(images, inc_angles)

            # Collect data
            # Outputs are already sigmoid probabilities
            preds = outputs.cpu().numpy().flatten()
            val_preds.extend(preds)
            val_targets.extend(labels)

            # Collect features for failure analysis
            # Flatten inc_angles to 1D array
            val_inc_angles.extend(inc_angles.cpu().numpy().flatten())

            # Calculate mean intensity of Band 1 (channel 0) as a simple image feature
            # images shape: (B, 3, 75, 75)
            img_means = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
            val_img_means.extend(img_means)

    # Convert to numpy arrays
    y_true = np.array(val_targets)
    y_pred = np.array(val_preds)

    # Calculate Metric (Log Loss)
    # Clip predictions to prevent log(0) errors, although sigmoid output is (0, 1)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_true, y_pred_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Create a DataFrame to compute correlations
    analysis_df = pd.DataFrame(
        {"error": errors, "inc_angle": val_inc_angles, "band_1_mean": val_img_means}
    )

    # Compute correlation of features with the error
    correlations = analysis_df.corr()["error"].drop("error")

    print("\nFailure Analysis - Correlation with Prediction Error:")
    print(correlations)

    # 5. Submission Logic
    target_threshold = 0.20320119103524176

    if final_metric < target_threshold:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({target_threshold})."
        )
        print("Generating submission file...")
        trainer.predict(test_loader)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({target_threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
