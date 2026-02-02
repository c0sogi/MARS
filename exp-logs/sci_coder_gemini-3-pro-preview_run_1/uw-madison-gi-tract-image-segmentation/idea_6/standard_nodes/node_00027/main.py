import os
import sys
import random
import numpy as np
import pandas as pd
import torch

# Import library components
from library.config import Config
from library.data import create_dataloaders
from library.trainer import Trainer
from library.utils import compute_metrics
from library.postprocessing import keep_largest_component_3d


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Configuration and Setup
    Config.setup()
    set_seed(Config.SEED)

    # We use the defaults from Config, ensuring we run on the full provided dataset.
    # The dataset size (approx 60 training volumes) is small enough to run
    # within the time limit using the efficient 3D ResNet backbone.
    Config.DEBUG = False

    # 2. Data Loading
    # load_cached_data=True allows skipping preprocessing if already done in previous runs
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(load_cached_data=True)

    # 3. Model Training
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    print("Starting Training...")
    trainer.fit()

    # 4. Validation and Failure Analysis
    print("Starting Failure Analysis on Validation Set...")

    # Load the best checkpoint found during training
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    trainer.model.eval()

    val_scores = []
    val_features = []

    # Disable gradient calculation for inference to save memory and speed up
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch
            # batch['image']: (1, 1, D, H, W) -> Single volume per batch for validation
            # batch['mask']: (1, C, D, H, W)
            image = batch["image"][0]  # Remove batch dim -> (1, D, H, W)
            mask = batch["mask"][0]  # Remove batch dim -> (C, D, H, W)

            # Perform 2.5D slice inference
            # Returns probabilities (C, D, H, W)
            pred_probs = trainer._predict_volume(image)

            # Threshold to binary
            pred_mask = (pred_probs > 0.5).float().cpu().numpy()
            gt_mask = mask.numpy()

            # Calculate metrics per class
            case_metrics = []
            for c in range(Config.NUM_CLASSES):
                # Apply post-processing (keep largest component)
                p_c = keep_largest_component_3d(pred_mask[c])
                g_c = gt_mask[c]

                m = compute_metrics(p_c, g_c)
                case_metrics.append(m["score"])

            # Average score for the case
            avg_score = np.mean(case_metrics)
            val_scores.append(avg_score)

            # Extract features for failure analysis
            # Feature 1: Volume Depth (Z-axis) - indicates scan coverage
            depth = image.shape[1]
            # Feature 2: Mean Intensity - indicates brightness/contrast
            mean_intensity = image.mean().item()
            # Feature 3: Foreground Ratio - indicates complexity/amount of organ present
            fg_ratio = gt_mask.mean().item()

            val_features.append(
                {
                    "depth": depth,
                    "mean_intensity": mean_intensity,
                    "fg_ratio": fg_ratio,
                    "score": avg_score,
                    "error": 1.0 - avg_score,  # Error magnitude
                }
            )

    # Compute Final Metric
    final_metric = np.mean(val_scores)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # Correlation Analysis
    print("--- Failure Analysis: Correlation between Error and Features ---")
    df_analysis = pd.DataFrame(val_features)

    for feature in ["depth", "mean_intensity", "fg_ratio"]:
        if df_analysis[feature].std() > 0:
            # Compute Pearson correlation using numpy
            corr = np.corrcoef(df_analysis["error"], df_analysis[feature])[0, 1]
            print(f"Correlation (Error vs {feature}): {corr:.4f}")
        else:
            print(f"Correlation (Error vs {feature}): N/A (Constant feature)")

    # 5. Submission
    # Threshold defined in task requirements
    SUBMISSION_THRESHOLD = 0.5184837797359911

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric ({final_metric}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        trainer.predict_and_submit()
    else:
        print(
            f"Validation metric ({final_metric}) does not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
