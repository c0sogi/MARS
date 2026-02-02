import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_fold_loaders
from library.model import SQWBN
from library.train_eval import train_fold, predict_ensemble


def main():
    # 1. Initialization
    print("Initializing SQ-WBN Pipeline...")
    seed_everything(Config.SEED)
    Config.initialize()

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 2. 5-Fold Cross-Validation & OOF Collection
    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    # Containers for global evaluation
    oof_preds = []
    oof_targets = []
    oof_inc_angles = []
    oof_img_means = []
    oof_img_stds = []

    device = torch.device(Config.DEVICE)

    for fold_idx in range(Config.NUM_FOLDS):
        # A. Train the fold
        # train_fold handles training, early stopping, and saving the best model
        print(f"\nProcessing Fold {fold_idx}/{Config.NUM_FOLDS - 1}")
        train_fold(fold_idx)

        # B. Generate OOF Predictions for Analysis
        # We reload the best saved model to ensure we evaluate the exact state used for inference
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        model = SQWBN().to(device)
        model_path = os.path.join(Config.ARTIFACT_DIR, f"model_fold_{fold_idx}.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Get validation loader (no shuffle, no augmentation)
        _, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        fold_probs = []
        fold_targets = []

        with torch.no_grad():
            for images, inc_angles, labels in val_loader:
                # Move inputs to device
                images_dev = images.to(device)
                inc_angles_dev = inc_angles.to(device)

                # Inference
                outputs = model(images_dev, inc_angles_dev)
                probs = outputs.cpu().numpy().flatten()

                # Store results
                fold_probs.extend(probs)
                fold_targets.extend(labels.numpy())

                # Store features for failure analysis
                oof_inc_angles.extend(inc_angles.numpy())

                # Calculate image stats on CPU (Mean and Std of the 3-channel image)
                imgs_np = images.numpy()  # (B, 3, 75, 75)
                # Average intensity per image
                oof_img_means.extend(np.mean(imgs_np, axis=(1, 2, 3)))
                # Contrast/Texture per image
                oof_img_stds.extend(np.std(imgs_np, axis=(1, 2, 3)))

        oof_preds.extend(fold_probs)
        oof_targets.extend(fold_targets)

    # 3. Global Evaluation
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Clip predictions to prevent log(0) errors
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)

    final_metric = log_loss(oof_targets, oof_preds_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(oof_targets - oof_preds)

    # Compute correlations
    # We use numpy corrcoef which returns a matrix [[1, r], [r, 1]]

    # 1. Error vs Incidence Angle
    corr_inc = np.corrcoef(errors, oof_inc_angles)[0, 1]
    print(f"Correlation (Error vs Inc Angle): {corr_inc:.4f}")

    # 2. Error vs Image Brightness (Mean)
    corr_mean = np.corrcoef(errors, oof_img_means)[0, 1]
    print(f"Correlation (Error vs Image Mean): {corr_mean:.4f}")

    # 3. Error vs Image Contrast (Std)
    corr_std = np.corrcoef(errors, oof_img_stds)[0, 1]
    print(f"Correlation (Error vs Image Std): {corr_std:.4f}")

    # 5. Submission Generation
    THRESHOLD = 0.16676861786296204

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.6f}) meets threshold ({THRESHOLD:.6f})."
        )
        print("Generating test set predictions...")

        # Run ensemble prediction (saves to Config.SUBMISSION_PATH)
        predict_ensemble()

        # Move to required location
        src_path = Config.SUBMISSION_PATH
        dst_path = "./submission/submission.csv"

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            df.to_csv(dst_path, index=False)
            print(f"Submission successfully saved to {dst_path}")
        else:
            print(f"Error: Generated submission not found at {src_path}")
    else:
        print(
            f"\nValidation metric ({final_metric:.6f}) did not meet threshold ({THRESHOLD:.6f})."
        )
        print("Submission skipped.")


if __name__ == "__main__":
    main()
