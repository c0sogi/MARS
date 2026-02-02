import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.trainer import run_fold
from library.model_factory import get_model
from library.data_loader import get_loaders, get_test_loader


def main():
    # 1. Configuration & Setup
    # Override epochs to 10 for a fast baseline execution
    config = Config(epochs=10)
    seed_everything(config.seed)

    print("==== Starting Pipeline ====")
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")

    # 2. Training & OOF Collection
    model_paths = []
    oof_preds = []
    oof_targets = []
    oof_file_paths = []

    for fold in range(config.n_folds):
        print(f"\n[Fold {fold}/{config.n_folds - 1}]")

        # Train the fold
        run_fold(fold, config)

        # Define path to the saved best model
        model_path = os.path.join(
            config.working_dir, f"{config.model_name}_fold_{fold}.pth"
        )
        model_paths.append(model_path)

        # --- OOF Inference ---
        # We reload the best model to ensure we are evaluating the checkpoint that maximized validation AUC
        print(f"Generating OOF predictions for Fold {fold}...")

        model = get_model(config)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.to(config.device)
        model.eval()

        # Get validation loader for this fold
        _, val_loader = get_loaders(fold, config)

        # Store metadata for failure analysis
        val_dataset = val_loader.dataset
        oof_file_paths.extend(val_dataset.df["file_path"].tolist())

        fold_probs = []
        fold_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.device)

                # Standard inference
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.append(probs.cpu().numpy())
                fold_labels.append(labels.numpy())

        oof_preds.append(np.concatenate(fold_probs, axis=0))
        oof_targets.append(np.concatenate(fold_labels, axis=0))

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # 3. Global Validation Metric
    oof_preds = np.concatenate(oof_preds, axis=0)
    oof_targets = np.concatenate(oof_targets, axis=0)

    try:
        final_val_metric = roc_auc_score(
            oof_targets, oof_preds, average="macro", multi_class="ovr"
        )
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC ({e}). Defaulting to 0.")
        final_val_metric = 0.0

    print(f"Final Validation Metric: {final_val_metric}")

    # 4. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate Mean Squared Error per sample
    # MSE = mean((y_true - y_pred)^2)
    sample_errors = np.mean((oof_targets - oof_preds) ** 2, axis=1)

    # Extract image stats
    widths = []
    heights = []
    intensities = []

    print("Extracting image features for correlation analysis...")
    for rel_path in oof_file_paths:
        full_path = os.path.join(config.input_dir, rel_path)
        # Default values
        w, h, i = 0, 0, 0.0

        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                # Calculate mean intensity (normalized)
                i = img.mean() / 255.0

        widths.append(w)
        heights.append(h)
        intensities.append(i)

    # Calculate correlations
    if len(sample_errors) == len(widths):
        # Using numpy corrcoef to avoid scipy dependency issues
        # corrcoef returns matrix [[1, r], [r, 1]]
        c_w = np.corrcoef(sample_errors, widths)[0, 1]
        c_h = np.corrcoef(sample_errors, heights)[0, 1]
        c_i = np.corrcoef(sample_errors, intensities)[0, 1]

        print(f"Correlation between Error and Width: {np.nan_to_num(c_w):.6f}")
        print(f"Correlation between Error and Height: {np.nan_to_num(c_h):.6f}")
        print(f"Correlation between Error and Intensity: {np.nan_to_num(c_i):.6f}")
    else:
        print("Mismatch in data lengths, skipping correlation.")

    # 5. Submission
    THRESHOLD = 0.9871488489626378

    if final_val_metric > THRESHOLD:
        print(
            f"\nMetric ({final_val_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader, test_df = get_test_loader(config)
        ensemble_preds = np.zeros((len(test_df), config.num_classes))

        for i, model_path in enumerate(model_paths):
            print(f"Inference Model {i+1}/{config.n_folds}")
            model = get_model(config)
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model.to(config.device)
            model.eval()

            fold_preds = []

            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(config.device)

                    # TTA: Original
                    out1 = model(images)
                    prob1 = torch.softmax(out1, dim=1)

                    # TTA: Horizontal Flip
                    images_flip = torch.flip(images, [3])  # Flip width dimension
                    out2 = model(images_flip)
                    prob2 = torch.softmax(out2, dim=1)

                    # Average probabilities
                    avg_prob = (prob1 + prob2) / 2.0
                    fold_preds.append(avg_prob.cpu().numpy())

            ensemble_preds += np.concatenate(fold_preds, axis=0)

            del model
            torch.cuda.empty_cache()

        # Average over folds
        ensemble_preds /= config.n_folds

        # Save submission
        submission = pd.DataFrame(ensemble_preds, columns=config.target_cols)
        submission.insert(0, "image_id", test_df["image_id"])
        submission.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")

    else:
        print(
            f"\nMetric ({final_val_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
