import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config, seed_everything
from library.train import run_training
from library.models import PawpularityModel
from library.data import (
    PawpularityDataset,
    get_transforms,
    DENSE_FEATURES,
    DENSE_FEATURES_ALT,
)
from library.utils import get_score


def main():
    # 1. Setup and Configuration Override
    seed_everything(Config.seed)

    # Override Config for a fast baseline execution within 2 hours
    # 5 folds * 2 models * 3 epochs approx 90 mins on A100
    Config.epochs = 3
    Config.num_folds = 5

    # Ensure working directory exists (handled in run_training but good to be safe)
    Config.create_dirs()

    print(
        f"Configuration: Epochs={Config.epochs}, Folds={Config.num_folds}, Models={Config.model_names}"
    )

    # 2. Run Training
    # This will train both Swin and ConvNeXt models across 5 folds
    print("\nStarting Training Pipeline...")
    run_training()
    print("Training Pipeline Completed.")

    # 3. Validation and Failure Analysis
    print("\nStarting Validation and Failure Analysis...")
    device = Config.device

    # Load Validation Data
    # We use the fixed validation set from metadata
    if not os.path.exists(Config.val_csv_path):
        raise FileNotFoundError(f"Validation CSV not found at {Config.val_csv_path}")

    val_df = pd.read_csv(Config.val_csv_path)

    # Create Validation Dataset and Loader
    # We use 'valid' transforms (resize + normalize only)
    val_ds = PawpularityDataset(
        val_df,
        Config.input_dir,
        transforms=get_transforms("valid", Config.image_size),
        is_test=False,  # This ensures we get targets for metric calculation
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load all trained models for the ensemble
    models = []
    for model_name in Config.model_names:
        for fold in range(Config.num_folds):
            checkpoint_path = os.path.join(
                Config.working_dir, f"{model_name}_fold_{fold}.pth"
            )
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping.")
                continue

            # Initialize model and load weights
            model = PawpularityModel(model_name=model_name, pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)

    if not models:
        raise RuntimeError("No models were loaded. Training might have failed.")

    print(f"Loaded {len(models)} models for ensemble inference.")

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    meta_features_list = []

    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            # Ensemble Prediction
            batch_preds = []
            for model in models:
                logits = model(images, metadata)
                probs = torch.sigmoid(logits)  # Convert logits to [0, 1]
                batch_preds.append(probs.cpu().numpy())

            # Average predictions across all models
            avg_preds = np.mean(batch_preds, axis=0).flatten()

            all_preds.extend(avg_preds)
            all_targets.extend(targets.numpy())
            meta_features_list.extend(metadata.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    meta_features_arr = np.array(meta_features_list)

    # Calculate Final Metric
    # get_score handles the scaling. all_targets are [0, 1], so it scales them to [1, 100]
    val_rmse = get_score(all_targets, all_preds)

    print(f"Final Validation Metric: {val_rmse}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute errors on the original [1, 100] scale
    y_true_orig = all_targets * 100.0
    y_pred_orig = all_preds * 100.0
    errors = np.abs(y_true_orig - y_pred_orig)

    # Determine feature names based on dataframe columns
    if "Subject Focus" in val_df.columns:
        feature_names = DENSE_FEATURES
    else:
        feature_names = DENSE_FEATURES_ALT

    print("Correlation between Absolute Error and Metadata Features:")
    for i, name in enumerate(feature_names):
        if i < meta_features_arr.shape[1]:
            feat_vals = meta_features_arr[:, i]
            # Handle constant features to avoid warnings
            if np.std(feat_vals) == 0 or np.std(errors) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(errors, feat_vals)
            print(f"  {name}: {corr}")

    # 4. Submission Generation
    # Condition: Metric < 17.184850648081728
    THRESHOLD = 17.184850648081728

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({val_rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_df = pd.read_csv(Config.test_csv_path)

        test_ds = PawpularityDataset(
            test_df,
            Config.input_dir,
            transforms=get_transforms("valid", Config.image_size),
            is_test=True,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds = []

        with torch.no_grad():
            for images, metadata, _ in test_loader:
                images = images.to(device)
                metadata = metadata.to(device)

                batch_preds = []
                for model in models:
                    logits = model(images, metadata)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

                avg_preds = np.mean(batch_preds, axis=0).flatten()
                test_preds.extend(avg_preds)

        # Rescale predictions to [1, 100]
        final_test_preds = np.array(test_preds) * 100.0

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"Id": test_df["Id"], "Pawpularity": final_test_preds}
        )

        # Save Submission
        os.makedirs(Config.submission_dir, exist_ok=True)
        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

        # Print first few rows for verification
        print(submission.head())

    else:
        print(
            f"\nValidation metric ({val_rmse}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
