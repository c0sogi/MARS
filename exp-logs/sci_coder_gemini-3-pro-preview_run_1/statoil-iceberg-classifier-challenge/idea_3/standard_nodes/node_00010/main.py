import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library modules
import library.config as config
import library.utils as utils_lib
import library.dataset as dataset_lib
import library.model as model_lib
import library.train as train_lib
import library.predict as predict_lib


def main():
    # 1. Execute Training Pipeline (Stratified 5-Fold CV)
    print("Starting Training Pipeline...")
    train_lib.run_training()

    # 2. Generate Out-Of-Fold (OOF) Predictions for Validation
    # We need to generate predictions for the hold-out validation set using models
    # that were not trained on those specific samples.
    print("Generating OOF predictions...")
    utils_lib.seed_everything(config.SEED)
    device = utils_lib.get_device()

    # Load all labeled data (Train + Val metadata combined)
    images, angles, labels = dataset_lib.load_data(mode="train", load_cached_data=True)

    # Reconstruct metadata to identify the hold-out validation indices
    # load_data concatenates TRAIN_META then VAL_META
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    df_val = pd.read_csv(config.VAL_META_PATH)
    df_meta = pd.concat([df_train, df_val], ignore_index=True)

    # Initialize array to store OOF predictions
    oof_preds = np.zeros(len(images), dtype=np.float32)

    # Re-create the Stratified K-Fold split to match the training phase
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        # Get validation subset for this fold
        X_val = images[val_idx]
        a_val = angles[val_idx]

        # Create Dataset and Loader
        # We pass labels=None so the loader yields (image, angle) compatible with predict_with_tta
        val_dataset = dataset_lib.IcebergDataset(
            X_val,
            a_val,
            labels=None,
            transform=dataset_lib.get_transforms(mode="valid"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load the model trained on the training subset of this fold
        model = model_lib.IcebergResNet()
        model.to(device)
        model_path = config.get_model_path(fold_idx)

        # Load weights
        utils_lib.load_checkpoint(model_path, model)

        # Predict using TTA to match submission quality
        preds = predict_lib.predict_with_tta(model, val_loader, device)

        # Store predictions
        oof_preds[val_idx] = preds.flatten()

    # 3. Evaluate on Hold-out Validation Set
    # The hold-out set corresponds to the rows from df_val, which are appended at the end
    val_start_idx = len(df_train)
    val_indices = np.arange(val_start_idx, len(df_meta))

    val_preds = oof_preds[val_indices]
    val_targets = labels[val_indices]

    # Calculate Log Loss
    final_metric = log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 4. Failure Analysis
    print("Performing Failure Analysis on Validation Set...")

    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Extract features for correlation analysis
    # 1. Incidence Angle (Normalized)
    val_angles = angles[val_indices]

    # 2. Image Mean Intensity (Proxy for brightness/clutter)
    # images shape is (N, 224, 224, 3)
    val_images = images[val_indices]
    val_img_means = np.mean(val_images, axis=(1, 2, 3))

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"error": errors, "inc_angle": val_angles, "img_mean": val_img_means}
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # 5. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.21099163245555455

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric:.6f} is better than threshold {THRESHOLD:.6f}."
        )
        print("Generating submission file...")
        predict_lib.generate_submission()
    else:
        print(
            f"Validation metric {final_metric:.6f} did not meet threshold {THRESHOLD:.6f}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
