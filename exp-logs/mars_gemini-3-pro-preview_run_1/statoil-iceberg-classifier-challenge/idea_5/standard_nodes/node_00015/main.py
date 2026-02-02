import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided library files
from library.utils import seed_everything, get_device
from library.data import process_split, IcebergDataset, get_transforms
from library.train import run_fold_training
from library.predict import generate_ensemble_submission

# Constants
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5/"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"
N_FOLDS = 5
EPOCHS = 6  # Fast baseline
BATCH_SIZE = 32
THRESHOLD = 0.21099163245555455


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    # Load original train split
    t_imgs, t_angs, t_lbls, t_ids = process_split(
        os.path.join(METADATA_DIR, "train_metadata.csv"), "train", load_cached_data=True
    )
    # Load original val split
    v_imgs, v_angs, v_lbls, v_ids = process_split(
        os.path.join(METADATA_DIR, "val_metadata.csv"), "val", load_cached_data=True
    )

    # Merge for K-Fold
    X_images = np.concatenate([t_imgs, v_imgs], axis=0)
    X_angles = np.concatenate([t_angs, v_angs], axis=0)
    y_labels = np.concatenate([t_lbls, v_lbls], axis=0)
    X_ids = np.concatenate([t_ids, v_ids], axis=0)

    # 3. Stratified K-Fold Training
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y_labels))
    oof_targets = np.zeros(len(y_labels))
    oof_indices = np.zeros(len(y_labels), dtype=int)

    # To store feature data for failure analysis
    feature_data = {
        "inc_angle": np.zeros(len(y_labels)),
        "img_mean": np.zeros(len(y_labels)),
        "img_std": np.zeros(len(y_labels)),
    }

    print(f"Starting {N_FOLDS}-Fold Cross-Validation on {len(X_images)} samples...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_images, y_labels)):
        # Create Datasets
        train_ds = IcebergDataset(
            X_images[train_idx],
            X_angles[train_idx],
            y_labels[train_idx],
            X_ids[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_images[val_idx],
            X_angles[val_idx],
            y_labels[val_idx],
            X_ids[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )

        # Train Model
        # run_fold_training saves the best model to WORKING_DIR
        model = run_fold_training(
            fold_idx, train_loader, val_loader, EPOCHS, device, WORKING_DIR
        )

        # Generate OOF Predictions for this fold
        model.eval()
        fold_probs = []
        fold_targets = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Standard inference (no TTA for validation metric to be strict)
                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                fold_probs.extend(probs)
                fold_targets.extend(labels.numpy())

        # Store OOF data
        oof_preds[val_idx] = fold_probs
        oof_targets[val_idx] = fold_targets
        oof_indices[val_idx] = val_idx

        # Store features for analysis (using raw numpy arrays)
        feature_data["inc_angle"][val_idx] = X_angles[val_idx]
        # Calculate simple image stats on the fly (using channel 0 for simplicity or mean of channels)
        # Images are (N, 224, 224, 3)
        imgs_val_np = X_images[val_idx]
        feature_data["img_mean"][val_idx] = np.mean(imgs_val_np, axis=(1, 2, 3))
        feature_data["img_std"][val_idx] = np.std(imgs_val_np, axis=(1, 2, 3))

        # Cleanup
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 4. Global Validation Assessment
    # Clip predictions to avoid log(0)
    clipped_preds = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_log_loss = log_loss(y_labels, clipped_preds)

    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_labels - oof_preds)

    # Create DataFrame for correlation calculation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": feature_data["inc_angle"],
            "img_mean": feature_data["img_mean"],
            "img_std": feature_data["img_std"],
        }
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission Generation
    if final_log_loss < THRESHOLD:
        print(
            f"\nValidation metric ({final_log_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_ensemble_submission(
            n_folds=N_FOLDS,
            batch_size=BATCH_SIZE,
            model_dir=WORKING_DIR,
            load_cached_data=True,
        )
    else:
        print(
            f"\nValidation metric ({final_log_loss}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
