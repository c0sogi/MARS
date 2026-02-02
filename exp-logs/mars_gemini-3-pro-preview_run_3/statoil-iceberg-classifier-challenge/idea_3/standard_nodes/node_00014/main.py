import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.model import CustomDenseNet
from library.data_loader import load_data, IcebergDataset, get_test_loader
from library.train import run_training_fold


def main():
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)

    # Adjust configuration for fast baseline execution
    Config.NUM_EPOCHS = 25

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Models (5-Fold CV)
    print("Starting 5-Fold Cross-Validation Training...")
    for fold_idx in range(Config.NUM_FOLDS):
        run_training_fold(fold_idx)

    # 3. Load Hold-out Validation Set
    # We use the metadata file to identify exactly which samples constitute the hold-out set
    print("Loading hold-out validation set from metadata...")
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    val_indices = val_meta["original_index"].values

    # Load the full training data arrays (cached)
    X_all, angles_all, y_all = load_data(mode="train", load_cached_data=True)

    # Extract the validation subset
    X_val = X_all[val_indices]
    angles_val = angles_all[val_indices]
    y_val = y_all[val_indices]

    # Create DataLoader for validation
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch size for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Ensemble Inference on Validation Set
    print("Evaluating ensemble on validation set...")
    models = []
    # Load the best model from each fold
    for fold_idx in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORK_DIR, f"fold_{fold_idx}", "model_best.pth")
        model = CustomDenseNet(drop_rate=Config.DROP_RATE, fc_dim=Config.FC_DIM)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    val_preds = []
    val_targets = []

    # Lists for failure analysis
    fa_angles = []
    fa_img_means = []

    with torch.no_grad():
        for images, angles, targets in val_loader:
            images = images.to(device)
            angles_gpu = angles.to(device)

            # Collect metadata for failure analysis
            fa_angles.extend(angles.cpu().numpy())
            # Calculate mean intensity (feature proxy)
            batch_means = images.view(images.size(0), -1).mean(dim=1).cpu().numpy()
            fa_img_means.extend(batch_means)

            # Ensemble Prediction
            fold_preds = []
            for model in models:
                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

            # Average predictions across folds
            avg_preds = np.mean(fold_preds, axis=0)
            val_preds.extend(avg_preds)
            val_targets.extend(targets.numpy())

    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()

    # 5. Calculate Metric
    # Clip predictions to avoid log(0) errors
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_targets, val_preds_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(val_targets - val_preds)

    # Correlation with Incidence Angle
    corr_angle = np.corrcoef(errors, np.array(fa_angles))[0, 1]
    print(f"Correlation between Error and Incidence Angle: {corr_angle:.6f}")

    # Correlation with Image Intensity
    corr_intensity = np.corrcoef(errors, np.array(fa_img_means))[0, 1]
    print(f"Correlation between Error and Image Mean Intensity: {corr_intensity:.6f}")

    # 7. Generate Submission
    THRESHOLD = 0.2089132981339209

    if final_metric < THRESHOLD:
        print(
            f"Validation metric meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader, test_ids = get_test_loader(load_cached_data=True)
        test_preds = []

        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                fold_preds = []
                for model in models:
                    logits = model(images, angles_gpu)
                    probs = torch.sigmoid(logits)
                    fold_preds.append(probs.cpu().numpy())

                avg_preds = np.mean(fold_preds, axis=0)
                test_preds.extend(avg_preds)

        test_preds = np.array(test_preds).flatten()

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"Validation metric {final_metric} >= {THRESHOLD}. Submission generation skipped."
        )


if __name__ == "__main__":
    main()
