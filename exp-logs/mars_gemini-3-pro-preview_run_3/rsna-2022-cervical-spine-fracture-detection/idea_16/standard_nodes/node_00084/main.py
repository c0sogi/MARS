import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config, seed_everything
from library.data import CervicalDataset, get_transforms
from library.model import FractureMILModel
from library.train import run_training
from library.utils import calculate_weighted_loss_metric


def get_dir_stats(study_uid, base_dir):
    """
    Calculates number of slices and average file size for a given study.
    """
    path = os.path.join(base_dir, study_uid)
    if not os.path.exists(path):
        return 0, 0

    files = glob.glob(os.path.join(path, "*"))
    num_slices = len(files)

    if num_slices > 0:
        sizes = [os.path.getsize(f) for f in files]
        avg_size = np.mean(sizes) / (1024 * 1024)  # MB
    else:
        avg_size = 0

    return num_slices, avg_size


def calculate_per_sample_loss(y_true, y_pred):
    """
    Calculates the weighted loss for each sample individually.
    Returns a numpy array of losses.
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Binary Cross Entropy per element: - (y * log(p) + (1-y) * log(1-p))
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Weighted sum across classes for each sample
    weighted_bce = np.sum(bce * weights, axis=1)

    # Normalize by sum of weights
    return weighted_bce / np.sum(weights)


def predict_fn(model, loader, device):
    """
    Runs inference on a loader and returns probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, dtype=torch.float32)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for fast baseline execution
    # Increased to 10 epochs to allow convergence with larger batch size
    Config.EPOCHS = 10

    print("=== Starting Training Phase ===")
    # 2. Train Model
    # This will save the model to Config.MODEL_SAVE_PATH
    run_training(debug=False)

    print("\n=== Starting Validation & Analysis Phase ===")

    # 3. Load Best Model
    model = FractureMILModel(config=Config)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Validation Inference
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_ds = CervicalDataset(
        val_df, transforms=get_transforms("val"), load_cached_data=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_probs = predict_fn(model, val_loader, device)

    # Get targets
    target_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
    val_targets = val_df[target_cols].values

    # 5. Calculate Metric
    metric = calculate_weighted_loss_metric(val_targets, val_probs)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    # Calculate loss per sample
    sample_losses = calculate_per_sample_loss(val_targets, val_probs)

    # Extract meta-features
    meta_features = []
    for uid in val_df["StudyInstanceUID"]:
        # Assuming images are in train_images based on metadata generation script
        n_slices, avg_size = get_dir_stats(uid, Config.TRAIN_IMAGES_DIR)
        meta_features.append({"num_slices": n_slices, "avg_file_size_mb": avg_size})

    meta_df = pd.DataFrame(meta_features)
    meta_df["loss"] = sample_losses

    # Correlations
    corr_slices = meta_df["num_slices"].corr(meta_df["loss"])
    corr_size = meta_df["avg_file_size_mb"].corr(meta_df["loss"])

    print("\n=== Failure Analysis ===")
    print(f"Correlation (Loss vs Num Slices): {corr_slices:.4f}")
    print(f"Correlation (Loss vs Avg File Size): {corr_size:.4f}")

    # 7. Submission
    THRESHOLD = 0.95

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        test_ds = CervicalDataset(
            test_df, transforms=get_transforms("test"), load_cached_data=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_probs = predict_fn(model, test_loader, device)

        # Map predictions to submission format
        # test_probs shape: (N_studies, 8)
        # Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall

        # Create a lookup dictionary: study_uid -> dict of probs
        pred_map = {}
        cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

        for idx, row in test_df.iterrows():
            uid = row["StudyInstanceUID"]
            probs = test_probs[idx]
            pred_map[uid] = {c: p for c, p in zip(cols, probs)}

        # Read sample submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Fill values
        # row_id format: StudyInstanceUID_Target
        new_fractured = []
        for row_id in sample_sub["row_id"]:
            # Split last part to get target, rest is UID
            # Target is always C1..C7 or patient_overall
            # But UID can contain underscores? Usually DICOM UIDs are dots.
            # Let's split by underscore.
            parts = row_id.rsplit("_", 1)

            if len(parts) == 2 and parts[1] in [
                "overall",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
            ]:
                # Handle 'patient_overall' case which splits into ['UID_patient', 'overall'] if we split by '_'
                # Actually, target is 'patient_overall'.
                # If we split by '_', '1.2.3_patient_overall' -> '1.2.3_patient', 'overall'. Incorrect.
                pass

            # Robust splitting
            target = None
            uid = None

            possible_targets = [
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "patient_overall",
            ]
            for t in possible_targets:
                if row_id.endswith(f"_{t}"):
                    target = t
                    uid = row_id[: -len(t) - 1]  # remove _Target
                    break

            if uid in pred_map and target in pred_map[uid]:
                new_fractured.append(pred_map[uid][target])
            else:
                # Fallback (should not happen if metadata is correct)
                new_fractured.append(0.5)

        sample_sub["fractured"] = new_fractured

        # Save
        sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({metric}) is NOT lower than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
