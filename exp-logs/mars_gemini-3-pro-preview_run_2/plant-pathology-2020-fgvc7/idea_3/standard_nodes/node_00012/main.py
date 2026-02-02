import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.utils import seed_everything, get_worker_init_fn, calculate_roc_auc
from library.augmentations import get_train_transforms, get_valid_transforms, CutMix
from library.dataset import AppleDataset, TARGET_COLS
from library.model import AppleDiseaseModel
from library.engine import train_model, generate_submission, validate
from library.loss import WeightedSoftCrossEntropy, get_class_weights

# --- Constants ---
SEED = 42
IMAGE_SIZE = 480
BATCH_SIZE = 16  # Adjusted for EfficientNetV2-M and 480x480 on A100
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBMISSION_THRESHOLD = 0.9902480620249655
MODEL_SAVE_PATH = "./working/best_model.pth"
SUBMISSION_DIR = "./submission"


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes the correlation between model error and input image features.
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()

    all_preds = []
    all_labels = []
    image_ids = val_loader.dataset.get_image_ids()

    # 1. Get Predictions and Labels
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 2. Calculate Error Magnitude (MAE per sample averaged across classes)
    # Shape: (N_samples,)
    errors = np.mean(np.abs(all_labels - all_preds), axis=1)

    # 3. Extract Image Features
    # We need to access the dataframe to get file paths
    val_df = val_loader.dataset.df
    features = []

    print("Extracting metadata features for correlation analysis...")
    for idx, row in val_df.iterrows():
        # Reconstruct full path as done in Dataset class
        full_path = os.path.join(val_loader.dataset.data_dir, row["file_path"])

        if not os.path.exists(full_path):
            continue

        # File Size
        f_size = os.path.getsize(full_path)

        # Image Stats (Read with OpenCV)
        img = cv2.imread(full_path)
        if img is None:
            continue

        h, w, c = img.shape
        mean_intensity = img.mean()

        features.append(
            {
                "image_id": row["image_id"],
                "file_size": f_size,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
                "mean_intensity": mean_intensity,
                "error": errors[idx],
            }
        )

    feat_df = pd.DataFrame(features)

    # 4. Calculate Correlations
    if not feat_df.empty:
        print("\nCorrelation between Error Magnitude and Input Features:")
        feature_cols = [
            "file_size",
            "width",
            "height",
            "aspect_ratio",
            "mean_intensity",
        ]

        for col in feature_cols:
            if col in feat_df.columns:
                # Drop NaNs if any
                valid_data = feat_df[[col, "error"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[col], valid_data["error"])
                    print(f"  {col}: {corr:.4f}")
                else:
                    print(f"  {col}: Not enough data")
    else:
        print("Could not extract features for failure analysis.")
    print("-" * 30)


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = AppleDataset(
        mode="train", transform=get_train_transforms(IMAGE_SIZE), load_cached_data=True
    )
    val_dataset = AppleDataset(
        mode="val", transform=get_valid_transforms(IMAGE_SIZE), load_cached_data=True
    )
    test_dataset = AppleDataset(
        mode="test", transform=get_valid_transforms(IMAGE_SIZE), load_cached_data=True
    )

    # Worker Init Function for reproducibility
    worker_init = get_worker_init_fn(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing EfficientNetV2-M Model...")
    model = AppleDiseaseModel(model_name="tf_efficientnetv2_m.in1k", num_classes=4)
    model.to(DEVICE)

    # 4. Training
    # CutMix Regularization
    cutmix = CutMix(alpha=1.0)

    print("Starting Training...")
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
        patience=3,  # Early stopping patience
        cutmix_fn=cutmix,
        save_path=MODEL_SAVE_PATH,
    )

    # 5. Final Validation & Metrics
    print("\nLoading best model for final validation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # We need the class weights for the loss calculation in validate,
    # though the metric (AUC) is independent of it.
    class_weights = get_class_weights(load_cached_data=True).to(DEVICE)
    criterion = WeightedSoftCrossEntropy(weights=class_weights)

    val_loss, final_metric = validate(model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, DEVICE)

    # 7. Submission
    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Metric ({final_metric}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, DEVICE, output_dir=SUBMISSION_DIR)
    else:
        print(
            f"Metric ({final_metric}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
