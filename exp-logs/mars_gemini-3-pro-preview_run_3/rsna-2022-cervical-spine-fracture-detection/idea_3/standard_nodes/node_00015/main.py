import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from library
from library.config import Config
from library.utils import seed_everything, get_weighted_log_loss
from library.dataset import CervicalSpineDataset
from library.model import AnatomicallyGuidedResNet
from library.engine import train_one_epoch, validate, HierarchicalCompoundLoss


def analyze_failures(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Correlates error magnitude with input features (number of slices).
    """
    model.eval()

    # 1. Collect Predictions and Targets
    all_preds = []
    all_targets = []
    all_study_ids = []

    with torch.no_grad():
        for images, positions, targets, study_ids in val_loader:
            images = images.to(device)
            positions = positions.to(device)

            logits = model(images, positions)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_study_ids.extend(study_ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Error Magnitude per Study
    # We use the weighted log loss contribution per study as the error metric
    # Weights: patient_overall=7, others=1. Total weight = 14.
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 7])  # C1..C7, Patient

    # BCE Loss per label: -(y*log(p) + (1-y)*log(1-p))
    epsilon = 1e-15
    pred_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    bce_loss = -(
        all_targets * np.log(pred_clipped)
        + (1 - all_targets) * np.log(1 - pred_clipped)
    )

    # Weighted sum across columns for each row (study)
    study_errors = np.sum(bce_loss * weights, axis=1) / np.sum(weights)

    # 3. Collect Input Features (Number of Slices)
    # We need to check the file system or cache to get the number of slices for each study
    slice_counts = []
    for study_id in all_study_ids:
        # Find the row in metadata
        row = val_df[val_df["StudyInstanceUID"] == study_id].iloc[0]
        full_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Count files
        if os.path.exists(full_path):
            # Fast count using scandir or glob
            # glob is safer given the structure
            cnt = len(glob.glob(os.path.join(full_path, "*.dcm")))
            slice_counts.append(cnt)
        else:
            slice_counts.append(0)

    # 4. Correlation
    if len(slice_counts) > 1 and np.std(slice_counts) > 0:
        corr, _ = pearsonr(study_errors, slice_counts)
        print(f"Correlation between Error Magnitude and Number of Slices: {corr:.4f}")
    else:
        print("Correlation could not be calculated (insufficient variance or data).")


def generate_submission(model, device):
    """
    Generates submission.csv for the test set.
    """
    # Load Test Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Setup Dataset & Loader
    test_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_dataset = CervicalSpineDataset(
        test_df, mode="test", transforms=test_transforms, load_cached_data=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    study_preds = {}
    class_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    with torch.no_grad():
        for images, positions, _, study_ids in test_loader:
            images = images.to(device)
            positions = positions.to(device)

            logits = model(images, positions)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, study_id in enumerate(study_ids):
                study_preds[study_id] = {}
                for j, cls in enumerate(class_names):
                    study_preds[study_id][cls] = probs[i, j]

    # Format Submission
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    def get_pred(row_id):
        if row_id.endswith("patient_overall"):
            study_id = row_id.replace("_patient_overall", "")
            cls = "patient_overall"
        else:
            parts = row_id.split("_")
            cls = parts[-1]
            study_id = "_".join(parts[:-1])

        if study_id in study_preds:
            return study_preds[study_id].get(cls, 0.5)
        return 0.5

    sample_sub["fractured"] = sample_sub["row_id"].apply(get_pred)
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    Config.EPOCHS = 5

    # 2. Data Loading
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Transforms
    train_transforms = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Datasets
    train_dataset = CervicalSpineDataset(
        train_df, mode="train", transforms=train_transforms, load_cached_data=True
    )
    val_dataset = CervicalSpineDataset(
        val_df, mode="val", transforms=val_transforms, load_cached_data=True
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Training Components
    model = AnatomicallyGuidedResNet(pretrained=Config.PRETRAINED).to(device)
    criterion = HierarchicalCompoundLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.EPOCHS * Config.T_MAX_MULTIPLIER)
    )

    # 4. Training Loop
    best_metric = float("inf")
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device, val_df)

        scheduler.step()

        # Checkpoint
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, final_metric = validate(model, val_loader, criterion, device, val_df)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    analyze_failures(model, val_loader, val_df, device)

    # 7. Conditional Submission
    THRESHOLD = 0.1307335607
    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
