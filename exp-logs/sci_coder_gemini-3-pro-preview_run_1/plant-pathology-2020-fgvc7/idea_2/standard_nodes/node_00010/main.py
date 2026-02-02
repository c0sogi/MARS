import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import (
    AppleDataset,
    get_transforms,
    load_full_train_data,
    load_test_data,
)
from library.model import AppleResNet34
from library.engine import get_weighted_criterion, fit


def run_failure_analysis(df, preds, device):
    """
    Performs failure analysis by correlating prediction error with image meta-features.
    """
    print("\n==== Failure Analysis ====")

    # Calculate error: 1.0 - probability of the true class
    # Get true class indices
    true_labels = (
        df[Config.CLASS_LABELS]
        .idxmax(axis=1)
        .apply(lambda x: Config.CLASS_LABELS.index(x))
        .values
    )

    # Extract probability assigned to the true class
    # preds is (N, 4)
    prob_true = preds[np.arange(len(preds)), true_labels]
    errors = 1.0 - prob_true

    # Collect meta-features
    widths = []
    heights = []
    intensities = []

    print("Extracting meta-features for failure analysis...")
    for idx, row in df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Read image to get stats (using OpenCV which is fast)
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing images (though check passed in metadata)
            widths.append(0)
            heights.append(0)
            intensities.append(0)
            continue

        h, w, c = img.shape
        # Calculate mean intensity (simple average of channels)
        mean_intensity = img.mean() / 255.0

        widths.append(w)
        heights.append(h)
        intensities.append(mean_intensity)

    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "intensity": intensities}
    )

    # Calculate correlations
    corr_width = analysis_df["error"].corr(analysis_df["width"])
    corr_height = analysis_df["error"].corr(analysis_df["height"])
    corr_intensity = analysis_df["error"].corr(analysis_df["intensity"])

    print(f"Correlation between Error and Width: {corr_width:.4f}")
    print(f"Correlation between Error and Height: {corr_height:.4f}")
    print(f"Correlation between Error and Intensity: {corr_intensity:.4f}")


def inference_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    all_preds = []
    ids = []

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # 1. Original Prediction
            out_orig = model(images)
            prob_orig = torch.softmax(out_orig, dim=1)

            # 2. Flipped Prediction (Horizontal Flip)
            # Input is (B, C, H, W), flip on last dimension (W)
            images_flipped = torch.flip(images, dims=[3])
            out_flip = model(images_flipped)
            prob_flip = torch.softmax(out_flip, dim=1)

            # Average
            avg_prob = (prob_orig + prob_flip) / 2.0

            all_preds.append(avg_prob.cpu().numpy())
            ids.extend(image_ids)

    return np.concatenate(all_preds), ids


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    full_train_df = load_full_train_data()

    # Prepare storage for OOF predictions
    # We need to align OOF preds with the dataframe index
    oof_preds = np.zeros((len(full_train_df), Config.NUM_CLASSES))

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We need a target label for stratification
    if "stratify_label" in full_train_df.columns:
        y_stratify = full_train_df["stratify_label"]
    else:
        y_stratify = full_train_df[Config.CLASS_LABELS].idxmax(axis=1)

    # 3. Training Loop (K-Fold)
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, y_stratify)):
        print(f"\n===== Fold {fold + 1}/{Config.N_FOLDS} =====")

        # Split DataFrames
        train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = AppleDataset(
            train_df, transforms=get_transforms("train"), mode="train"
        )
        val_dataset = AppleDataset(
            val_df, transforms=get_transforms("valid"), mode="train"
        )

        # Create DataLoaders
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

        # Initialize Model, Optimizer, Criterion
        model = AppleResNet34(pretrained=True).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        # Cite solution_lesson_node_00008: Optimization dynamics are primary.
        # Using CosineAnnealingWarmRestarts to improve convergence over constant LR.
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6
        )
        criterion = get_weighted_criterion(train_df, device)

        # Train
        model_save_path = os.path.join(Config.MODELS_DIR, f"resnet34_fold_{fold}.pth")

        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=Config.EPOCHS,
            patience=11,  # Increased patience to allow for scheduler restarts
            save_path=model_save_path,
            scheduler=scheduler,
        )

        # Generate OOF predictions for this fold using the best model
        # Load best weights
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        oof_preds[val_idx] = fold_preds

    # 4. Global Validation Metric
    print("\nCalculating Global Validation Metric...")
    # Get ground truth for all data
    y_true = full_train_df[Config.CLASS_LABELS].values

    final_metric = calculate_roc_auc(y_true, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    run_failure_analysis(full_train_df, oof_preds, device)

    # 6. Submission Logic
    THRESHOLD = 0.9871488489626378

    if final_metric > THRESHOLD:
        print("\nMetric threshold passed. Generating submission...")

        test_df = load_test_data()
        test_dataset = AppleDataset(
            test_df, transforms=get_transforms("valid"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Inference
        ensemble_preds = np.zeros((len(test_df), Config.NUM_CLASSES))

        for fold in range(Config.N_FOLDS):
            print(f"Inference with model from Fold {fold + 1}...")
            model_path = os.path.join(Config.MODELS_DIR, f"resnet34_fold_{fold}.pth")

            model = AppleResNet34(pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            preds, image_ids = inference_tta(model, test_loader, device)
            ensemble_preds += preds

        # Average
        ensemble_preds /= Config.N_FOLDS

        # Create Submission DataFrame
        submission_df = pd.DataFrame(ensemble_preds, columns=Config.CLASS_LABELS)
        submission_df.insert(0, "image_id", image_ids)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
