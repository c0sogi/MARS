import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# Import library modules
from library.config import Config, seed_everything
from library.training import run_training
from library.modeling import CassavaClassifier
from library.data import CassavaDataset, get_transforms
from library.utils import get_logger


def setup_fast_baseline():
    """
    Overrides Config parameters for a fast baseline run and creates a data subset.
    """
    print("Setting up fast baseline configuration...")

    # 1. Reduce Epochs
    Config.EPOCHS_WARMUP = 1
    Config.EPOCHS_BASE = 1
    Config.EPOCHS_FINE = 1
    Config.EPOCHS_SWA = 1

    # 2. Create Training Subset
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    # Stratified subsample to 2000 images for speed
    subset_df, _ = train_test_split(
        full_train_df,
        train_size=2000,
        stratify=full_train_df["label"],
        random_state=Config.SEED,
    )

    subset_path = os.path.join(Config.OUTPUT_DIR, "train_subset.csv")
    subset_df.to_csv(subset_path, index=False)

    # Point Config to the subset
    Config.TRAIN_CSV = subset_path
    print(f"Training subset created at {subset_path} with {len(subset_df)} samples.")
    print(
        f"Epoch configuration: Warmup={Config.EPOCHS_WARMUP}, Base={Config.EPOCHS_BASE}, "
        f"Fine={Config.EPOCHS_FINE}, SWA={Config.EPOCHS_SWA}"
    )


def load_models(device):
    """
    Loads the trained SWA models for ensemble inference.
    """
    models = []
    for arch in Config.MODEL_ARCHS:
        checkpoint_path = os.path.join(Config.OUTPUT_DIR, f"{arch}_swa_final.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for {arch} at {checkpoint_path}")
            continue

        model = CassavaClassifier(arch, pretrained=False)
        model.to(device)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle state dict key mismatch if necessary (though library.utils handles it usually,
        # here we load directly from what run_training saves)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        # Strip module. prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()
        models.append(model)

    return models


def predict_with_tta(models, loader, device):
    """
    Performs inference with Test Time Augmentation (Horizontal and Vertical Flips)
    and averages predictions across all models in the ensemble.
    """
    all_probs = []
    all_targets = []
    all_image_ids = []

    # TTA: Original, HFlip, VFlip
    # Since we use DataLoader, we apply flips on the batch tensor

    with torch.no_grad():
        for batch in loader:
            # Handle different return signatures of CassavaDataset (test vs val)
            if len(batch) == 2:
                images, targets_or_ids = batch
            else:
                raise ValueError("Unexpected batch format")

            images = images.to(device)

            batch_probs = torch.zeros(images.size(0), Config.NUM_CLASSES, device=device)

            # Iterate over each model in the ensemble
            for model in models:
                # 1. Original
                p1 = F.softmax(model(images), dim=1)

                # 2. Horizontal Flip
                p2 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)

                # 3. Vertical Flip
                p3 = F.softmax(model(torch.flip(images, dims=[2])), dim=1)

                # Average TTA for this model
                model_avg = (p1 + p2 + p3) / 3.0
                batch_probs += model_avg

            # Average across ensemble
            if len(models) > 0:
                batch_probs /= len(models)

            all_probs.append(batch_probs.cpu())

            # Store targets or IDs depending on mode
            if isinstance(targets_or_ids, torch.Tensor):
                all_targets.append(targets_or_ids.cpu())
            else:
                all_image_ids.extend(targets_or_ids)

    all_probs = torch.cat(all_probs, dim=0)

    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
        return all_probs, all_targets
    else:
        return all_probs, all_image_ids


def perform_failure_analysis(val_df, probs, targets):
    """
    Analyzes prediction errors correlated with input features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude: 1.0 - probability assigned to the correct class
    # Gather probabilities for the true class
    true_class_probs = probs[torch.arange(len(targets)), targets]
    error_magnitudes = 1.0 - true_class_probs.numpy()

    # Extract metadata features
    file_sizes = []
    aspect_ratios = []

    print("Extracting metadata features for validation set...")
    # We need to read files to get this info as it's not in val.csv
    # Paths are relative to ./input
    input_root = "./input"

    for _, row in val_df.iterrows():
        full_path = os.path.join(input_root, row["file_path"])

        # File Size
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))

            # Aspect Ratio (Read header only if possible, but cv2 reads full)
            # To be fast, we assume standard reading.
            # Since we are in failure analysis, speed is less critical than training loop
            try:
                img = cv2.imread(full_path)
                if img is not None:
                    h, w, _ = img.shape
                    aspect_ratios.append(w / h if h > 0 else 0)
                else:
                    aspect_ratios.append(0)
            except:
                aspect_ratios.append(0)
        else:
            file_sizes.append(0)
            aspect_ratios.append(0)

    file_sizes = np.array(file_sizes)
    aspect_ratios = np.array(aspect_ratios)

    # Correlations
    if len(file_sizes) > 1:
        corr_size, _ = pearsonr(file_sizes, error_magnitudes)
        print(f"Correlation between Error Magnitude and File Size: {corr_size:.4f}")

    if len(aspect_ratios) > 1:
        corr_ar, _ = pearsonr(aspect_ratios, error_magnitudes)
        print(f"Correlation between Error Magnitude and Aspect Ratio: {corr_ar:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    setup_fast_baseline()

    device = Config.DEVICE

    # 2. Train Models
    for arch in Config.MODEL_ARCHS:
        try:
            run_training(arch, logger_name=f"train_{arch}.log")
        except Exception as e:
            print(f"Error training {arch}: {e}")
            # Continue to next model or validation if one fails

    # 3. Validation Inference
    print("\nStarting Validation Inference...")
    val_df = pd.read_csv(Config.VAL_CSV)

    # Use High-Res for validation
    val_dataset = CassavaDataset(
        val_df,
        transform=get_transforms(Config.IMG_SIZE_HIGH, mode="valid"),
        mode="valid",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Double batch size for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    models = load_models(device)
    if not models:
        print("No models loaded. Exiting.")
        return

    val_probs, val_targets = predict_with_tta(models, val_loader, device)

    # Calculate Accuracy
    val_preds = val_probs.argmax(dim=1)
    correct = val_preds.eq(val_targets).sum().item()
    val_accuracy = correct / len(val_targets)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_accuracy}")

    # 4. Failure Analysis
    perform_failure_analysis(val_df, val_probs, val_targets)

    # 5. Submission
    THRESHOLD = 0.9076101468624833

    if val_accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({val_accuracy}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_df = pd.read_csv(Config.TEST_CSV)
        test_dataset = CassavaDataset(
            test_df,
            transform=get_transforms(Config.IMG_SIZE_HIGH, mode="test"),
            mode="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_probs, test_ids = predict_with_tta(models, test_loader, device)
        test_preds = test_probs.argmax(dim=1).numpy()

        submission_df = pd.DataFrame({"image_id": test_ids, "label": test_preds})

        sub_path = Config.SUBMISSION_FILE
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nValidation accuracy ({val_accuracy}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
