import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config, seed_everything
from library.dataset import process_metadata, DogDataset, get_transforms
from library.model import DogClassifier
from library.trainer import run_fold, predict
from library.inference import generate_submission
from library.utils import get_checkpoint_path


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    print("Starting execution of runfile.py...")

    # 2. Data Preparation
    # Load metadata (utilizing cache if available for speed)
    train_meta, val_meta, test_meta, classes = process_metadata(load_cached_data=True)

    # Combine provided train and val metadata to perform our own Stratified K-Fold split
    # This ensures we have control over the fold definitions
    all_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Define Folds
    # We use 5 splits to maintain the standard 80/20 distribution structure
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
    all_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df["breed"])):
        all_df.loc[val_idx, "fold"] = fold

    # 3. Train Fold 0 (Fast Baseline)
    # We only train the first fold to minimize runtime while still verifying the idea
    fold_idx = 0
    print(f"Training Fold {fold_idx} only (Fast Baseline)...")

    train_df = all_df[all_df["fold"] != fold_idx].reset_index(drop=True)
    val_df = all_df[all_df["fold"] == fold_idx].reset_index(drop=True)

    # Execute the 3-Phase training strategy (Head Adapt -> Finetune -> SWA)
    ckpt_name = run_fold(fold_idx, train_df, val_df, classes, device)

    # 4. Validation Assessment
    print("\n=== Validation Assessment ===")

    # Load the trained SWA model
    model = DogClassifier(num_classes=len(classes), pretrained=False)
    ckpt_path = get_checkpoint_path(ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)

    # Handle SWA/AveragedModel state dict keys (strip 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        elif k == "n_averaged":
            continue
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Prepare Validation Loader
    class_to_idx = {c: i for i, c in enumerate(classes)}
    val_dataset = DogDataset(
        val_df, transform=get_transforms("val"), mode="val", label_map=class_to_idx
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Generate Predictions on Validation Set
    print("Generating validation predictions...")
    val_probs = predict(model, val_loader, device, use_tta=Config.use_tta)
    val_probs_np = val_probs.numpy()
    val_targets_np = np.array(val_dataset.labels)

    # Calculate Final Metric (Multi Class Log Loss)
    final_metric = log_loss(
        val_targets_np, val_probs_np, labels=list(range(len(classes)))
    )
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate per-sample loss (Negative Log Likelihood of the true class)
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs_np, eps, 1 - eps)
    # Extract probability assigned to the true class
    true_class_probs = val_probs_clipped[np.arange(len(val_targets_np)), val_targets_np]
    losses = -np.log(true_class_probs)

    # Extract Input Features (Width, Height, Aspect Ratio)
    print("Extracting image metadata for analysis...")
    widths = []
    heights = []
    aspect_ratios = []

    for rel_path in val_df["file_path"]:
        full_path = os.path.join(Config.input_dir, rel_path)
        # Read image dimensions
        img = cv2.imread(full_path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            # Fallback (should not occur with verified metadata)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "loss": losses,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "area": np.array(widths) * np.array(heights),
        }
    )

    # Calculate Correlations
    correlations = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error Magnitude (Loss) and Input Features:")
    print(correlations)

    # 6. Submission Logic
    threshold = 0.14144190501755333
    if final_metric < threshold:
        print(f"\nMetric {final_metric:.6f} is lower than threshold {threshold:.6f}.")
        print("Generating submission for Test Set...")

        # Prepare Test Loader
        test_dataset = DogDataset(
            test_meta, transform=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Generate Submission using the trained model
        # We pass the list of checkpoints (just one in this baseline case)
        generate_submission(test_loader, [ckpt_name], classes, device)

    else:
        print(
            f"\nMetric {final_metric:.6f} is NOT lower than threshold {threshold:.6f}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
