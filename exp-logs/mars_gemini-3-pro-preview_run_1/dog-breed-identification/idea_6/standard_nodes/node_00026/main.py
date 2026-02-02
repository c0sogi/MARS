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

    # 3. Train All Folds (Ensemble Strategy)
    print(f"Training {Config.n_folds} Folds (Ensemble Strategy)...")

    model_paths = []
    oof_probs_list = []
    oof_targets_list = []
    oof_file_paths = []

    class_to_idx = {c: i for i, c in enumerate(classes)}

    for fold_idx in range(Config.n_folds):
        print(f"\n--- Processing Fold {fold_idx} ---")
        train_df = all_df[all_df["fold"] != fold_idx].reset_index(drop=True)
        val_df = all_df[all_df["fold"] == fold_idx].reset_index(drop=True)

        # Execute the 3-Phase training strategy
        ckpt_name = run_fold(fold_idx, train_df, val_df, classes, device)
        model_paths.append(ckpt_name)

        # --- Validation for this Fold ---
        print(f"Generating predictions for Fold {fold_idx} validation set...")
        model = DogClassifier(num_classes=len(classes), pretrained=False)
        ckpt_path = get_checkpoint_path(ckpt_name)

        state_dict = torch.load(ckpt_path, map_location=device)
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

        val_probs = predict(model, val_loader, device, use_tta=Config.use_tta)

        oof_probs_list.append(val_probs.numpy())
        oof_targets_list.append(np.array(val_dataset.labels))
        oof_file_paths.extend(val_df["file_path"].tolist())

    # 4. Validation Assessment (Global OOF)
    print("\n=== Validation Assessment (OOF) ===")

    oof_probs_np = np.concatenate(oof_probs_list, axis=0)
    oof_targets_np = np.concatenate(oof_targets_list, axis=0)

    # Calculate Final Metric (Multi Class Log Loss)
    final_metric = log_loss(
        oof_targets_np, oof_probs_np, labels=list(range(len(classes)))
    )
    print(f"Final Validation Metric (OOF): {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    eps = 1e-15
    val_probs_clipped = np.clip(oof_probs_np, eps, 1 - eps)
    true_class_probs = val_probs_clipped[np.arange(len(oof_targets_np)), oof_targets_np]
    losses = -np.log(true_class_probs)

    print("Extracting image metadata for analysis...")
    widths = []
    heights = []
    aspect_ratios = []

    for rel_path in oof_file_paths:
        full_path = os.path.join(Config.input_dir, rel_path)
        img = cv2.imread(full_path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    analysis_df = pd.DataFrame(
        {
            "loss": losses,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "area": np.array(widths) * np.array(heights),
        }
    )

    correlations = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error Magnitude (Loss) and Input Features:")
    print(correlations)

    # 6. Submission Logic
    threshold = 0.12970461086690332
    if final_metric < threshold:
        print(f"\nMetric {final_metric:.6f} is lower than threshold {threshold:.6f}.")
        print("Generating submission for Test Set (Ensemble)...")

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

        generate_submission(test_loader, model_paths, classes, device)

    else:
        print(
            f"\nMetric {final_metric:.6f} is NOT lower than threshold {threshold:.6f}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
