import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from skmultilearn.model_selection import IterativeStratification

# Import library modules
from library.config import CFG
from library.utils import seed_everything, calculate_metric
from library.data import BirdDataset, get_transforms, load_image_dict, get_test_loader
from library.models import get_model
from library.losses import WeightedBCELoss, WeightedDistillationLoss
from library.engine import (
    train_one_epoch,
    train_distill_one_epoch,
    valid_one_epoch,
    predict_with_tta,
)


# --- Custom Dataset for Distillation (Stage 2) ---
class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dict, teacher_probs, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_dict = image_dict
        self.teacher_probs = teacher_probs  # (N, num_classes)
        self.transforms = transforms
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = self.df[self.label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec_id = self.df.iloc[idx]["rec_id"]
        image = self.image_dict.get(rec_id)

        # Safety fallback
        if image is None:
            image = np.zeros((CFG.img_height, CFG.img_width, 3), dtype=np.uint8)

        if self.transforms:
            image = self.transforms(image=image)["image"]
        else:
            # Basic to tensor if no transforms
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        # Pack labels: [GroundTruth (19), TeacherProbs (19)]
        gt = self.labels[idx]
        teacher = self.teacher_probs[idx]
        packed = np.concatenate([gt, teacher])

        return image, torch.tensor(packed, dtype=torch.float32)


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    os.makedirs(CFG.working_dir, exist_ok=True)

    # 2. Load Data
    # Combine train and val to form the full development set for CV
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    dev_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    test_df = pd.read_csv(CFG.test_csv)

    # Pre-load images into memory/cache
    print("Loading images...")
    image_dict = load_image_dict(dev_df, load_cached_data=True, cache_name="dev_pool")

    # 3. Create Stratified Folds
    X = dev_df["rec_id"].values.reshape(-1, 1)
    y = dev_df[[c for c in dev_df.columns if c.startswith("species_")]].values.astype(
        int
    )

    k_fold = IterativeStratification(n_splits=CFG.n_folds, order=1)
    dev_df["fold"] = -1

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        dev_df.loc[val_indices, "fold"] = fold_idx

    # Initialize OOF arrays
    oof_preds_resnet = np.zeros((len(dev_df), CFG.num_classes))
    oof_preds_effnet = np.zeros((len(dev_df), CFG.num_classes))
    oof_preds_densenet = np.zeros((len(dev_df), CFG.num_classes))

    # Store model paths for final inference
    model_paths = []

    # 4. Stage 1: Train Anchors (ResNet18, EfficientNet-B0)
    print("\n=== Stage 1: Training Anchors ===")
    anchors = ["resnet18", "efficientnet_b0"]

    for fold in range(CFG.n_folds):
        print(f"-- Fold {fold} --")

        # Split Data
        train_fold = dev_df[dev_df["fold"] != fold].reset_index(drop=True)
        valid_fold = dev_df[dev_df["fold"] == fold].reset_index(drop=True)
        val_indices = dev_df[dev_df["fold"] == fold].index

        # Datasets & Loaders
        train_ds = BirdDataset(
            train_fold, image_dict, transforms=get_transforms("train")
        )
        valid_ds = BirdDataset(
            valid_fold, image_dict, transforms=get_transforms("valid")
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )

        # Train each anchor
        for model_name in anchors:
            print(f"Training {model_name}...")
            model = get_model(model_name, pretrained=True).to(CFG.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
            )
            criterion = WeightedBCELoss(device=CFG.device)

            best_auc = 0
            best_path = os.path.join(CFG.working_dir, f"{model_name}_fold_{fold}.pth")

            for epoch in range(CFG.epochs):
                _ = train_one_epoch(
                    model, train_loader, optimizer, criterion, CFG.device, epoch
                )
                _, val_auc = valid_one_epoch(model, valid_loader, criterion, CFG.device)

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_path)

            # Save path
            model_paths.append((model_name, best_path))

            # Generate OOF
            model.load_state_dict(torch.load(best_path))
            preds = predict_with_tta(model, valid_loader, CFG.device)

            if model_name == "resnet18":
                oof_preds_resnet[val_indices] = preds
            else:
                oof_preds_effnet[val_indices] = preds

    # 5. Stage 2: Train Student (DenseNet121) with Distillation
    print("\n=== Stage 2: Training Student (DenseNet121) ===")

    # Compute Teacher Targets (Average of Anchors)
    teacher_targets = (oof_preds_resnet + oof_preds_effnet) / 2.0

    for fold in range(CFG.n_folds):
        print(f"-- Fold {fold} --")

        train_idx = dev_df[dev_df["fold"] != fold].index
        val_idx = dev_df[dev_df["fold"] == fold].index

        train_fold_df = dev_df.loc[train_idx]
        val_fold_df = dev_df.loc[val_idx]

        # Get teacher probs for the training split
        train_teacher_probs = teacher_targets[train_idx]

        # Distillation Dataset
        train_ds = DistillationDataset(
            train_fold_df,
            image_dict,
            train_teacher_probs,
            transforms=get_transforms("train"),
        )
        valid_ds = BirdDataset(
            val_fold_df, image_dict, transforms=get_transforms("valid")
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )

        # Model Setup
        model = get_model("densenet121", pretrained=True).to(CFG.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        criterion = WeightedDistillationLoss(device=CFG.device)
        val_criterion = WeightedBCELoss(device=CFG.device)

        best_auc = 0
        best_path = os.path.join(CFG.working_dir, f"densenet121_fold_{fold}.pth")

        for epoch in range(CFG.epochs):
            _ = train_distill_one_epoch(
                model, train_loader, optimizer, criterion, CFG.device, epoch
            )
            _, val_auc = valid_one_epoch(model, valid_loader, val_criterion, CFG.device)

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_path)

        model_paths.append(("densenet121", best_path))

        # Generate OOF
        model.load_state_dict(torch.load(best_path))
        oof_preds_densenet[val_idx] = predict_with_tta(model, valid_loader, CFG.device)

    # 6. Final Validation & Failure Analysis
    print("\n=== Validation & Analysis ===")

    # Ensemble OOFs
    oof_ensemble = (oof_preds_resnet + oof_preds_effnet + oof_preds_densenet) / 3.0

    # Calculate Final Metric
    final_auc = calculate_metric(y, oof_ensemble)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation of Error with Image Stats
    # Error metric: Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(y - oof_ensemble), axis=1)

    pixel_means = []
    pixel_stds = []

    for _, row in dev_df.iterrows():
        img = image_dict.get(row["rec_id"])
        if img is not None:
            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))
        else:
            pixel_means.append(0)
            pixel_stds.append(0)

    corr_mean, _ = pearsonr(mae_per_sample, pixel_means)
    corr_std, _ = pearsonr(mae_per_sample, pixel_stds)

    print(f"Correlation (Error vs Pixel Mean): {corr_mean}")
    print(f"Correlation (Error vs Pixel Std): {corr_std}")

    # 7. Submission
    threshold = 0.9167709334579945
    if final_auc > threshold:
        print("\n=== Generating Submission ===")

        # Load test images
        test_loader = get_test_loader(
            test_df, batch_size=CFG.batch_size, load_cached_data=True
        )

        final_preds = np.zeros((len(test_df), CFG.num_classes))

        # Run inference on all models
        for model_name, model_path in model_paths:
            model = get_model(model_name, pretrained=False).to(CFG.device)
            model.load_state_dict(torch.load(model_path))
            preds = predict_with_tta(model, test_loader, CFG.device)
            final_preds += preds

        # Average
        final_preds /= len(model_paths)

        # Format Submission
        submission_rows = []
        for i, row in test_df.iterrows():
            rec_id = int(row["rec_id"])
            for species_idx in range(CFG.num_classes):
                prob = final_preds[i, species_idx]
                # ID format: rec_id * 100 + species_idx
                sub_id = rec_id * 100 + species_idx
                submission_rows.append([sub_id, prob])

        sub_df = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

        os.makedirs("submission", exist_ok=True)
        sub_df.to_csv("submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
    else:
        print(f"Metric {final_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
