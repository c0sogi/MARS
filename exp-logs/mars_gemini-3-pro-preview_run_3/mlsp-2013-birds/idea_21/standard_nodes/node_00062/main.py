import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
import torchvision.transforms.functional as TF
import warnings

# Import provided library modules
from library.utils import seed_everything, compute_multilabel_auc
from library.dataset import (
    process_and_cache_data,
    BirdDataset,
    get_transforms,
    IMG_SIZE,
)
from library.models import get_model
from library.trainer import Trainer
from library.sam import SAM

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 4
FOLDS = 5
MODELS_TO_RUN = ["resnet18", "efficientnet_b0", "densenet121"]
SUBMISSION_THRESHOLD = 0.9479806884980326
CHECKPOINT_BASE = "./working/checkpoints"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Suppress warnings
warnings.filterwarnings("ignore")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DeterministicShift:
    """
    Applies a deterministic horizontal shift for TTA.
    """

    def __init__(self, shift_pct):
        self.shift_pct = shift_pct

    def __call__(self, img):
        # img is PIL Image
        w, h = img.size
        # translate=(tx, ty). tx is pixels.
        # If shift_pct is positive, shifts right. Negative shifts left.
        # We fill with 0 (black padding)
        return TF.affine(
            img, angle=0, translate=(self.shift_pct * w, 0), scale=1.0, shear=0, fill=0
        )


def get_tta_transforms(shift_pct=0.0):
    """
    Returns transforms for TTA with deterministic shift.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    ts = [transforms.Resize((IMG_SIZE, IMG_SIZE))]

    if shift_pct != 0.0:
        ts.append(DeterministicShift(shift_pct))

    ts.append(transforms.ToTensor())
    ts.append(transforms.Normalize(mean, std))

    return transforms.Compose(ts)


def run_training_pipeline():
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load and Merge Data
    print("Loading and merging data...")
    train_imgs, train_lbls = process_and_cache_data("train", load_cached_data=True)
    val_imgs, val_lbls = process_and_cache_data("val", load_cached_data=True)

    # Merge
    all_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    all_lbls = np.concatenate([train_lbls, val_lbls], axis=0)

    print(f"Total development samples: {len(all_imgs)}")

    # 2. Setup Cross-Validation
    # Using KFold with shuffle as a robust fallback for IterativeStratification
    # given the extremely low sample count for some classes (min=2).
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    # Store OOF predictions: (N_samples, N_classes)
    # We need to aggregate OOF preds from all models.
    # Strategy: Average OOF probabilities across models for the same sample?
    # Or just track one set of OOF per model to compute CV score.
    # We will compute the ensemble OOF score.
    ensemble_oof_preds = np.zeros_like(all_lbls, dtype=np.float32)

    # We also need to track which indices were predicted to normalize if needed,
    # but with KFold every sample is predicted exactly once.

    # 3. Training Loop
    model_oof_auc_scores = {}

    for model_name in MODELS_TO_RUN:
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        model_oof_preds = np.zeros_like(all_lbls, dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_imgs, all_lbls)):
            print(f"\n--- Fold {fold} ---")

            # Prepare Datasets
            train_subset = BirdDataset(
                all_imgs[train_idx],
                all_lbls[train_idx],
                transform=get_transforms("train"),
            )
            val_subset = BirdDataset(
                all_imgs[val_idx], all_lbls[val_idx], transform=get_transforms("val")
            )

            train_loader = DataLoader(
                train_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

            # Initialize Model
            model = get_model(model_name, num_classes=19, pretrained=True)
            model = model.to(device)

            # Optimizer & Scheduler
            # SAM with AdamW
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(
                model.parameters(), base_optimizer, lr=1e-3, weight_decay=1e-2
            )

            # Cosine Annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer, T_max=EPOCHS
            )

            # Trainer
            ckpt_dir = os.path.join(CHECKPOINT_BASE, model_name)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                fold=fold,
                epochs=EPOCHS,
                patience=PATIENCE,
                checkpoint_dir=ckpt_dir,
            )

            # Fit
            trainer.fit()

            # Generate OOF Preds for this fold
            # Reload best model
            best_model_path = os.path.join(ckpt_dir, f"best_model_fold_{fold}.pth")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)
                    fold_preds.append(preds.cpu().numpy())

            model_oof_preds[val_idx] = np.concatenate(fold_preds, axis=0)

        # Compute AUC for this model
        model_auc = compute_multilabel_auc(all_lbls, model_oof_preds)
        model_oof_auc_scores[model_name] = model_auc
        print(f"Model {model_name} CV AUC: {model_auc:.6f}")

        # Add to ensemble OOF
        ensemble_oof_preds += model_oof_preds

    # Average ensemble OOF
    ensemble_oof_preds /= len(MODELS_TO_RUN)
    final_cv_auc = compute_multilabel_auc(all_lbls, ensemble_oof_preds)

    print(f"\nFinal Validation Metric: {final_cv_auc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Compute per-sample error: Mean Absolute Error across classes
    # error shape: (N,)
    per_sample_error = np.mean(np.abs(all_lbls - ensemble_oof_preds), axis=1)

    # Feature 1: Number of Labels (Cardinality)
    num_labels = np.sum(all_lbls, axis=1)

    # Feature 2: Image Brightness (approximate signal feature)
    # Calculate mean brightness for each image
    brightness = np.mean(all_imgs, axis=(1, 2, 3))  # (N,)

    # Correlations
    corr_labels = np.corrcoef(per_sample_error, num_labels)[0, 1]
    corr_bright = np.corrcoef(per_sample_error, brightness)[0, 1]

    print(f"Correlation (Error vs Label Count): {corr_labels:.6f}")
    print(f"Correlation (Error vs Brightness): {corr_bright:.6f}")

    # 5. Submission
    if final_cv_auc > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric {final_cv_auc} > Threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        generate_submission(device)
    else:
        print(
            f"\nMetric {final_cv_auc} <= Threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


def generate_submission(device):
    # Load Test Data
    test_imgs, _ = process_and_cache_data("test", load_cached_data=True)
    test_df = pd.read_csv("./metadata/test.csv")

    # TTA Transforms
    transforms_list = [
        get_tta_transforms(shift_pct=0.0),  # Original
        get_tta_transforms(shift_pct=-0.15),  # Left Shift
        get_tta_transforms(shift_pct=0.15),  # Right Shift
    ]

    ensemble_test_preds = np.zeros((len(test_imgs), 19), dtype=np.float32)

    # Iterate over all models and folds
    total_models = 0

    for model_name in MODELS_TO_RUN:
        ckpt_dir = os.path.join(CHECKPOINT_BASE, model_name)

        for fold in range(FOLDS):
            model_path = os.path.join(ckpt_dir, f"best_model_fold_{fold}.pth")
            if not os.path.exists(model_path):
                continue

            # Load Model
            model = get_model(model_name, num_classes=19, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()

            # TTA Inference
            fold_tta_preds = np.zeros((len(test_imgs), 19), dtype=np.float32)

            for t_idx, transform in enumerate(transforms_list):
                # Create dataset with specific TTA transform
                ds = BirdDataset(test_imgs, None, transform=transform)
                dl = DataLoader(
                    ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                )

                pass_preds = []
                with torch.no_grad():
                    for images in dl:
                        images = images.to(device)
                        outputs = model(images)
                        preds = torch.sigmoid(outputs)
                        pass_preds.append(preds.cpu().numpy())

                fold_tta_preds += np.concatenate(pass_preds, axis=0)

            # Average TTA passes
            fold_tta_preds /= len(transforms_list)

            # Add to ensemble
            ensemble_test_preds += fold_tta_preds
            total_models += 1

    # Average Ensemble
    if total_models > 0:
        ensemble_test_preds /= total_models

    # Format Submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    ids = []
    probs = []

    for idx, row in test_df.iterrows():
        rec_id = int(row["rec_id"])
        preds = ensemble_test_preds[idx]

        for species_idx in range(19):
            ids.append(rec_id * 100 + species_idx)
            probs.append(preds[species_idx])

    sub_df = pd.DataFrame({"Id": ids, "Probability": probs})
    sub_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")


if __name__ == "__main__":
    run_training_pipeline()
