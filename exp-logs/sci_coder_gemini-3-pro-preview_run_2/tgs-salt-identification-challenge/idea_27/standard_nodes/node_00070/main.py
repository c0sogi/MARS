import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.utils import set_seed
from library.model import ResNet34WideLinkNet
from library.dataset import SaltDataset, get_transforms
from library.training import SaltTrainer, validate
from library.inference import predict_with_tta, optimize_threshold, generate_submission

# Constants
BATCH_SIZE = 32
LR = 1e-4
EPOCHS_STAGE1 = 8  # Reduced for fast baseline execution
EPOCHS_STAGE3 = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
WORKING_DIR = "./working"

# Set seeds
set_seed(SEED)


class PseudoDataset(Dataset):
    """
    Wraps the test dataset to provide soft pseudo-labels for Student training.
    Pads the 101x101 pseudo-masks to 128x128 to match the image transforms.
    """

    def __init__(self, test_dataset, pseudo_masks):
        self.test_dataset = test_dataset
        self.pseudo_masks = pseudo_masks  # Shape: (N, 101, 101)

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        # Get image from test dataset (already transformed/padded to 128x128)
        # test_dataset returns: image, depth, id
        image, depth, id_ = self.test_dataset[idx]

        # Get mask (101x101)
        mask_101 = self.pseudo_masks[idx]

        # Pad mask to 128x128 to match image geometry
        # Padding logic matches library/inference.py cropping: 13 top/left, 14 bottom/right
        mask_padded = np.pad(mask_101, ((13, 14), (13, 14)), mode="reflect")

        # Convert to tensor
        mask_tensor = torch.from_numpy(mask_padded).float()

        return image, mask_tensor, depth, id_


def main():
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Data Loading
    # Full training set for CV
    train_full = SaltDataset(mode="train", transform=get_transforms("train"))
    # Training set with val transforms for fold validation
    train_full_val = SaltDataset(mode="train", transform=get_transforms("val"))

    # Hold-out validation set (Fixed)
    val_holdout = SaltDataset(mode="val", transform=get_transforms("val"))

    # Test set
    test_dataset = SaltDataset(mode="test", transform=get_transforms("val"))

    # DataLoaders
    val_loader = DataLoader(
        val_holdout, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. Stage 1: 5-Fold Ensemble Training
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    indices = np.arange(len(train_full))

    model_paths = []
    model_scores = []

    print("--- Stage 1: 5-Fold Ensemble Training ---")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
        fold_dir = os.path.join(WORKING_DIR, f"fold_{fold}")

        # Create subsets
        train_sub = Subset(train_full, train_idx)
        valid_sub = Subset(train_full_val, valid_idx)

        train_loader_fold = DataLoader(
            train_sub,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        valid_loader_fold = DataLoader(
            valid_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = ResNet34WideLinkNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE1)

        # Train
        trainer = SaltTrainer(
            model, DEVICE, optimizer, scheduler=scheduler, checkpoint_dir=fold_dir
        )
        trainer.fit(
            train_loader_fold, valid_loader_fold, epochs=EPOCHS_STAGE1, patience=3
        )

        # Store results
        model_scores.append(trainer.best_score)
        model_paths.append(os.path.join(fold_dir, "best_model.pth"))

        # Cleanup to save memory
        del (
            model,
            optimizer,
            trainer,
            train_loader_fold,
            valid_loader_fold,
            train_sub,
            valid_sub,
        )
        torch.cuda.empty_cache()

    # 3. Stage 2: Quality Gating & Pseudo-Labeling
    print("--- Stage 2: Gating & Pseudo-Labeling ---")
    valid_models = []
    for path, score in zip(model_paths, model_scores):
        if score >= 0.75:
            valid_models.append(path)
        else:
            print(f"Discarding model {path} (mAP: {score:.4f} < 0.75)")

    if not valid_models:
        print("Warning: No models passed gating. Using the best available model.")
        best_idx = np.argmax(model_scores)
        valid_models.append(model_paths[best_idx])

    # Generate Soft Pseudo Labels (Ensemble Averaging)
    avg_preds = None

    for path in valid_models:
        model = ResNet34WideLinkNet().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

        # Predict on test set (returns cropped 101x101 probabilities)
        preds, _, _ = predict_with_tta(model, test_loader, DEVICE)

        if avg_preds is None:
            avg_preds = preds
        else:
            avg_preds += preds

        del model
        torch.cuda.empty_cache()

    avg_preds /= len(valid_models)

    # 4. Stage 3: Student Training
    print("--- Stage 3: Student Training ---")

    # Create Pseudo Dataset
    pseudo_dataset = PseudoDataset(test_dataset, avg_preds)
    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Full Labeled Train Loader
    train_loader_full = DataLoader(
        train_full, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
    )

    # Initialize Student Model
    student_model = ResNet34WideLinkNet().to(DEVICE)
    optimizer = optim.AdamW(student_model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE3)

    student_dir = os.path.join(WORKING_DIR, "student")
    trainer = SaltTrainer(
        student_model,
        DEVICE,
        optimizer,
        scheduler=scheduler,
        checkpoint_dir=student_dir,
    )

    # Train Student (Semi-Supervised)
    trainer.fit(
        train_loader_full,
        val_loader,
        epochs=EPOCHS_STAGE3,
        patience=3,
        student_mode=True,
        unlabeled_loader=pseudo_loader,
    )

    # 5. Final Evaluation
    print("--- Final Evaluation ---")
    best_student = ResNet34WideLinkNet().to(DEVICE)
    best_student_path = os.path.join(student_dir, "best_model.pth")
    best_student.load_state_dict(torch.load(best_student_path, map_location=DEVICE))

    final_metric = validate(best_student, val_loader, DEVICE)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("--- Failure Analysis ---")
    # Get predictions and ground truths on Val set
    preds, gts, _ = predict_with_tta(best_student, val_loader, DEVICE)

    # Binarize for IoU calculation (using 0.5 for analysis)
    preds_bin = (preds > 0.5).astype(np.float32)

    # Calculate IoU per image
    intersection = np.logical_and(preds_bin, gts).sum(axis=(1, 2))
    union = np.logical_or(preds_bin, gts).sum(axis=(1, 2))
    ious = np.where(union == 0, 1.0, intersection / union)

    # Calculate Error (1 - IoU)
    errors = 1.0 - ious

    # Get Depths for Val set
    val_depths = val_holdout.depths

    # Calculate Correlation
    correlation = np.corrcoef(errors, val_depths)[0, 1]
    print(f"Correlation between Error (1-IoU) and Depth: {correlation:.10f}")

    # 7. Submission
    if final_metric > 0.7985:
        best_thresh = optimize_threshold(best_student, val_loader, DEVICE)
        generate_submission(
            best_student, test_loader, DEVICE, best_thresh, output_dir="./submission"
        )
    else:
        print("Metric too low for submission.")


if __name__ == "__main__":
    main()
