import os
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    log_message,
    print_metric,
    ensure_directory,
)
from library.dataset import load_datasets, DogDataset, get_transforms
from library.model import DogClassifier
from library.engine import train_loop, predict_tta, evaluate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    ensure_directory(Config.WORKING_DIR)

    log_message("Starting Stratified K-Fold Ensemble Training...")

    # 2. Load Data
    # load_datasets handles combining train/val and debug subsetting
    full_train_df, test_df, class_to_idx, classes = load_datasets()

    log_message(f"Total Training Samples: {len(full_train_df)}")
    log_message(f"Total Test Samples: {len(test_df)}")
    log_message(f"Number of Classes: {len(classes)}")

    # 3. Initialize K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF predictions and Test predictions
    # OOF: (N_train, N_classes)
    oof_preds = np.zeros((len(full_train_df), Config.NUM_CLASSES), dtype=np.float32)
    # Store true labels for verification/metric calc
    oof_labels = np.zeros(len(full_train_df), dtype=np.int64)

    # Test Accumulator: (N_test, N_classes)
    test_preds_sum = np.zeros((len(test_df), Config.NUM_CLASSES), dtype=np.float32)

    # 4. Training Loop
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["breed"])
    ):
        log_message(
            f"\n========================= FOLD {fold_idx + 1}/{Config.N_FOLDS} ========================="
        )

        # 4.1 Prepare DataFrames for this fold
        train_fold_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_train_df.iloc[val_idx].reset_index(drop=True)

        # 4.2 Create Datasets and DataLoaders
        train_dataset = DogDataset(
            train_fold_df, transform=get_transforms("train"), class_to_idx=class_to_idx
        )
        val_dataset = DogDataset(
            val_fold_df, transform=get_transforms("val"), class_to_idx=class_to_idx
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 4.3 Initialize Model
        model = DogClassifier(
            model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES
        ).to(device)

        # 4.4 Train (Phase 1 & 2)
        train_loop(model, train_loader, val_loader, fold_idx, device)

        # 4.5 Load Best Model for Inference
        best_model_path = Config.get_model_path(fold_idx)
        log_message(f"Loading best model from {best_model_path} for inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # 4.6 Validation Inference (OOF)
        # evaluate returns: metric, preds, labels
        _, fold_val_preds, fold_val_labels = evaluate(model, val_loader, device)

        # Store OOF predictions
        oof_preds[val_idx] = fold_val_preds
        oof_labels[val_idx] = fold_val_labels

        # 4.7 Test Inference (TTA)
        test_dataset = DogDataset(
            test_df,
            transform=get_transforms("test"),
            return_ids=True,
            input_dir=Config.INPUT_DIR,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        _, fold_test_preds = predict_tta(model, test_loader, device)
        test_preds_sum += fold_test_preds

        # Cleanup
        del (
            model,
            train_loader,
            val_loader,
            test_loader,
            train_dataset,
            val_dataset,
            test_dataset,
        )
        torch.cuda.empty_cache()

    # 5. Final Validation Analysis
    log_message("\n========================= FINAL ANALYSIS =========================")

    # 5.1 Calculate Overall Metric
    final_metric = log_loss(
        oof_labels, oof_preds, labels=list(range(Config.NUM_CLASSES))
    )
    print_metric("Final Validation Metric", final_metric)

    # 5.2 Failure Analysis
    log_message("Performing Failure Analysis...")

    # Calculate Cross Entropy Loss per sample
    # Get probability assigned to the true class
    true_probs = oof_preds[np.arange(len(oof_labels)), oof_labels]
    true_probs = np.clip(true_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_probs)

    # Extract Image Metadata
    widths, heights, aspect_ratios, areas = [], [], [], []

    for idx, row in full_train_df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(file_path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            areas.append(w * h)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            areas.append(0)

    # Compute Correlation
    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "area": areas,
        }
    )

    correlations = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error Magnitude (Log Loss) and Input Features:")
    print(correlations)

    # 6. Submission Generation
    TARGET_METRIC = 0.14144190501755333

    if final_metric < TARGET_METRIC:
        log_message(
            f"\nMetric {final_metric} meets threshold {TARGET_METRIC}. Generating submission..."
        )

        # Average test predictions
        avg_test_preds = test_preds_sum / Config.N_FOLDS

        # Create Submission DataFrame
        submission = pd.DataFrame(avg_test_preds, columns=classes)
        submission.insert(0, "id", test_df["id"])

        # Save
        submission_path = "./submission/submission.csv"
        ensure_directory(os.path.dirname(submission_path))
        submission.to_csv(submission_path, index=False)
        log_message(f"Submission saved to {submission_path}")
    else:
        log_message(
            f"\nMetric {final_metric} does NOT meet threshold {TARGET_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
