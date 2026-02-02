import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.train import run_fold
from library.models import BirdClassifier
from library.dataset import BirdDataset, get_transforms, get_data_splits
from library.utils import seed_everything, calculate_roc_auc, load_checkpoint

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure directories exist
    Config.setup()

    # 2. Train Ensemble & Inference
    # We iterate over Architectures, Data Sources, and Folds
    architectures = Config.ARCHITECTURES
    data_sources = Config.DATA_SOURCES
    n_folds = Config.N_FOLDS

    # Load the folds dataframe to know the size and alignment for OOF
    df_folds = get_data_splits(load_cached_data=True)
    num_samples = len(df_folds)
    num_classes = Config.NUM_CLASSES

    # Accumulator for OOF predictions: (N_samples, N_classes)
    oof_preds_accumulator = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA)
    # Accumulator for Test predictions: (N_test_samples, N_classes)
    test_preds_accumulator = np.zeros((len(df_test), num_classes), dtype=np.float32)

    total_models = len(architectures) * len(data_sources) * n_folds
    model_count = 0

    print(f"Starting execution for {total_models} models...")

    for arch in architectures:
        for source in data_sources:
            for fold in range(n_folds):
                model_count += 1
                # print(f"Processing: {arch} | {source} | Fold {fold}")

                # --- A. Training ---
                # Check if checkpoint already exists
                ckpt_name = f"{arch}_{source}_fold_{fold}.pth"
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{ckpt_name}")

                if not os.path.exists(ckpt_path):
                    # Train the model
                    run_fold(fold, arch, source)

                # --- B. Inference ---
                # Initialize model structure
                model = BirdClassifier(backbone_name=arch, pretrained=False)
                model.to(device)

                # Load best weights (EMA)
                try:
                    _, _ = load_checkpoint(f"best_{ckpt_name}", model, device=device)
                except FileNotFoundError:
                    print(
                        f"Error: Checkpoint {ckpt_path} not found. Skipping inference for this model."
                    )
                    continue

                model.eval()

                # 1. OOF Inference (Validation Fold)
                # Identify validation indices for this fold
                val_indices = df_folds[df_folds["fold"] == fold].index.values
                val_df = df_folds.iloc[val_indices].reset_index(drop=True)

                val_dataset = BirdDataset(
                    val_df,
                    data_source=source,
                    phase="val",
                    transform=get_transforms("val"),
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                )

                fold_val_preds = []
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(device)
                        logits = model(images)
                        probs = torch.sigmoid(logits)
                        fold_val_preds.append(probs.cpu().numpy())

                if fold_val_preds:
                    fold_val_preds = np.concatenate(fold_val_preds, axis=0)
                    # Accumulate predictions
                    oof_preds_accumulator[val_indices] += fold_val_preds

                # 2. Test Inference
                test_dataset = BirdDataset(
                    df_test,
                    data_source=source,
                    phase="test",
                    transform=get_transforms("test"),
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                )

                fold_test_preds = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        images = images.to(device)
                        logits = model(images)
                        probs = torch.sigmoid(logits)
                        fold_test_preds.append(probs.cpu().numpy())

                if fold_test_preds:
                    fold_test_preds = np.concatenate(fold_test_preds, axis=0)
                    test_preds_accumulator += fold_test_preds

                # Cleanup to free memory
                del model, val_loader, test_loader, val_dataset, test_dataset
                torch.cuda.empty_cache()

    # 3. Finalize Predictions

    # Average OOF predictions
    # Each sample was predicted by (len(architectures) * len(data_sources)) models
    num_ensemble_members = len(architectures) * len(data_sources)
    oof_preds_final = oof_preds_accumulator / num_ensemble_members

    # Average Test predictions
    # Each sample was predicted by TOTAL_MODELS (30)
    test_preds_final = test_preds_accumulator / total_models

    # 4. Validation Metric
    # Construct y_true for OOF from the dataframe
    y_true_oof = np.zeros((num_samples, num_classes))
    for idx, row in df_folds.iterrows():
        l_str = str(row["labels"])
        if l_str != "?" and l_str.lower() != "nan" and l_str.strip():
            try:
                indices = [int(x) for x in l_str.split()]
                indices = [i for i in indices if 0 <= i < num_classes]
                y_true_oof[idx, indices] = 1
            except ValueError:
                pass

    final_val_metric = calculate_roc_auc(y_true_oof, oof_preds_final)
    print(f"Final Validation Metric: {final_val_metric}")

    # 5. Failure Analysis
    print("Failure Analysis:")
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(y_true_oof - oof_preds_final), axis=1)

    # Get number of labels per sample
    num_labels = y_true_oof.sum(axis=1)

    # Correlation
    if np.std(mae_per_sample) > 0 and np.std(num_labels) > 0:
        corr = np.corrcoef(mae_per_sample, num_labels)[0, 1]
        print(f"Correlation between MAE and Num Labels: {corr}")
    else:
        print("Correlation between MAE and Num Labels: Undefined (zero variance)")

    # 6. Submission
    threshold = 0.9479806884980326
    if final_val_metric > threshold:
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = test_preds_final[i]
            for species_idx, prob in enumerate(probs):
                # Format: rec_id * 100 + species_id
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
