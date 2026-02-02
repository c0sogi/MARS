import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_robust_roc_auc
from library.dataset import BirdDataset
from library.models import get_model
from library.engine import fit_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Override Config parameters for execution speed within time limits
    # Increase epochs and patience to allow Mixup to converge (Cite solution_lesson_node_00040)
    # Ensure T_MAX matches EPOCHS for proper Cosine Annealing decay
    Config.EPOCHS = 50
    Config.PATIENCE = 10
    Config.T_MAX = Config.EPOCHS

    # Initialize directories
    Config.setup()

    # Define directory for fold-specific CSVs
    split_dir = os.path.join(Config.WORKING_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)

    print("Loading training metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA)

    # 2. Stratified K-Fold Split
    # Prepare data for IterativeStratification (Multi-label stratification)
    X = train_df["rec_id"].values.reshape(-1, 1)

    # Convert string labels to binary matrix
    num_samples = len(train_df)
    num_classes = Config.NUM_CLASSES
    y = np.zeros((num_samples, num_classes), dtype=int)

    for idx, row in train_df.iterrows():
        lbl_str = row["labels"]
        if pd.notna(lbl_str) and lbl_str != "?":
            try:
                indices = [int(x) for x in str(lbl_str).split()]
                for i in indices:
                    if 0 <= i < num_classes:
                        y[idx, i] = 1
            except ValueError:
                pass

    k_fold = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)

    folds = {}  # Dictionary to store paths for each fold

    print("Generating stratified folds...")
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        # Split DataFrame
        fold_train_df = train_df.iloc[train_indices].copy()
        fold_val_df = train_df.iloc[val_indices].copy()

        # Save split CSVs
        train_path = os.path.join(split_dir, f"train_fold_{fold_idx}.csv")
        val_path = os.path.join(split_dir, f"val_fold_{fold_idx}.csv")

        fold_train_df.to_csv(train_path, index=False)
        fold_val_df.to_csv(val_path, index=False)

        folds[fold_idx] = (train_path, val_path)

    # 3. Training Loop
    # Ensemble Strategy: 3 Architectures x 5 Folds = 15 Models
    trained_models = []

    for model_name in Config.MODELS:
        for fold_idx in range(Config.NUM_FOLDS):
            print(f"\n=== Training {model_name} | Fold {fold_idx} ===")

            train_csv_path, val_csv_path = folds[fold_idx]

            # Initialize Datasets
            train_ds = BirdDataset(train_csv_path, phase="train", load_cached_data=True)
            val_ds = BirdDataset(val_csv_path, phase="val", load_cached_data=True)

            # Initialize DataLoaders
            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
            )

            # Initialize Model
            model = get_model(model_name)

            # Train Model
            # fit_model handles the training loop, validation, and saving best weights
            _ = fit_model(model, train_loader, val_loader, fold_idx, model_name)

            # Register trained model info
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_idx}_best.pth"
            )
            trained_models.append(
                {"model_name": model_name, "fold": fold_idx, "path": ckpt_path}
            )

            # Cleanup to free memory
            del model, train_loader, val_loader, train_ds, val_ds
            torch.cuda.empty_cache()

    # 4. Validation Assessment (Hold-out Set)
    print("\n=== Performing Final Validation on Hold-out Set ===")

    # Load the official hold-out validation set
    val_ds_holdout = BirdDataset(
        Config.VAL_METADATA, phase="val", load_cached_data=True
    )
    val_loader_holdout = DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    device = Config.DEVICE
    ensemble_preds = np.zeros((len(val_ds_holdout), Config.NUM_CLASSES))
    y_true = val_ds_holdout.labels  # Ground truth from dataset

    # Run inference for all models in the ensemble
    for info in trained_models:
        model = get_model(info["model_name"])
        model.load_state_dict(torch.load(info["path"], map_location=device))
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for images, _ in val_loader_holdout:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        ensemble_preds += preds

        del model
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds /= len(trained_models)

    # Compute Final Metric
    final_metric = compute_robust_roc_auc(y_true, ensemble_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate Mean Absolute Error per sample
    errors = np.abs(y_true - ensemble_preds)
    mean_error_per_sample = errors.mean(axis=1)

    # Correlate error with "Label Count" (complexity)
    val_df = pd.read_csv(Config.VAL_METADATA)

    def get_num_labels(lbl_str):
        if pd.isna(lbl_str) or lbl_str == "?":
            return 0
        return len(str(lbl_str).split())

    val_df["num_labels"] = val_df["labels"].apply(get_num_labels)

    if val_df["num_labels"].std() > 0:
        corr = np.corrcoef(val_df["num_labels"].values, mean_error_per_sample)[0, 1]
        print(f"Correlation between Error Magnitude and Label Count: {corr}")
    else:
        print("Correlation between Error Magnitude and Label Count: N/A (No variance)")

    # 6. Submission Generation
    threshold = 0.9072993371210134

    if final_metric > threshold:
        print("\n=== Generating Submission ===")

        # Load Test Data
        test_ds = BirdDataset(Config.TEST_METADATA, phase="test", load_cached_data=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        test_preds_sum = np.zeros((len(test_ds), Config.NUM_CLASSES))

        # Ensemble Inference on Test Set
        for info in trained_models:
            model = get_model(info["model_name"])
            model.load_state_dict(torch.load(info["path"], map_location=device))
            model.to(device)
            model.eval()

            preds = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    preds.append(probs.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            test_preds_sum += preds

            del model
            torch.cuda.empty_cache()

        # Average predictions
        avg_test_preds = test_preds_sum / len(trained_models)

        # Format Submission
        test_df = pd.read_csv(Config.TEST_METADATA)
        rec_ids = test_df["rec_id"].values

        submission_rows = []
        for i, rec_id in enumerate(rec_ids):
            probs = avg_test_preds[i]
            for species_idx in range(Config.NUM_CLASSES):
                # ID format: rec_id * 100 + species_idx
                row_id = int(rec_id * 100 + species_idx)
                prob = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
