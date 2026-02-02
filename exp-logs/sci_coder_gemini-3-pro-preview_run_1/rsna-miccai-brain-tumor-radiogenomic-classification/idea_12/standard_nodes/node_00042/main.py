import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.data import generate_instance_metadata, get_dataloader
from library.model import SILNet
from library.engine import train_model, predict


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Load Metadata
    print("Loading metadata...")
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Generate Instance-Level Data
    # This expands each subject into 3 instances (offsets -2, 0, +2)
    print("Generating instance-level data...")
    df_train_instances = generate_instance_metadata(
        df_train_meta, "train", load_cached_data=True
    )
    df_val_instances = generate_instance_metadata(
        df_val_meta, "val", load_cached_data=True
    )
    df_test_instances = generate_instance_metadata(
        df_test_meta, "test", load_cached_data=True
    )

    # Prepare storage for ensemble predictions
    # We will accumulate probabilities from each fold
    val_preds_accum = np.zeros(len(df_val_instances))
    test_preds_accum = np.zeros(len(df_test_instances))

    # 4. 5-Fold Group Cross-Validation
    # We split the training instances based on BraTS21ID to ensure no subject leakage
    gkf = GroupKFold(n_splits=Config.N_FOLDS)

    # We need groups for the splitter
    groups = df_train_instances["BraTS21ID"].values

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, valid_idx) in enumerate(
        gkf.split(df_train_instances, groups=groups)
    ):
        print(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Split data
        fold_train_df = df_train_instances.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = df_train_instances.iloc[valid_idx].reset_index(drop=True)

        # Create DataLoaders
        train_loader = get_dataloader(fold_train_df, "train")
        valid_loader = get_dataloader(fold_valid_df, "val")  # Internal CV validation

        # Initialize Model
        model = SILNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train
        # Note: train_model saves best_model.pth to working dir.
        # We let it overwrite for each fold as we use the returned model object immediately.
        model = train_model(
            model,
            train_loader,
            valid_loader,
            optimizer,
            device,
            num_epochs=Config.NUM_EPOCHS,
            patience=5,
        )

        # Inference on Hold-out Validation Set
        # We use a custom loop here because engine.predict aggregates by mean,
        # but we want instance-level probs for ensembling first.
        print("Predicting on hold-out validation set...")
        val_loader_full = get_dataloader(
            df_val_instances, "val", batch_size=Config.BATCH_SIZE * 2
        )
        model.eval()
        fold_val_probs = []
        with torch.no_grad():
            for images, _ in val_loader_full:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_val_probs.extend(probs)
        val_preds_accum += np.array(fold_val_probs)

        # Inference on Test Set
        print("Predicting on test set...")
        test_loader_full = get_dataloader(
            df_test_instances, "test", batch_size=Config.BATCH_SIZE * 2
        )
        fold_test_probs = []
        with torch.no_grad():
            for images, _ in test_loader_full:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_test_probs.extend(probs)
        test_preds_accum += np.array(fold_test_probs)

        # Clean up to save memory
        del model, optimizer, train_loader, valid_loader
        torch.cuda.empty_cache()

    # 5. Averaging and Aggregation
    print("\nAggregating predictions...")

    # Average over folds
    val_preds_avg = val_preds_accum / Config.N_FOLDS
    test_preds_avg = test_preds_accum / Config.N_FOLDS

    # Add predictions back to the instance dataframes
    df_val_instances["prob"] = val_preds_avg
    df_test_instances["prob"] = test_preds_avg

    # Aggregate to Subject Level (Mean of the 3 instances per subject)
    val_subject_preds = (
        df_val_instances.groupby("BraTS21ID")["prob"].mean().reset_index()
    )
    test_subject_preds = (
        df_test_instances.groupby("BraTS21ID")["prob"].mean().reset_index()
    )

    # Merge with ground truth for validation
    val_results = pd.merge(
        val_subject_preds, df_val_meta[["BraTS21ID", "MGMT_value"]], on="BraTS21ID"
    )

    # 6. Validation Metrics
    final_auc = roc_auc_score(val_results["MGMT_value"], val_results["prob"])
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate subject-level error
    val_results["error"] = (val_results["prob"] - val_results["MGMT_value"]).abs()

    # Correlation with Target
    corr_target = val_results["error"].corr(val_results["MGMT_value"])
    print(f"Correlation between Error and Target Class (MGMT_value): {corr_target:.4f}")

    # Correlation with Instance Offset (using instance-level errors)
    # We assign the target to the instance and calc error
    df_val_instances["target"] = df_val_instances["BraTS21ID"].map(
        df_val_meta.set_index("BraTS21ID")["MGMT_value"]
    )
    df_val_instances["instance_error"] = (
        df_val_instances["prob"] - df_val_instances["target"]
    ).abs()
    corr_offset = df_val_instances["instance_error"].corr(
        df_val_instances["instance_offset"]
    )
    print(f"Correlation between Error and Instance Offset: {corr_offset:.4f}")

    # 8. Submission
    threshold = 0.6705454545454544
    if final_auc > threshold:
        print(
            f"\nValidation metric {final_auc} > {threshold}. Generating submission..."
        )
        submission_df = test_subject_preds.rename(columns={"prob": "MGMT_value"})
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {final_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
