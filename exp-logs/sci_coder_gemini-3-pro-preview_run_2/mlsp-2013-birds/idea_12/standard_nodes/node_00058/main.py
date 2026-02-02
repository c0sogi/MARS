import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import importlib
import library.config

# Force reload to ensure Config updates are picked up in persistent environment
importlib.reload(library.config)

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_device, worker_init_fn
from library.data import BirdDataset, get_transforms, prepare_folds
from library.models import get_model
from library.engine import train_fold


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # 2. Data Preparation
    # Load folds data (cached or generated)
    # This merges train and val metadata and creates stratified folds
    df_folds = prepare_folds(load_cached_data=True)

    # Load Test Data for final inference
    df_test = pd.read_csv(Config.TEST_CSV)

    # Placeholders for OOF (Out-Of-Fold) and Test Predictions
    # OOF: Dictionary to store predictions for each recording ID
    oof_preds_dict = {}
    oof_targets_dict = {}

    # Test: Accumulator for averaging predictions across all models
    test_preds_sum = np.zeros((len(df_test), Config.NUM_CLASSES))

    # Define the Heterogeneous Ensemble Configuration
    models_config = [
        {"name": Config.MODEL_A_NAME, "size": Config.RESNET_INPUT_SIZE},  # ResNet18
        {
            "name": Config.MODEL_B_NAME,  # DenseNet121
            "size": Config.DENSENET_INPUT_SIZE,
        },
        {
            "name": Config.MODEL_C_NAME,  # EfficientNet-B0
            "size": Config.EFFICIENTNET_INPUT_SIZE,
        },
    ]

    # 3. Training Loop (5 Folds x 3 Models)
    for fold_idx in range(Config.NUM_FOLDS):
        # Split Data into Train and Validation for the current fold
        train_df = df_folds[df_folds["fold"] != fold_idx].reset_index(drop=True)
        val_df = df_folds[df_folds["fold"] == fold_idx].reset_index(drop=True)

        # Store targets for OOF evaluation
        label_cols = [c for c in val_df.columns if c.startswith("species_")]
        for _, row in val_df.iterrows():
            rec_id = row["rec_id"]
            oof_targets_dict[rec_id] = row[label_cols].values.astype(float)
            if rec_id not in oof_preds_dict:
                oof_preds_dict[rec_id] = []

        # Iterate over the heterogeneous models
        for model_cfg in models_config:
            model_name = model_cfg["name"]
            input_size = model_cfg["size"]

            print(
                f"\n--- Processing Fold {fold_idx}, Model {model_name} (Input: {input_size}) ---"
            )

            # Create Datasets & Loaders with resolution-specific transforms
            train_dataset = BirdDataset(
                train_df,
                transform=get_transforms(input_size[0], input_size[1], phase="train"),
                phase="train",
            )
            val_dataset = BirdDataset(
                val_df,
                transform=get_transforms(input_size[0], input_size[1], phase="valid"),
                phase="valid",
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                worker_init_fn=worker_init_fn,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                worker_init_fn=worker_init_fn,
            )

            # Train the model (includes SWA logic)
            # The function saves the best SWA model to disk
            _ = train_fold(fold_idx, model_name, train_loader, val_loader, device)

            # --- Inference Phase ---

            # Load the best saved model
            model = get_model(model_name, pretrained=False, device=device)
            save_path = os.path.join(
                Config.WORK_DIR, f"model_fold{fold_idx}_{model_name}.pth"
            )
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            # 1. Inference on Validation Fold (for OOF Metric)
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    rec_ids = batch["rec_id"].numpy()

                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()

                    for rid, p in zip(rec_ids, probs):
                        oof_preds_dict[rid].append(p)

            # 2. Inference on Test Set (Accumulate for Submission)
            test_dataset = BirdDataset(
                df_test,
                transform=get_transforms(input_size[0], input_size[1], phase="test"),
                phase="test",
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                worker_init_fn=worker_init_fn,
            )

            fold_model_test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["image"].to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    fold_model_test_preds.append(probs)

            if fold_model_test_preds:
                # Add to global accumulator
                test_preds_sum += np.concatenate(fold_model_test_preds, axis=0)

    # 4. Aggregate OOF Predictions & Calculate Metric
    # oof_preds_dict contains lists of predictions (one from each model type) for each rec_id
    # We average them (Bagging).

    oof_rec_ids = sorted(oof_preds_dict.keys())
    y_true = np.array([oof_targets_dict[rid] for rid in oof_rec_ids])

    # Average predictions for each sample across all models
    y_pred = np.array([np.mean(oof_preds_dict[rid], axis=0) for rid in oof_rec_ids])

    # Compute Macro ROC AUC
    try:
        final_auc = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        final_auc = 0.5

    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    # Calculate Mean Absolute Error per sample
    errors = np.abs(y_true - y_pred).mean(axis=1)

    # Feature: Label Cardinality (Number of active species)
    cardinality = y_true.sum(axis=1)

    # Correlation between Error and Cardinality
    if np.std(errors) > 0 and np.std(cardinality) > 0:
        corr = np.corrcoef(errors, cardinality)[0, 1]
    else:
        corr = 0.0

    print(
        f"Failure Analysis: Correlation between Error and Label Cardinality: {corr:.4f}"
    )

    # 6. Submission
    threshold = 0.9129501920716607

    if final_auc > threshold:
        print("Metric above threshold. Generating submission...")

        # Average test predictions
        # We summed over (Num_Folds * Num_Models) iterations
        num_ensemble_members = Config.NUM_FOLDS * len(models_config)
        avg_test_preds = test_preds_sum / num_ensemble_members

        # Format submission
        # "Id" column is rec_id * 100 + species_id
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for idx, rec_id in enumerate(rec_ids):
            probs = avg_test_preds[idx]
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_auc} not above threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
