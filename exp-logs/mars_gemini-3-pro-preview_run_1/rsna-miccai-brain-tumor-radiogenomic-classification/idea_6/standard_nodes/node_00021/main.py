import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.dataset import get_dataloader
from library.model import MultiPlanarSiameseNet
from library.trainer import train_one_epoch, validate


def extract_metadata_features(df):
    """
    Extracts simple file system metadata (counts and sizes) for failure analysis.
    """
    features = []
    modalities = ["flair", "t1wce", "t2w"]  # Using the ones used in the model

    for idx, row in df.iterrows():
        feat_row = {}
        for mod in modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            count = 0
            avg_size = 0

            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                count = len(files)
                if count > 0:
                    sizes = [
                        os.path.getsize(os.path.join(full_path, f)) for f in files[:10]
                    ]
                    avg_size = np.mean(sizes)

            feat_row[f"{mod}_count"] = count
            feat_row[f"{mod}_avg_size"] = avg_size
        features.append(feat_row)

    return pd.DataFrame(features)


def predict_ensemble(model_paths, loader, device):
    """
    Generates averaged predictions from an ensemble of models.
    """
    all_preds = []

    # Iterate over each saved model state
    for path in model_paths:
        model = MultiPlanarSiameseNet(pretrained=False)  # Architecture only
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in loader:
                axial = batch["axial"].to(device)
                coronal = batch["coronal"].to(device)
                sagittal = batch["sagittal"].to(device)

                logits = model(axial, coronal, sagittal)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        all_preds.append(fold_preds)

    # Average predictions across models
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds.flatten()


def main():
    # 1. Setup
    # Override Config for fast baseline execution
    Config.MAX_EPOCHS = 10
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Running on device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Load Metadata
    df_train_all = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_holdout = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. 5-Fold Cross-Validation Training
    # We split the provided training set into 5 folds for internal CV
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    model_paths = []

    print(
        f"\nStarting {Config.N_FOLDS}-Fold Cross-Validation on {len(df_train_all)} samples..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(df_train_all, df_train_all["MGMT_value"])
    ):
        print(f"\n--- Fold {fold_idx} ---")

        # Create Fold Subsets
        df_fold_train = df_train_all.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train_all.iloc[val_idx].reset_index(drop=True)

        # Unique split names to prevent cache collisions
        train_split = f"train_fold_{fold_idx}"
        val_split = f"val_fold_{fold_idx}"

        # Initialize DataLoaders
        train_loader = get_dataloader(
            df_fold_train,
            split_name=train_split,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            load_cached_data=True,
        )
        val_loader = get_dataloader(
            df_fold_val,
            split_name=val_split,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            load_cached_data=True,
        )

        # Initialize Model & Optimizer
        model = MultiPlanarSiameseNet(pretrained=True).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_auc = 0.0
        best_model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")

        for epoch in range(Config.MAX_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold_idx} Finished. Best Internal AUC: {best_auc}")
        model_paths.append(best_model_path)

    # 4. Evaluation on Hold-out Validation Set
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    # Load holdout data
    holdout_loader = get_dataloader(
        df_holdout,
        split_name="holdout",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=True,
    )

    # Generate Predictions
    holdout_preds = predict_ensemble(model_paths, holdout_loader, device)
    holdout_targets = df_holdout["MGMT_value"].values

    # Compute Metric
    final_metric = compute_auc(holdout_targets, holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Error
    errors = np.abs(holdout_targets - holdout_preds)

    # Extract Metadata Features for Correlation
    df_features = extract_metadata_features(df_holdout)

    print("Correlation between Error Magnitude and Input Features:")
    for col in df_features.columns:
        # Handle constant columns to avoid warnings
        if df_features[col].std() == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, df_features[col])
        print(f"  {col}: {corr:.4f}")

    # 6. Submission Generation
    threshold = 0.6705454545454544

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )

        # Load Test Data
        test_loader = get_dataloader(
            df_test,
            split_name="test",
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            load_cached_data=True,
        )

        # Generate Predictions
        test_preds = predict_ensemble(model_paths, test_loader, device)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": test_preds}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
