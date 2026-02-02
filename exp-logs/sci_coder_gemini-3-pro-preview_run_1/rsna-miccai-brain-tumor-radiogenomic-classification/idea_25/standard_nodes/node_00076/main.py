import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import provided library modules
from library import config, utils, dataset, model, train_eval


def extract_metadata_features(df):
    """
    Extracts file counts as proxy features for failure analysis.
    Iterates through subject directories to count files in each modality.
    """
    features = []
    modalities = ["FLAIR", "T1wCE", "T2w"]

    for _, row in df.iterrows():
        feat = {}
        for mod in modalities:
            # Construct path based on metadata convention
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(config.INPUT_DIR, rel_path)

            count = 0
            if os.path.exists(full_path):
                try:
                    # Fast count of files
                    count = len(
                        [
                            name
                            for name in os.listdir(full_path)
                            if os.path.isfile(os.path.join(full_path, name))
                        ]
                    )
                except OSError:
                    count = 0

            feat[f"{mod}_count"] = count
        features.append(feat)

    return pd.DataFrame(features)


def main():
    # 1. Setup and Initialization
    utils.seed_everything(config.SEED)
    logger = utils.get_logger("runfile")
    device = utils.get_device()

    logger.info("Starting C-AA-WIV Pipeline Execution...")

    # 2. Load Metadata and Perform Integrity Check
    if not os.path.exists(config.TRAIN_METADATA_PATH) or not os.path.exists(
        config.VAL_METADATA_PATH
    ):
        logger.error(
            "Metadata files missing. Ensure metadata generation was successful."
        )
        sys.exit(1)

    df_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(config.VAL_METADATA_PATH)

    # Combine train and val for Stratified K-Fold
    full_df = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Integrity Check: Ensure we have the expected number of subjects
    # The full training set (excluding 3 bad cases) is ~523 subjects.
    dataset_size = len(full_df)
    logger.info(f"Loaded dataset with {dataset_size} subjects.")

    if dataset_size < 500 and not config.DEBUG:
        logger.warning(
            f"Dataset size ({dataset_size}) is significantly smaller than expected (~523)."
        )

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    X = full_df.drop(columns=["MGMT_value"])
    y = full_df["MGMT_value"]

    # Arrays to store Out-Of-Fold predictions and targets
    oof_preds = np.zeros(dataset_size)
    oof_targets = np.zeros(dataset_size)

    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"=== Running Fold {fold_idx}/{config.NUM_FOLDS - 1} ===")

        train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_df.iloc[val_idx].reset_index(drop=True)

        # A. Train the model for this fold
        # run_fold handles training, validation monitoring, and saving the best model
        best_auc, model_path = train_eval.run_fold(fold_idx, train_fold_df, val_fold_df)

        # B. Generate OOF Predictions
        # Load the best saved model for this fold
        net = model.EfficientNet9Channel(
            backbone_name=config.BACKBONE, pretrained=False, num_classes=1
        )
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()

        # Create validation loader (reusing cache if available)
        val_loader = dataset.get_dataloader(
            val_fold_df,
            phase=f"fold_{fold_idx}_val",
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            load_cached_data=True,
        )

        fold_probs = []
        fold_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)

                # Forward pass
                logits = net(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                fold_probs.extend(probs)
                fold_targets_list.extend(targets.cpu().numpy())

        # Store predictions in the global OOF array
        oof_preds[val_idx] = np.array(fold_probs)
        oof_targets[val_idx] = np.array(fold_targets_list)

        # Clean up to save memory
        del net, val_loader
        torch.cuda.empty_cache()

    # 4. Final Validation Metric
    final_auc = roc_auc_score(oof_targets, oof_preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis on OOF predictions...")

    # Compute absolute error
    errors = np.abs(oof_targets - oof_preds)

    # Extract metadata features (file counts)
    logger.info("Extracting metadata features for correlation analysis...")
    meta_features_df = extract_metadata_features(full_df)

    # Calculate correlations
    print("Failure Analysis - Correlation with Error Magnitude:")
    for col in meta_features_df.columns:
        feature_values = meta_features_df[col]
        if feature_values.std() > 0:
            corr, _ = pearsonr(feature_values, errors)
            print(f"{col}: {corr:.6f}")
        else:
            print(f"{col}: NaN (Constant feature)")

    # 6. Submission Generation
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        logger.info(
            f"Metric ({final_auc:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        train_eval.generate_submission()
    else:
        logger.warning(
            f"Metric ({final_auc:.6f}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
