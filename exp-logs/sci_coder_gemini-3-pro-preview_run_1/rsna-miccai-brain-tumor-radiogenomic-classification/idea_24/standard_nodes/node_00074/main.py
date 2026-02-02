import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_device, setup_logger
from library.data_processing import get_centroids_with_caching
from library.dataset import BraTSDataset, get_transforms
from library.model import CAWIVModel
from library.trainer import train_one_epoch, validate, predict


def extract_metadata_features(df, input_dir):
    """
    Extracts file counts for failure analysis.
    """
    features = []
    modalities = ["flair", "t1wce", "t2w"]

    for _, row in df.iterrows():
        feat = {}
        for mod in modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(input_dir, rel_path)
            if os.path.exists(full_path):
                # Fast count
                count = len(
                    [name for name in os.listdir(full_path) if name.endswith(".dcm")]
                )
            else:
                count = 0
            feat[f"{mod}_count"] = count
        features.append(feat)
    return pd.DataFrame(features)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = setup_logger()  # Logs to stdout

    # Fast Baseline Configuration overrides
    Config.NUM_EPOCHS = 10  # Reduced for speed
    Config.BATCH_SIZE = 32

    logger.info("Starting Runfile Execution...")
    logger.info(f"Device: {device}")

    # 2. Load Metadata
    train_meta_path = Config.TRAIN_METADATA
    val_meta_path = Config.VAL_METADATA

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        logger.error("Metadata files not found.")
        return

    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)

    logger.info(f"Train size: {len(df_train)}, Val size: {len(df_val)}")

    # 3. Prepare Centroids
    logger.info("Loading/Computing Centroids...")
    centroids_train = get_centroids_with_caching(
        df_train, Config.INPUT_DIR, cache_name="centroids_train", load_cached_data=True
    )
    centroids_val = get_centroids_with_caching(
        df_val, Config.INPUT_DIR, cache_name="centroids_val", load_cached_data=True
    )

    # 4. Datasets & Loaders
    train_ds = BraTSDataset(
        df_train,
        centroids_train,
        Config.INPUT_DIR,
        transform=get_transforms(mode="train"),
        mode="train",
    )
    val_ds = BraTSDataset(
        df_val,
        centroids_val,
        Config.INPUT_DIR,
        transform=get_transforms(mode="val"),
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Initialization
    model = CAWIVModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )

    # 6. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_runfile_model.pth")

    logger.info("Starting Training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 7. Final Evaluation on Hold-out Set
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get predictions on validation set
    # predict returns (ids, probs) - but BraTSDataset in 'val' mode returns (image, target)
    # library.trainer.predict handles (image, identifiers)
    # We need to use the validation loader which yields (images, targets)
    # predict() handles this by returning targets as IDs
    val_targets, val_preds = predict(model, val_loader, device)

    final_metric = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    logger.info("Performing Failure Analysis...")
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    errors = np.abs(val_targets - val_preds)

    # Extract features
    meta_features = extract_metadata_features(df_val, Config.INPUT_DIR)

    # Calculate correlations
    logger.info("Correlation between Error Magnitude and Metadata Features:")
    for col in meta_features.columns:
        if meta_features[col].std() > 0:
            corr, _ = pearsonr(errors, meta_features[col])
            print(f"Feature: {col}, Correlation with Error: {corr:.4f}")
        else:
            print(f"Feature: {col}, Correlation: N/A (Constant)")

    # 9. Submission
    THRESHOLD = 0.6705454545454544
    if final_metric > THRESHOLD:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_meta_path = Config.TEST_METADATA
        if os.path.exists(test_meta_path):
            df_test = pd.read_csv(test_meta_path)

            # Centroids for test
            centroids_test = get_centroids_with_caching(
                df_test,
                Config.INPUT_DIR,
                cache_name="centroids_test",
                load_cached_data=True,
            )

            test_ds = BraTSDataset(
                df_test,
                centroids_test,
                Config.INPUT_DIR,
                transform=get_transforms(mode="val"),
                mode="test",
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            test_ids, test_probs = predict(model, test_loader, device)

            submission_df = pd.DataFrame(
                {"BraTS21ID": test_ids, "MGMT_value": test_probs}
            )

            # Format check
            submission_df = submission_df[["BraTS21ID", "MGMT_value"]]
            submission_df.sort_values("BraTS21ID", inplace=True)

            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        else:
            logger.error("Test metadata not found.")
    else:
        logger.info(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
