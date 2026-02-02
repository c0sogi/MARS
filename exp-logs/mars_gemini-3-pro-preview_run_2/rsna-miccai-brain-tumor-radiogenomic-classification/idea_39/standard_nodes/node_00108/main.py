import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, data_loader, model, trainer

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
config.NUM_EPOCHS = 5
config.DEBUG_SAMPLE_SIZE = None  # Use full dataset (small enough for <2h)
config.BATCH_SIZE = 32


def main():
    # 1. Setup
    utils.seed_everything()
    logger = utils.get_logger("RUNFILE")
    logger.info("Starting Dual-Scale Asymmetric EfficientNet Ensemble Pipeline")

    # 2. Data Loading
    # Returns: train_ds_A, train_ds_B, val_ds_A, val_ds_B
    # A = Texture (Stride 2), B = Context (Stride 5)
    logger.info("Loading datasets...")
    train_ds_A, train_ds_B, val_ds_A, val_ds_B = data_loader.get_datasets(
        load_cache=True
    )

    # 3. Train Texture Expert (Model A)
    logger.info(">>> Training Phase A: Texture Expert (Stride 2)")
    path_A = trainer.run_training_phase(train_ds_A, val_ds_A, "model_A_texture")

    # 4. Train Context Expert (Model B)
    logger.info(">>> Training Phase B: Context Expert (Stride 5)")
    path_B = trainer.run_training_phase(train_ds_B, val_ds_B, "model_B_context")

    # 5. Validation & Metric Calculation
    logger.info(">>> Performing Final Validation Ensemble")

    # Load Best Models
    net_A = model.AsymmetricEfficientNet().to(config.DEVICE)
    net_A.load_state_dict(torch.load(path_A, map_location=config.DEVICE))
    net_A.eval()

    net_B = model.AsymmetricEfficientNet().to(config.DEVICE)
    net_B.load_state_dict(torch.load(path_B, map_location=config.DEVICE))
    net_B.eval()

    # Create Validation Loaders
    val_loader_A = DataLoader(
        val_ds_A,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    val_loader_B = DataLoader(
        val_ds_B,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Helper for inference
    def get_preds(model_inst, loader):
        preds = []
        targets = []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(config.DEVICE)
                # Forward pass
                out = model_inst(data)
                prob = torch.sigmoid(out)

                preds.extend(prob.cpu().numpy().flatten())
                targets.extend(target.numpy().flatten())
        return np.array(preds), np.array(targets)

    # Get predictions
    preds_A, targets_A = get_preds(net_A, val_loader_A)
    preds_B, targets_B = get_preds(net_B, val_loader_B)

    # Ensemble (Average)
    final_val_preds = (preds_A + preds_B) / 2.0

    # Calculate Metric
    # Targets A and B should be identical, use A
    final_val_metric = roc_auc_score(targets_A, final_val_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    logger.info(">>> Performing Failure Analysis")

    # Calculate absolute error
    errors = np.abs(targets_A - final_val_preds)

    # Load metadata to extract features
    df_val = pd.read_csv(config.VAL_METADATA)

    # Extract 'slice_count' (FLAIR depth) as a proxy for tumor/scan volume
    slice_counts = []
    for idx, row in df_val.iterrows():
        flair_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Quick count of files
            cnt = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
        except Exception:
            cnt = 0
        slice_counts.append(cnt)

    df_val["slice_count"] = slice_counts
    df_val["error"] = errors

    # Calculate correlation
    if df_val["slice_count"].std() > 0:
        corr = df_val["slice_count"].corr(df_val["error"])
    else:
        corr = 0.0

    print(f"Correlation between Error and Slice Count: {corr}")

    # 7. Submission
    threshold = 0.6321818181818182
    if final_val_metric > threshold:
        logger.info(
            f"Metric ({final_val_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        test_ds_A, test_ds_B = data_loader.get_test_datasets(load_cache=True)
        test_ids = data_loader.get_test_ids()

        test_loader_A = DataLoader(
            test_ds_A,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        test_loader_B = DataLoader(
            test_ds_B,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        # Predict with TTA (Test-Time Augmentation)
        logger.info("Predicting Test Set A (Texture)...")
        preds_test_A = trainer.predict_with_tta(net_A, test_loader_A, config.DEVICE)

        logger.info("Predicting Test Set B (Context)...")
        preds_test_B = trainer.predict_with_tta(net_B, test_loader_B, config.DEVICE)

        # Ensemble
        final_test_preds = (preds_test_A + preds_test_B) / 2.0

        # Create Submission
        submission_df = pd.DataFrame(
            {"BraTS21ID": test_ids, "MGMT_value": final_test_preds}
        )

        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Metric ({final_val_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
