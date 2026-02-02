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
# Configuration Overrides
# -----------------------------------------------------------------------------
# config.NUM_EPOCHS = 10 (Default in config.py is 10, which is good)
config.DEBUG_SAMPLE_SIZE = None  # Use full dataset
config.BATCH_SIZE = 32


def main():
    # 1. Setup
    utils.seed_everything()
    logger = utils.get_logger("RUNFILE")
    logger.info("Starting Single-Model Asymmetric EfficientNet Pipeline")

    # 2. Data Loading
    logger.info("Loading datasets...")
    train_ds, val_ds = data_loader.get_datasets(load_cache=True)

    # 3. Train Model
    logger.info(">>> Training Model")
    best_model_path = trainer.run_training_phase(train_ds, val_ds, "model_best")

    # 4. Validation & Metric Calculation
    logger.info(">>> Performing Final Validation")

    # Load Best Model
    net = model.AsymmetricEfficientNet().to(config.DEVICE)
    net.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    net.eval()

    # Create Validation Loader
    val_loader = DataLoader(
        val_ds,
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
    final_val_preds, targets = get_preds(net, val_loader)

    # Calculate Metric
    final_val_metric = roc_auc_score(targets, final_val_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_val_metric}")

    # 5. Failure Analysis
    logger.info(">>> Performing Failure Analysis")

    # Calculate absolute error
    errors = np.abs(targets - final_val_preds)

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

    # 6. Submission
    threshold = 0.6321818181818182
    if final_val_metric > threshold:
        logger.info(
            f"Metric ({final_val_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        test_ds = data_loader.get_test_datasets(load_cache=True)
        test_ids = data_loader.get_test_ids()

        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        # Predict with TTA (Test-Time Augmentation)
        logger.info("Predicting Test Set...")
        final_test_preds = trainer.predict_with_tta(net, test_loader, config.DEVICE)

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
