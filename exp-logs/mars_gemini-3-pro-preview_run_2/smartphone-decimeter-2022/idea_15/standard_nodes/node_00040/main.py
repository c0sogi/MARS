import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library modules
from library.config import Config
from library.utils import setup_logger
from library.data_loader import DataProcessor, GNSSDataset
from library.model import SARTransformer
from library.trainer import Trainer
from library.inference import generate_submission, get_test_indices_and_baseline


def run_pipeline():
    # -------------------------------------------------------------------------
    # 0. Setup
    # -------------------------------------------------------------------------
    logger = setup_logger("runfile.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # Set reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Fast Baseline Configuration
    # Reducing epochs to ensure completion within time limit
    Config.EPOCHS = 5
    logger.info(f"Fast Baseline Config: EPOCHS set to {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 1. Data Preparation
    # -------------------------------------------------------------------------
    logger.info("--- Data Preparation ---")

    # 1.1 Training Data (Subsampled for speed)
    logger.info("Processing Training Data (Sample Fraction: 0.1)...")
    train_processor = DataProcessor(mode="train")
    # We use a small fraction of training data for the fast baseline
    X_kin_train, X_sky_train, y_train = train_processor.process_data(
        load_cached_data=False, sample_frac=0.1
    )

    # 1.2 Validation Data (Full set for accurate metric)
    logger.info("Processing Validation Data (Full Set)...")
    val_processor = DataProcessor(mode="validation")

    # Transfer fitted scalers from training processor to validation processor
    # This ensures validation data is scaled exactly like training data
    val_processor.scaler_kin = train_processor.scaler_kin
    val_processor.scaler_sky = train_processor.scaler_sky
    val_processor.is_fitted = True

    X_kin_val, X_sky_val, y_val = val_processor.process_data(
        load_cached_data=False, sample_frac=1.0
    )

    # 1.3 Create Datasets and Loaders
    train_dataset = GNSSDataset(X_kin_train, X_sky_train, y_train)
    val_dataset = GNSSDataset(X_kin_val, X_sky_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 2. Model Initialization & Training
    # -------------------------------------------------------------------------
    logger.info("--- Model Training ---")

    model = SARTransformer(
        kinematic_input_dim=len(Config.KINEMATIC_FEATURES),
        sky_input_dim=len(Config.SKY_FEATURES),
        output_dim=len(Config.TARGET_COLS),
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        nhead=Config.NHEAD,
        dropout=Config.DROPOUT,
    )

    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    logger.info("--- Validation Assessment ---")

    # Load best model state
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_sky_features = []  # For failure analysis

    # Run inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            x_kin, x_sky, y = batch
            x_kin = x_kin.to(device)
            x_sky_dev = x_sky.to(device)

            preds = model(x_kin, x_sky_dev)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
            all_sky_features.append(x_sky.numpy())

    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)
    sky_arr = np.vstack(all_sky_features)

    # Calculate Euclidean distance errors in meters
    # preds and targets are (dLat_m, dLon_m)
    diff = preds_arr - targets_arr
    errors = np.sqrt(np.sum(diff**2, axis=1))

    # Reconstruct mapping to compute metric per phone
    logger.info("Reconstructing validation indices for metric calculation...")
    # This function works for validation processor too because it uses the metadata path
    # configured in the processor instance (which is VAL_METADATA_PATH)
    valid_indices_map, _ = get_test_indices_and_baseline(val_processor)

    if len(errors) != len(valid_indices_map):
        logger.warning(
            f"Shape mismatch: Errors {len(errors)} vs Map {len(valid_indices_map)}. Using global approximation."
        )
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        metric = (p50 + p95) / 2
    else:
        # Group errors by tripId (which effectively groups by phone)
        trip_errors = {}
        for (trip_id, ts), err in zip(valid_indices_map, errors):
            if trip_id not in trip_errors:
                trip_errors[trip_id] = []
            trip_errors[trip_id].append(err)

        # Compute metric: mean of (mean of 50th and 95th per phone)
        phone_scores = []
        for trip_id, errs in trip_errors.items():
            p50 = np.percentile(errs, 50)
            p95 = np.percentile(errs, 95)
            score = (p50 + p95) / 2
            phone_scores.append(score)

        metric = np.mean(phone_scores)

    print(f"Final Validation Metric: {metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("--- Failure Analysis ---")

    # Calculate correlation between error magnitude and sky features
    # sky_arr is scaled, but correlation is scale-invariant
    feature_names = Config.SKY_FEATURES
    correlations = {}

    for i, name in enumerate(feature_names):
        if i < sky_arr.shape[1]:
            feat_values = sky_arr[:, i]
            # Handle potential constant values (std=0)
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations[name] = corr
            else:
                correlations[name] = 0.0

    print("Correlation between Error Magnitude and Environmental Features:")
    for name, corr in sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 4.256982128481356
    if metric < threshold:
        logger.info(
            f"Validation metric {metric} is below threshold {threshold}. Generating submission..."
        )
        generate_submission(load_cached_data=False)
    else:
        logger.info(
            f"Validation metric {metric} is NOT below threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
