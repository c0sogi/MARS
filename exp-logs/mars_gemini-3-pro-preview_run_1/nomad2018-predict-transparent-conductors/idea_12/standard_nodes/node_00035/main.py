import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_processing import DataHandler
from library.model import PAWDS, collate_fn
from library.train import Trainer
from library.predict import generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    # Override number of epochs for a fast baseline execution as per requirements
    Config.NUM_EPOCHS = 30

    # Initialize configuration and seeding
    Config.setup()
    seed_everything(Config.SEED)

    # Setup logging and device
    logger = get_logger("runfile")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    logger.info("Initializing DataHandler and preparing datasets...")
    data_handler = DataHandler()

    # This will load from cache if available, or compute and cache features
    # It also fits the scalers on the training data
    train_dataset, val_dataset, test_dataset = data_handler.get_datasets()

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer if using GPU
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Training
    # -------------------------------------------------------------------------
    logger.info("Initializing PA-WDS Model...")
    model = PAWDS().to(device)

    logger.info("Starting Training...")
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS)

    # -------------------------------------------------------------------------
    # 4. Validation Assessment
    # -------------------------------------------------------------------------
    logger.info("Loading best model for validation assessment...")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        logger.error(f"Model file not found at {Config.MODEL_SAVE_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    val_preds_log = []
    val_targets_log = []
    val_global_feats = []

    logger.info("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            atomic_x = batch["atomic_features"].to(device)
            global_x = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_x, global_x, mask)

            val_preds_log.append(outputs.cpu().numpy())
            val_targets_log.append(targets.cpu().numpy())
            val_global_feats.append(batch["global_features"].cpu().numpy())

    val_preds_log = np.vstack(val_preds_log)
    val_targets_log = np.vstack(val_targets_log)
    val_global_feats = np.vstack(val_global_feats)

    # Metric Calculation: Column-wise Root Mean Squared Logarithmic Error
    # Note: The model output and targets are already in log(1+x) space.
    # So we calculate RMSE on these values directly.
    mse_per_col = np.mean((val_preds_log - val_targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate error magnitude (Mean Absolute Error in log space)
    # This represents the deviation ratio in original space
    errors = np.mean(np.abs(val_preds_log - val_targets_log), axis=1)

    # Global feature names based on DataHandler logic:
    # 3 Lattice Len, 3 Lattice Ang, 1 Vol, 1 Dens, 3 Comp (Al, Ga, In)
    feature_names = [
        "Lattice_A",
        "Lattice_B",
        "Lattice_C",
        "Angle_Alpha",
        "Angle_Beta",
        "Angle_Gamma",
        "Volume",
        "Density",
        "Comp_Al",
        "Comp_Ga",
        "Comp_In",
    ]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(val_global_feats, columns=feature_names)
    analysis_df["Error"] = errors

    # Calculate correlation
    correlations = (
        analysis_df.corr()["Error"].drop("Error").sort_values(key=abs, ascending=False)
    )

    print("\nCorrelation between Input Features and Prediction Error:")
    print(correlations)
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device=device.type,
        )
    else:
        logger.warning(
            f"Validation metric {final_metric} is NOT lower than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
