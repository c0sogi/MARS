import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_weighted_loss
from library.dataset import CervicalSpineDataset
from library.model import CervicalFractureModel
from library.engine import fit, loss_fn


def main():
    # --- 1. Setup & Configuration ---
    seed_everything(Config.SEED)
    logger = get_logger("Runfile")

    # Configure for a fast baseline run
    # We use 5 epochs to ensure completion within the 2-hour limit
    Config.setup(epochs=5, batch_size=8)

    logger.info("Initializing DataLoaders...")

    # --- 2. Data Loading ---
    train_ds = CervicalSpineDataset(phase="train")
    val_ds = CervicalSpineDataset(phase="valid")

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

    # --- 3. Model Initialization & Training ---
    logger.info("Initializing Model...")
    model = CervicalFractureModel(n_classes=Config.N_CLASSES)
    model = model.to(Config.DEVICE)

    logger.info("Starting Training...")
    fit(model, train_loader, val_loader, Config.DEVICE, epochs=Config.EPOCHS)

    # --- 4. Validation & Failure Analysis ---
    logger.info("Loading best model for validation analysis...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
    else:
        logger.warning("No model checkpoint found. Using current model state.")

    model.eval()

    # Validation Inference Loop
    val_preds = []
    val_targets = []
    val_study_ids = []

    cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    with torch.no_grad():
        for images, targets, study_ids in val_loader:
            images = images.to(Config.DEVICE, dtype=torch.float32)

            logits = model(images)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())
            val_study_ids.extend(study_ids)

    val_preds_arr = np.concatenate(val_preds, axis=0)
    val_targets_arr = np.concatenate(val_targets, axis=0)

    df_val_pred = pd.DataFrame(val_preds_arr, columns=cols)
    df_val_pred["StudyInstanceUID"] = val_study_ids

    df_val_true = pd.DataFrame(val_targets_arr, columns=cols)
    df_val_true["StudyInstanceUID"] = val_study_ids

    # Calculate Metric
    final_metric = calculate_weighted_loss(df_val_true, df_val_pred)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate per-row log loss to see correlation with targets
    epsilon = 1e-15
    y_p = np.clip(val_preds_arr, epsilon, 1.0 - epsilon)
    y_t = val_targets_arr
    # BCE per element
    bce_matrix = -(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    # Mean loss per sample
    mean_loss_per_sample = np.mean(bce_matrix, axis=1)

    # Correlation between error magnitude and patient_overall label
    # (Are fractures harder to predict?)
    corr = np.corrcoef(mean_loss_per_sample, df_val_true["patient_overall"].values)[
        0, 1
    ]
    logger.info(
        f"Correlation between Error Magnitude and Fracture Presence: {corr:.4f}"
    )

    # --- 5. Submission ---
    THRESHOLD = 0.1307335607

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_ds = CervicalSpineDataset(phase="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []
        test_study_ids = []

        with torch.no_grad():
            for images, _, study_ids in tqdm(test_loader, desc="Inference"):
                images = images.to(Config.DEVICE, dtype=torch.float32)
                logits = model(images)
                probs = torch.sigmoid(logits)

                test_preds.append(probs.cpu().numpy())
                test_study_ids.extend(study_ids)

        test_preds_arr = np.concatenate(test_preds, axis=0)

        # Create a DataFrame of predictions: StudyUID | patient_overall | C1 ... C7
        df_preds = pd.DataFrame(test_preds_arr, columns=cols)
        df_preds["StudyInstanceUID"] = test_study_ids

        # Melt to row_id format
        # We need to map [StudyUID, Column] -> row_id
        # The competition format is: row_id, fractured
        # row_id is typically StudyInstanceUID_Column

        # Load raw test.csv to ensure we match the requested row_ids
        # If test.csv is not available or doesn't have all rows, we construct row_ids manually.
        # The prompt says test.csv has metadata for prediction structure.

        submission_rows = []
        for idx, row in df_preds.iterrows():
            study_uid = row["StudyInstanceUID"]
            for col in cols:
                row_id = f"{study_uid}_{col}"
                prob = row[col]
                submission_rows.append({"row_id": row_id, "fractured": prob})

        submission_df = pd.DataFrame(submission_rows)

        # Save
        sub_path = os.path.join("working", "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path} with {len(submission_df)} rows.")

    else:
        logger.info(
            f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
