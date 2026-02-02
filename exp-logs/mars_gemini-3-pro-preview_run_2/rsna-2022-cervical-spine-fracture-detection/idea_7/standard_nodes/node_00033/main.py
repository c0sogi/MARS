import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import gc

# Suppress warnings and progress bars for clean output
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_logger, competition_metric
from library.data import get_loaders
from library.model import CervicalSpineModel
from library.engine import fit, predict

# Setup Logger
logger = get_logger("runfile")


def analyze_failures(model, loader, device, metadata_path):
    """
    Performs inference on validation set, calculates metric, and analyzes errors.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_uids = []

    # Inference Loop
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, dtype=torch.float32)
            study_targets = batch["study_targets"].to(device, dtype=torch.float32)
            uids = batch["row_id"]

            # Forward pass
            study_logits, _ = model(images)
            study_probs = torch.sigmoid(study_logits)

            all_preds.append(study_probs.cpu().numpy())
            all_targets.append(study_targets.cpu().numpy())
            all_uids.extend(uids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 1. Calculate Final Metric
    final_metric = competition_metric(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    # Calculate weighted log loss per study
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    # Get weights from Config and convert to numpy
    weights = Config.LOSS_WEIGHTS
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    # Compute loss matrix: (N, 8)
    # L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    bce = all_targets * np.log(preds_clipped) + (1 - all_targets) * np.log(
        1 - preds_clipped
    )
    weighted_loss = -weights * bce

    # Average loss across columns to get a single error scalar per study
    study_errors = np.mean(weighted_loss, axis=1)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"StudyInstanceUID": all_uids, "error": study_errors})

    # Merge with metadata to get ground truth features
    if os.path.exists(metadata_path):
        val_meta = pd.read_csv(metadata_path)
        analysis_df = analysis_df.merge(val_meta, on="StudyInstanceUID", how="left")

        # Calculate correlations
        # We check if error correlates with having a fracture (patient_overall)
        # or the number of fractures (complexity)
        if "patient_overall" in analysis_df.columns:
            # Calculate fracture count if columns exist
            frac_cols = [f"C{i}" for i in range(1, 8)]
            if all(col in analysis_df.columns for col in frac_cols):
                analysis_df["fracture_count"] = analysis_df[frac_cols].sum(axis=1)

            print("Failure Analysis - Correlations with Error:")

            # Correlation with patient_overall
            corr_overall = analysis_df["error"].corr(analysis_df["patient_overall"])
            print(f"  patient_overall: {corr_overall:.4f}")

            # Correlation with fracture_count
            if "fracture_count" in analysis_df.columns:
                corr_count = analysis_df["error"].corr(analysis_df["fracture_count"])
                print(f"  fracture_count: {corr_count:.4f}")

    return final_metric


def main():
    # --- 1. Configuration Overrides ---
    # Override submission path to match requirements
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Limit epochs for fast baseline execution
    Config.EPOCHS = 5

    # --- 2. Initialization ---
    seed_everything(Config.SEED)

    # --- 3. Data Loading ---
    logger.info("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # --- 4. Model Setup ---
    logger.info("Initializing Model...")
    model = CervicalSpineModel()

    # --- 5. Training ---
    logger.info("Starting Training...")
    fit(model, train_loader, val_loader, epochs=Config.EPOCHS, device=Config.DEVICE)

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # --- 6. Validation & Analysis ---
    logger.info("Starting Validation and Failure Analysis...")

    # Load best model weights
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)

    final_metric = analyze_failures(
        model, val_loader, Config.DEVICE, Config.VAL_METADATA_PATH
    )

    # --- 7. Submission Generation ---
    # Threshold defined in task
    THRESHOLD = 0.15364714496434773

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        predict(model, test_loader, device=Config.DEVICE)
    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
