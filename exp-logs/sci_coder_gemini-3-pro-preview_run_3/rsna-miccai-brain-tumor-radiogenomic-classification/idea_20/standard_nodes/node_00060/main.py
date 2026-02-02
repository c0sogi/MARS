import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, log_message, get_device
from library.data_loader import get_dataloaders
from library.model import MGSHDNetwork
from library.train import run_training
from library.predict import run_inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override epochs for a fast baseline execution
    Config.EPOCHS = 10

    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    log_message("Configuration setup complete. Starting pipeline...")

    # ==========================================
    # 2. Training
    # ==========================================
    # Run the training loop (handles training, validation, and saving best model)
    # Using patience=3 for early stopping to ensure speed
    run_training(load_cached_data=True, patience=3)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    log_message("Loading best model for validation assessment...")

    # Initialize model and load best weights
    model = MGSHDNetwork().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        log_message("Error: Model checkpoint not found. Validation may be invalid.")

    model.eval()

    # Get DataLoaders
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    if val_loader is None:
        log_message("Error: Validation loader is None.")
        return

    all_targets = []
    all_probs = []
    all_ids = []

    # Inference on Validation Set
    with torch.no_grad():
        for inputs, targets, ids in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Collect results
            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_ids.extend(ids)

    # Calculate Metric
    if len(np.unique(all_targets)) > 1:
        val_auc = roc_auc_score(all_targets, all_probs)
    else:
        val_auc = 0.5

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    log_message("Performing Failure Analysis...")

    # Load validation metadata to get features
    if os.path.exists(Config.VAL_META_PATH):
        val_df = pd.read_parquet(Config.VAL_META_PATH)

        # Calculate Error Magnitude
        errors = np.abs(np.array(all_targets) - np.array(all_probs))

        # Map IDs to Errors
        id_to_error = {str(pid): err for pid, err in zip(all_ids, errors)}

        analysis_data = []
        for idx, row in val_df.iterrows():
            pid = str(row["BraTS21ID"])
            if pid in id_to_error:
                # Extract Meta-features
                flair_len = len(row.get("flair_paths", []))
                t1w_len = len(row.get("t1w_paths", []))
                t1wce_len = len(row.get("t1wce_paths", []))
                t2w_len = len(row.get("t2w_paths", []))

                analysis_data.append(
                    {
                        "error": id_to_error[pid],
                        "flair_count": flair_len,
                        "t1w_count": t1w_len,
                        "t1wce_count": t1wce_len,
                        "t2w_count": t2w_len,
                    }
                )

        # Calculate Correlations
        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            # Drop columns with 0 variance to avoid NaN correlation
            analysis_df = analysis_df.loc[:, analysis_df.std() > 0]

            if "error" in analysis_df.columns:
                correlations = analysis_df.corr()["error"].drop(
                    "error", errors="ignore"
                )
                log_message("Correlation between Error Magnitude and Input Features:")
                print(correlations)
            else:
                log_message("Could not calculate correlations (insufficient variance).")
    else:
        log_message("Validation metadata not found. Skipping failure analysis.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        log_message(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference(load_cached_data=True)
    else:
        log_message(
            f"Validation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
