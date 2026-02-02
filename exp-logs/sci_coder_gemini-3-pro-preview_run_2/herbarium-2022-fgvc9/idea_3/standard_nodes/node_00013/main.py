import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.train import Trainer, validate
from library.inference import predict_and_submit
from library.dataset import process_taxonomy

# Initialize Logger
logger = get_logger("runfile")

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# We limit the epochs to ensure execution within the 2-hour limit.
# 532k images ~ 25 mins/epoch on A100. 2 epochs ~ 50 mins.
Config.NUM_EPOCHS = 2

# Increase batch size to maximize A100 utilization
Config.BATCH_SIZE = 64
Config.GRAD_ACCUM_STEPS = 2  # Effective batch size = 128

# Ensure GPU usage
Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model failures on the validation set.
    Calculates correlation between error and:
    1. Training Class Frequency
    2. Image File Size
    """
    logger.info("Starting Failure Analysis...")
    model.eval()

    preds_list = []
    targets_list = []

    # Collect predictions
    from torch.cuda.amp import autocast

    with torch.no_grad():
        for images, species_labels, _, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with autocast():
                # species_label=None -> returns scaled cosine similarities
                outputs = model(images, species_label=None)
                preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()
                preds_list.extend(preds)
                targets_list.extend(species_labels.numpy())

    preds_arr = np.array(preds_list)
    targets_arr = np.array(targets_list)

    # Calculate Error (0 = Correct, 1 = Incorrect)
    errors = (preds_arr != targets_arr).astype(int)

    # Feature 1: Class Frequency in Training
    # Load taxonomy to map category_id to species_idx
    tax_map, _ = process_taxonomy(load_cached_data=True)
    cat_to_idx = dict(zip(tax_map["category_id"], tax_map["species_idx"]))

    # Load train metadata to count frequencies
    train_df = pd.read_csv(Config.TRAIN_CSV)
    train_df["species_idx"] = train_df["label"].map(cat_to_idx)

    # Count occurrences of each species_idx
    train_counts_map = train_df["species_idx"].value_counts().to_dict()

    # Map targets (ground truth species_idx) to their training counts
    # Use .get(t, 0) to handle cases where a class might be missing from train (unlikely with stratification)
    target_counts = np.array([train_counts_map.get(t, 0) for t in targets_arr])

    # Feature 2: Image File Size
    # Retrieve image paths from the validation dataset
    val_df = val_loader.dataset.df
    file_sizes = []
    for path in val_df["image_path"]:
        full_path = os.path.join(Config.INPUT_DIR, path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except OSError:
            file_sizes.append(0)
    file_sizes = np.array(file_sizes)

    # Calculate Correlations
    if len(errors) > 1 and np.std(errors) > 0:
        corr_freq = np.corrcoef(errors, target_counts)[0, 1]
        corr_size = np.corrcoef(errors, file_sizes)[0, 1]

        print(
            f"Correlation between Error and Training Class Frequency: {corr_freq:.4f}"
        )
        print(f"Correlation between Error and Image File Size: {corr_size:.4f}")
    else:
        logger.warning("Insufficient variance in errors to calculate correlation.")


def main():
    seed_everything(Config.SEED)

    # 1. Train the Model
    logger.info("Step 1: Training Model...")
    trainer = Trainer(debug=False)
    trainer.fit()

    # 2. Validation
    logger.info("Step 2: Validating Best Model...")

    # Load the best model weights
    model = trainer.model
    if os.path.exists(Config.BEST_MODEL_PATH):
        logger.info(f"Loading weights from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        logger.warning("Best model checkpoint not found. Using current model state.")

    model.eval()
    device = torch.device(Config.DEVICE)

    # Run validation
    # Note: validate() returns (avg_loss, macro_f1)
    _, final_f1 = validate(model, trainer.val_loader, trainer.criterion, device)

    # Print required metric format
    print(f"Final Validation Metric: {final_f1}")

    # 3. Failure Analysis
    perform_failure_analysis(model, trainer.val_loader, device)

    # 4. Submission
    THRESHOLD = 0.6021914648406147
    logger.info(f"Step 3: Checking Threshold ({final_f1} vs {THRESHOLD})...")

    if final_f1 > THRESHOLD:
        logger.info("Threshold passed. Generating submission...")
        predict_and_submit(
            checkpoint_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_FILE
        )
    else:
        logger.info("Threshold not passed. Skipping submission generation.")


if __name__ == "__main__":
    main()
