import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_metrics
from library.data import get_loaders
from library.engine import run_training, predict_and_submit
from library.model import AnimalModel
from library.loss import get_class_weights


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")

    # Adjust Config for runtime constraints (Fast Baseline)
    # 5 epochs on A100 with ~160k images is sufficient for convergence
    # and fits comfortably within the time limit.
    Config.EPOCHS = 5

    logger.info("Initializing pipeline...")

    # 2. Data Loading
    # Load cached data if available to speed up startup
    train_loader, val_loader, test_loader = get_loaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Training
    if os.path.exists(Config.BEST_MODEL_PATH):
        logger.info(
            f"Found existing model at {Config.BEST_MODEL_PATH}. Skipping training."
        )
        best_f1_score = 0.0  # Placeholder, will be re-evaluated
    else:
        logger.info(f"Starting training for {Config.EPOCHS} epochs...")
        best_f1_score = run_training(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=Config.EPOCHS,
            patience=3,
        )

    # 4. Evaluation & Failure Analysis
    logger.info("Loading best model for evaluation...")

    # Initialize model structure
    model = AnimalModel(pretrained=False)

    # Load weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        logger.error(f"Best model not found at {Config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # Run Inference on Validation Set
    logger.info("Running inference on validation set...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Final Metric
    final_metric = calculate_metrics(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing failure analysis...")

    # Calculate error (0 for correct, 1 for incorrect)
    errors = (all_preds != all_targets).astype(int)

    # Get class weights to use as a feature (proxy for class rarity)
    # We need to reconstruct the weight map.
    # Note: We re-read the train csv here just to map weights to the validation samples quickly
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    class_weights_tensor = get_class_weights(df_train)
    class_weights_map = {i: w.item() for i, w in enumerate(class_weights_tensor)}

    # Map targets to their class weights
    target_weights = np.array([class_weights_map.get(t, 1.0) for t in all_targets])

    # Calculate correlation between Error and Class Weight
    # Hypothesis: Higher weight (rarer class) -> Higher Error
    corr, p_value = pearsonr(errors, target_weights)

    print(
        f"Correlation between Error and Class Rarity (Weight): {corr:.4f} (p-value: {p_value:.4f})"
    )

    # 5. Submission
    threshold = 0.44583477715072195
    if final_metric > threshold:
        logger.info(
            f"Validation metric {final_metric} > {threshold}. Generating submission..."
        )
        predict_and_submit(model, test_loader, Config.DEVICE, Config.SUBMISSION_PATH)
    else:
        logger.warning(
            f"Validation metric {final_metric} <= {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
