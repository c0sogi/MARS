import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import DataCacher, MRIDataset, get_transforms
from library.model import AsymmetricEfficientNet
from library.train import train_model
from library.inference import Predictor


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Initialize Config and Directories
    Config.setup()

    # Set reproducibility
    seed_everything(Config.SEED)

    # Configure for Fast Baseline
    # We reduce epochs to ensure the run completes quickly while still learning.
    # The dataset is small (~500 samples), so 10 epochs is sufficient for a baseline.
    Config.EPOCHS = 10

    logger = get_logger(__name__)
    logger.info("Starting runfile.py execution...")

    # --------------------------------------------------------------------------
    # 2. Training Pipeline
    # --------------------------------------------------------------------------
    logger.info("--- Initiating Training Phase ---")
    # train_model() handles data loading, model init, training loop, and saving best_model.pth
    train_model()

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    logger.info("--- Initiating Validation & Failure Analysis Phase ---")

    device = torch.device(Config.DEVICE)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        logger.error("Best model not found! Training might have failed.")
        sys.exit(1)

    # Load the best model
    model = AsymmetricEfficientNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Metadata and Cache
    # Note: DataCacher will load the cache created during training (efficient)
    df_val = pd.read_csv(Config.VAL_METADATA)
    val_cache = DataCacher.process_data(df_val, cache_key="val", load_cached_data=True)

    # Initialize Dataset
    ds_val = MRIDataset(val_cache, df_val, transform=get_transforms("val"))

    y_true = []
    y_pred = []
    errors = []

    logger.info("Running Validation Inference...")

    with torch.no_grad():
        for i in range(len(ds_val)):
            # 1. Get Data
            img, label_tensor = ds_val[i]
            label = label_tensor.item()

            # 2. Construct TTA Batch
            # Strategy: [Orig, HFlip, VFlip]
            batch_imgs = []
            batch_imgs.append(img)
            batch_imgs.append(torch.flip(img, dims=[2]))  # Horizontal Flip (W)
            batch_imgs.append(torch.flip(img, dims=[1]))  # Vertical Flip (H)

            batch_tensor = torch.stack(batch_imgs).to(device)

            # 3. Inference
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

            # 4. Aggregation
            avg_prob = np.mean(probs)

            # 5. Store Results
            y_true.append(label)
            y_pred.append(avg_prob)

            # 6. Error Calculation
            error = abs(label - avg_prob)
            errors.append(error)

    # Calculate Final Metric
    final_auc = roc_auc_score(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Basic Stats
    print("\nFailure Analysis:")
    print(f"Mean Absolute Error: {np.mean(errors):.4f}")

    # --------------------------------------------------------------------------
    # 4. Submission Generation
    # --------------------------------------------------------------------------
    # Threshold defined in task requirements
    THRESHOLD = 0.6321818181818182

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predictor = Predictor()
        predictor.run()
    else:
        logger.info(
            f"Validation AUC ({final_auc}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
