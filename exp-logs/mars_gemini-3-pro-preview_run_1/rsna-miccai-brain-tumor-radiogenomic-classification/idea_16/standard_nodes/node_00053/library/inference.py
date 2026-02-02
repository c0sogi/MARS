import os
import pandas as pd
from library.config import Config
from library.utils import log_message
from library.trainer import Trainer
from library.dataset import get_dataloader


def predict_and_submit(load_cached_data=True):
    """
    Executes the inference pipeline for the WIIS-Net solution.

    Steps:
    1. Initializes the Trainer (which builds the model structure).
    2. Loads the test dataset using the standardized DataLoader factory.
       This handles the deterministic expansion of test subjects into 3 slabs each.
    3. Runs the prediction loop, which loads the best saved model weights.
    4. Aggregates the slab-level probabilities into subject-level predictions via averaging.
    5. Saves the properly formatted submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 for the test set to speed up initialization.
    """
    log_message("Starting inference pipeline...")

    # 1. Initialize Trainer
    # The trainer handles model instantiation and device placement
    trainer = Trainer()

    # 2. Load Test Data
    # get_dataloader handles metadata reading, slab expansion, and caching
    log_message("Loading test data...")
    test_loader = get_dataloader("test", load_cached=load_cached_data)

    if len(test_loader.dataset) == 0:
        log_message("Warning: Test dataset is empty. Creating empty submission.")
        df_submission = pd.DataFrame(columns=["BraTS21ID", "MGMT_value"])
    else:
        # 3. Generate Predictions
        # trainer.predict loads the best model weights, runs inference,
        # and performs the consensus aggregation (mean of 3 slabs per subject)
        log_message("Running prediction and consensus aggregation...")
        df_submission = trainer.predict(test_loader)

    # 4. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH

    # Ensure correct column order and types
    # BraTS21ID should be integer, MGMT_value float
    if not df_submission.empty:
        df_submission["BraTS21ID"] = df_submission["BraTS21ID"].astype(int)
        df_submission = df_submission[["BraTS21ID", "MGMT_value"]]

    df_submission.to_csv(save_path, index=False)
    log_message(f"Submission saved to {save_path}")
    log_message(f"Submission shape: {df_submission.shape}")
