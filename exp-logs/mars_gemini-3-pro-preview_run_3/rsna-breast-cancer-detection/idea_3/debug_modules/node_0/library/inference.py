import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import BreastCancerDataset
from library.model import BreastCancerModel


def predict_and_submit(
    checkpoint_path: str = os.path.join(Config.WORKING_DIR, "best_model.pth"),
    output_path: str = Config.SUBMISSION_PATH,
    test_meta_path: str = Config.TEST_META_PATH,
    debug: bool = Config.DEBUG,
    subset_size: int = Config.DEBUG_SUBSET_SIZE,
):
    """
    Runs the inference pipeline:
    1. Loads the test metadata and creates a DataLoader.
    2. Loads the trained EfficientNetV2 model.
    3. Predicts logits for each image.
    4. Applies Analytical Prior Correction to adjust for balanced training.
    5. Aggregates predictions by prediction_id using Max Pooling.
    6. Saves the submission CSV.

    Args:
        checkpoint_path: Path to the trained model weights.
        output_path: Path to save the submission CSV.
        test_meta_path: Path to the test metadata CSV.
        debug: If True, runs on a subset of the test data.
        subset_size: Number of samples to use in debug mode.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting Inference on device: {device}")

    # 2. Data Loading
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    df_test = pd.read_csv(test_meta_path)

    if debug:
        print(f"DEBUG MODE: Subsetting test data to {subset_size} samples.")
        df_test = df_test.head(subset_size)

    test_dataset = BreastCancerDataset(df_test, mode="test")

    # Use num_workers from Config, but ensure it's safe for the environment
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Test DataLoader initialized with {len(test_dataset)} samples.")

    # 3. Model Initialization
    # We initialize with pretrained=False because we are loading specific weights
    # and want to avoid potential connection errors if internet is disabled.
    model = BreastCancerModel(pretrained=False)

    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using random weights.")

    model.to(device)
    model.eval()

    # 4. Analytical Prior Correction Setup
    # Formula: L_corrected = L_pred - log(P_train/(1-P_train)) + log(P_test/(1-P_test))
    p_train = Config.P_TRAIN
    p_test = Config.P_TEST

    term_train = np.log(p_train / (1 - p_train))
    term_test = np.log(p_test / (1 - p_test))

    correction_factor = term_test - term_train
    print(f"Applying Analytical Logit Correction Factor: {correction_factor:.4f}")

    # 5. Inference Loop
    all_probs = []
    all_ids = []

    print("Running prediction loop...")
    with torch.no_grad():
        for images, pred_ids in test_loader:
            images = images.to(device)

            # Forward pass to get raw logits
            logits = model(images)

            # Apply Analytical Correction
            # This aligns the predicted probabilities with the natural test prevalence
            corrected_logits = logits + correction_factor

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(corrected_logits)

            # Store results
            # Flatten probs to 1D array
            all_probs.extend(probs.cpu().numpy().flatten())
            all_ids.extend(pred_ids)

    # 6. Aggregation (Max Pooling)
    print("Aggregating predictions...")
    df_pred = pd.DataFrame({"prediction_id": all_ids, "cancer": all_probs})

    # Group by prediction_id and take the maximum probability (Max Pooling)
    # This handles cases where multiple images (views) map to a single breast prediction
    submission = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # 7. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission.shape}")
    print("First 5 rows:")
    print(submission.head())
