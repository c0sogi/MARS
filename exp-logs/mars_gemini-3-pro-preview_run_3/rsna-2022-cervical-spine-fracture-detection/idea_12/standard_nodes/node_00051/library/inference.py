import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import RSNAModel
from library.data import RSNADataset
from library.utils import get_logger, seed_everything


def predict(debug=False, batch_size=None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
        batch_size (int): Batch size for inference. Defaults to Config.BATCH_SIZE * 2.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("inference")
    device = torch.device(Config.DEVICE)

    if batch_size is None:
        batch_size = Config.BATCH_SIZE * 2

    logger.info(f"Starting inference. Device: {device}, Debug: {debug}")

    # 2. Data Preparation
    # We load metadata manually here to support the 'debug' slicing
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)
        logger.info(f"Debug mode: Sliced test data to {len(test_df)} samples.")

    dataset = RSNADataset(test_df, subset="test", transform=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # We use pretrained=False because we are loading specific weights or using random init
    model = RSNAModel(pretrained=False)
    model.to(device)

    # Load Weights
    weights_path = Config.MODEL_SAVE_PATH
    if os.path.exists(weights_path):
        logger.info(f"Loading model weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"No model weights found at {weights_path}. Using random initialization for demonstration."
        )

    model.eval()

    # 4. Inference Loop
    all_probs = []

    logger.info(f"Running inference on {len(dataset)} studies...")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            # Mixed Precision Inference
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        all_probs = np.zeros((0, 8))

    # 5. Post-processing & Formatting
    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Assign probabilities to the dataframe
    # test_df index aligns with loader iteration because shuffle=False
    pred_df = test_df.copy()
    for i, col in enumerate(target_cols):
        pred_df[col] = all_probs[:, i]

    # Melt to row-wise format: [StudyInstanceUID, prediction_type, fractured]
    melted_df = pred_df.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=target_cols,
        var_name="prediction_type",
        value_name="fractured",
    )

    # Create row_id: StudyInstanceUID + "_" + prediction_type
    melted_df["row_id"] = (
        melted_df["StudyInstanceUID"] + "_" + melted_df["prediction_type"]
    )

    # 6. Submission Generation
    # We rely on sample_submission.csv to define the required rows and order
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into the sample submission structure
        # We merge left on sample_sub to ensure we output exactly the rows requested
        submission = sample_sub[["row_id"]].merge(
            melted_df[["row_id", "fractured"]], on="row_id", how="left"
        )

        # Fill missing values (e.g., if debug mode skipped some studies)
        # We fill with a low probability default
        submission["fractured"] = submission["fractured"].fillna(0.05)

    else:
        # Fallback if sample_submission is missing
        submission = melted_df[["row_id", "fractured"]]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print sample for verification
    logger.info("First 5 predictions:")
    logger.info("\n" + submission.head().to_string())
