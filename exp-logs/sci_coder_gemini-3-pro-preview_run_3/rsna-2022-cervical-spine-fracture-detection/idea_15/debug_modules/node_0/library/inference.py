import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, get_device
from library.dataset import CervicalSpineDataset, get_transforms
from library.model import ConvNeXtMIL


def predict_test_set(debug_size=None):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        debug_size (int, optional): If set, limits the number of test samples processed.
    """
    logger = get_logger()
    device = get_device()

    # --- 1. Load Test Metadata ---
    if not os.path.exists(Config.TEST_METADATA_PATH):
        logger.error(f"Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug_size is not None:
        logger.info(f"Debug mode: limiting inference to {debug_size} samples.")
        test_df = test_df.iloc[:debug_size]

    logger.info(f"Test samples to process: {len(test_df)}")

    # --- 2. Setup Data ---
    # The Dataset class handles the pipeline: Load -> Window -> Resize -> Stack -> Augment(None)
    dataset = CervicalSpineDataset(
        metadata_df=test_df,
        images_dir=Config.TEST_IMAGES_DIR,
        transform=get_transforms(split="test"),
        split="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Load Model ---
    # We set pretrained=False because we are loading our own trained weights
    # and want to avoid attempting to download from the internet.
    model = ConvNeXtMIL(
        model_name=Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        in_channels=Config.IN_CHANNELS,
    )

    weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(weights_path):
        logger.warning(
            f"Model weights not found at {weights_path}. Using random initialization (predictions will be random)."
        )
    else:
        logger.info(f"Loading model weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # --- 4. Inference Loop ---
    # Dictionary to store results: {StudyInstanceUID: [p_C1, ..., p_Overall]}
    results = {}

    logger.info("Starting inference...")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            uids = batch["study_uid"]

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy()

            # Store results
            for i, uid in enumerate(uids):
                results[uid] = probs[i]

    # --- 5. Generate Submission ---
    logger.info("Generating submission file...")

    # Load the template (test.csv) which defines the rows required for submission
    test_template_path = os.path.join(Config.INPUT_DIR, "test.csv")
    if not os.path.exists(test_template_path):
        logger.error(f"test.csv not found at {test_template_path}")
        return

    submission_df = pd.read_csv(test_template_path)

    # Map prediction types to the index in the model output
    # Model output order matches Dataset target order:
    # ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    col_map = {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C5": 4,
        "C6": 5,
        "C7": 6,
        "patient_overall": 7,
    }

    # Convert results dictionary to a DataFrame for efficient merging
    res_data = []
    for uid, probs in results.items():
        row = {"StudyInstanceUID": uid}
        for name, idx in col_map.items():
            row[name] = probs[idx]
        res_data.append(row)

    if not res_data:
        logger.warning(
            "No predictions generated. Submission will contain default values."
        )
        res_df = pd.DataFrame(columns=["StudyInstanceUID"] + list(col_map.keys()))
    else:
        res_df = pd.DataFrame(res_data)

    # Melt to long format: StudyInstanceUID, prediction_type, fractured_pred
    target_cols = list(col_map.keys())
    long_df = res_df.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=target_cols,
        var_name="prediction_type",
        value_name="fractured_pred",
    )

    # Merge with the submission template
    # We use left merge on the template to ensure we keep exactly the rows requested
    final_df = submission_df.merge(
        long_df, on=["StudyInstanceUID", "prediction_type"], how="left"
    )

    # Fill missing values (e.g., if debug_size was used, or if a study failed to load)
    # 0.5 is a neutral probability
    final_df["fractured"] = final_df["fractured_pred"].fillna(0.5)

    # Select required columns
    final_df = final_df[["row_id", "fractured"]]

    # Save to disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
