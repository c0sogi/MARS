import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineMIL


def predict_test_set(load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    print("Starting Inference Pipeline...")

    # 1. Setup
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    # We use the test metadata which contains unique studies to avoid redundant processing
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Error: Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    dataset = CervicalSpineDataset(
        Config.TEST_METADATA_PATH, phase="test", load_cached_data=load_cached_data
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    model = CervicalSpineMIL(pretrained=False)  # Architecture only

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            "Warning: No trained model found. Using random initialization for debugging."
        )

    model.to(device)
    model.eval()

    # 4. Generate Predictions
    # Dictionary to store results: study_id -> {prediction_type -> probability}
    study_predictions = {}

    print("Running inference on test batches...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]

            outputs = model(images)

            # Apply Sigmoid to get probabilities
            probs_vert = torch.sigmoid(outputs["vertebrae_logits"]).cpu().numpy()
            probs_patient = torch.sigmoid(outputs["patient_logit"]).cpu().numpy()

            for i, study_id in enumerate(study_ids):
                preds = {
                    "C1": float(probs_vert[i, 0]),
                    "C2": float(probs_vert[i, 1]),
                    "C3": float(probs_vert[i, 2]),
                    "C4": float(probs_vert[i, 3]),
                    "C5": float(probs_vert[i, 4]),
                    "C6": float(probs_vert[i, 5]),
                    "C7": float(probs_vert[i, 6]),
                    "patient_overall": float(probs_patient[i, 0]),
                }
                study_predictions[study_id] = preds

    # 5. Format Submission
    # We need to map the study-level predictions to the specific rows requested in test.csv
    test_csv_path = os.path.join(Config.INPUT_DIR, "test.csv")
    sample_submission_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")

    # Prefer test.csv as it contains the structure, fallback to sample_submission
    if os.path.exists(test_csv_path):
        df = pd.read_csv(test_csv_path)
    elif os.path.exists(sample_submission_path):
        df = pd.read_csv(sample_submission_path)
    else:
        print("Error: No test.csv or sample_submission.csv found.")
        return

    results = []

    # Check if we have the helper columns 'StudyInstanceUID' and 'prediction_type'
    # If not (e.g. using sample_submission), we must parse row_id
    has_metadata_cols = (
        "StudyInstanceUID" in df.columns and "prediction_type" in df.columns
    )

    for idx, row in df.iterrows():
        row_id = row["row_id"]

        if has_metadata_cols:
            study_id = row["StudyInstanceUID"]
            pred_type = row["prediction_type"]
        else:
            # Parse row_id: [StudyID]_[PredictionType]
            if row_id.endswith("_patient_overall"):
                # Handle the underscore in 'patient_overall'
                pred_type = "patient_overall"
                study_id = row_id.replace("_patient_overall", "")
            else:
                # Handle C1-C7
                parts = row_id.rsplit("_", 1)
                if len(parts) == 2:
                    study_id = parts[0]
                    pred_type = parts[1]
                else:
                    # Fallback
                    study_id = ""
                    pred_type = ""

        # Retrieve prediction
        prob = 0.5  # Default
        if study_id in study_predictions:
            if pred_type in study_predictions[study_id]:
                prob = study_predictions[study_id][pred_type]

        results.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
