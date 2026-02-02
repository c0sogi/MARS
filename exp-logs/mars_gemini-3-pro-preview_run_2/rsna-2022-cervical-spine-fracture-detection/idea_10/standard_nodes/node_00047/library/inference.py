import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed
from library.model import CervicalFractureNet
from library.data import get_dataloaders


def predict_test_set():
    """
    Runs inference on the test set and generates the submission file.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    # 2. Data
    # We only need the test loader
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = CervicalFractureNet()
    model.to(device)

    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # 4. Inference
    # Store predictions: study_id -> [p_C1, p_C2, ..., p_C7, p_overall]
    study_predictions = {}

    # Column mapping based on library/data.py ordering
    # Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
    target_columns = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    col_to_idx = {col: i for i, col in enumerate(target_columns)}

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]  # List of strings

            # Forward pass
            outputs = model(images)
            logits = outputs["study_logits"]
            probs = torch.sigmoid(logits).cpu().numpy()

            # Store results
            for i, study_id in enumerate(study_ids):
                study_predictions[study_id] = probs[i]

    # 5. Format Submission
    # Load test.csv to get the required row_ids
    test_csv_path = os.path.join(Config.INPUT_ROOT, "test.csv")
    if not os.path.exists(test_csv_path):
        # Fallback if test.csv is not in input (e.g. during some local tests), use sample_submission logic
        # But per instructions, test.csv is in input.
        print(f"Error: {test_csv_path} not found.")
        return

    test_df = pd.read_csv(test_csv_path)

    # We need to fill the 'fractured' column
    # test_df has: row_id, StudyInstanceUID, prediction_type

    results = []

    # Default probability if study somehow missing (should not happen with correct metadata)
    default_prob = 0.05

    for _, row in test_df.iterrows():
        row_id = row["row_id"]
        study_id = row["StudyInstanceUID"]
        pred_type = row["prediction_type"]

        prob = default_prob

        if study_id in study_predictions:
            preds = study_predictions[study_id]
            if pred_type in col_to_idx:
                idx = col_to_idx[pred_type]
                prob = float(preds[idx])
            else:
                # Should not happen
                pass

        results.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(results)

    # 6. Save
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
