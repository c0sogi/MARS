import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import PulmonaryDataset
from library.model import TriSlabModel


def generate_submission():
    """
    Generates the submission file for the competition.

    Steps:
    1. Loads test metadata.
    2. Runs inference on unique patients to get trajectory parameters (alpha, sigma_base, sigma_growth).
    3. Extrapolates FVC and Confidence for all requested weeks using the parameters.
    4. Saves the result to submission.csv.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure output directory exists
    submission_dir = os.path.dirname(Config.SUBMISSION_FILE)
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Optimize Inference: Predict parameters per Patient
    # The model predicts slope (alpha) and uncertainty params which are constant per patient
    # We don't need to run the CNN for every single week, just once per patient.
    unique_patients_df = (
        test_df.drop_duplicates(subset=["Patient"]).copy().reset_index(drop=True)
    )

    # Create Dataset for unique patients
    # We use mode='test' which expects specific columns present in Config.TEST_CSV
    dataset = PulmonaryDataset(unique_patients_df, mode="test")

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Load Model
    model = TriSlabModel(Config)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Using random initialization."
        )

    model = model.to(device)
    model.eval()

    # 5. Inference Loop
    patient_params = []

    print(f"Running inference on {len(unique_patients_df)} unique patients...")
    with torch.no_grad():
        for batch_idx, (imgs, tabular, base_fvc, time_delta, _) in enumerate(loader):
            imgs = imgs.to(device)
            tabular = tabular.to(device)

            # Predict parameters: [alpha, sigma_base, sigma_growth]
            preds = model(imgs, tabular)
            preds = preds.cpu().numpy()

            # Get patient IDs for this batch
            # The dataset indices align with unique_patients_df
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + imgs.size(0)
            batch_patient_ids = unique_patients_df.iloc[start_idx:end_idx][
                "Patient"
            ].values

            for pid, pred in zip(batch_patient_ids, preds):
                patient_params.append(
                    {
                        "Patient": pid,
                        "alpha": pred[0],
                        "sigma_base": pred[1],
                        "sigma_growth": pred[2],
                    }
                )

    # Create DataFrame from predictions
    params_df = pd.DataFrame(patient_params)

    # 6. Merge Predictions back to full Test Set
    # We join on Patient ID so every week row gets the patient's params
    full_df = pd.merge(test_df, params_df, on="Patient", how="left")

    # 7. Calculate FVC and Confidence for each week
    # Formula: FVC = Baseline + alpha * delta_t
    # Formula: Conf = |sigma_base| + |sigma_growth| * |delta_t|

    # Ensure Time_Delta is calculated correctly
    # In test.csv metadata, we have Predict_Week and Baseline_Week
    full_df["Time_Delta"] = full_df["Predict_Week"] - full_df["Baseline_Week"]

    # Calculate FVC
    full_df["FVC_Pred"] = (
        full_df["Baseline_FVC"] + full_df["alpha"] * full_df["Time_Delta"]
    )

    # Calculate Confidence
    # We use abs() for sigma parameters as uncertainty magnitude must be positive
    # and abs() for Time_Delta as uncertainty grows in both past and future directions
    full_df["Confidence_Pred"] = np.abs(full_df["sigma_base"]) + np.abs(
        full_df["sigma_growth"]
    ) * np.abs(full_df["Time_Delta"])

    # Apply Clipping Rules
    # Confidence clipped at 70 ml
    full_df["Confidence_Pred"] = np.maximum(full_df["Confidence_Pred"], Config.Q_CLIP)

    # 8. Format Submission
    submission_df = full_df[["Patient_Week", "FVC_Pred", "Confidence_Pred"]].copy()
    submission_df.columns = ["Patient_Week", "FVC", "Confidence"]

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Total rows: {len(submission_df)}")
    print("Sample:")
    print(submission_df.head())
