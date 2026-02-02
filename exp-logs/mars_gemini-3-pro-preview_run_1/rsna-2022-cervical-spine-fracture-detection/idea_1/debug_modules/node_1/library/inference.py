import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import FractureSliceDataset, get_transforms, prepare_inference_data
from library.model import FractureClassifier
from library.train import seed_everything


def run_inference(load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Loads test data (slices).
    2. Loads the trained model.
    3. Generates slice-level predictions.
    4. Aggregates to patient-level via Max Pooling.
    5. Formats and saves the submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Starting Inference...")

    # 2. Data Loading
    # prepare_inference_data handles caching internally based on the flag
    test_df = prepare_inference_data(load_cached_data=load_cached_data)

    if len(test_df) == 0:
        print(
            "Warning: No test data found. Generating empty submission based on sample file."
        )
        # We still need to generate a submission file to avoid errors
        if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
            sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
            sub.to_csv(Config.SUBMISSION_PATH, index=False)
        return

    test_dataset = FractureSliceDataset(
        test_df, Config.TEST_IMAGES_DIR, transform=get_transforms("val"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Loading
    model = FractureClassifier(
        pretrained=False
    )  # Pretrained weights not needed for structure, we load state_dict
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    print(f"Predicting on {len(test_dataset)} slices...")

    with torch.no_grad():
        for images, uids, slice_nums in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            preds = outputs.cpu().numpy()

            # Store results
            # uids is a tuple of strings from the dataloader
            for i in range(len(uids)):
                row_data = {"StudyInstanceUID": uids[i]}
                for idx, col in enumerate(Config.TARGET_COLS):
                    row_data[col] = preds[i][idx]
                results.append(row_data)

    # 5. Aggregation (Max Pooling)
    print("Aggregating predictions...")
    if not results:
        # Fallback if loop didn't run
        df_pred_agg = pd.DataFrame(columns=["StudyInstanceUID"] + Config.TARGET_COLS)
    else:
        df_results = pd.DataFrame(results)
        # Group by StudyInstanceUID and take the max probability for each target column
        df_pred_agg = (
            df_results.groupby("StudyInstanceUID")[Config.TARGET_COLS]
            .max()
            .reset_index()
        )

    # 6. Submission Formatting
    print("Formatting submission...")

    # Load sample submission to get the correct row_ids
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    else:
        # If sample submission is missing, we can't easily guess the exact rows required
        # unless we strictly follow test.csv. We'll assume it exists as per prompt.
        print("Error: sample_submission.csv not found.")
        return

    # Create a lookup dictionary for fast mapping
    # Key: "{StudyInstanceUID}_{TargetName}" -> Value: Probability
    prediction_map = {}

    for idx, row in df_pred_agg.iterrows():
        uid = row["StudyInstanceUID"]
        for col in Config.TARGET_COLS:
            key = f"{uid}_{col}"
            prediction_map[key] = row[col]

    # Function to map row_id to probability
    def get_probability(row_id):
        # row_id format: "1.2.826..._C1" or "1.2.826..._patient_overall"
        if row_id in prediction_map:
            return prediction_map[row_id]
        else:
            # If we didn't predict this ID (e.g. image folder missing),
            # return a low probability or neutral value.
            # Since this is a fracture detection task, missing data usually implies
            # we can't detect a fracture.
            return 0.05

    # Apply mapping
    sample_sub["fractured"] = sample_sub["row_id"].apply(get_probability)

    # Save
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(sample_sub.head())
