import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import CMTSINModel
from library.data import get_dataloaders


def predict_and_submit():
    """
    Loads the trained model, generates predictions on the test set,
    aggregates them by prediction_id, and saves the submission file.
    """
    print("Initializing Inference...")

    # 1. Setup Device
    device = torch.device(Config.DEVICE)

    # 2. Load Model
    print(f"Loading model architecture: {Config.MODEL_NAME}")
    model = CMTSINModel()
    model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.eval()

    # 3. Load Test Data
    print("Loading test data...")
    # We only need the test loader (3rd return value)
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 4. Inference Loop
    print("Starting inference...")
    results = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            prediction_ids = batch["prediction_id"]  # List of strings

            # Forward pass
            outputs = model(images, meta)

            # Extract cancer logits and apply sigmoid
            # outputs['cancer'] shape is (B,)
            logits = outputs["cancer"]
            probs = torch.sigmoid(logits).cpu().numpy()

            # Store results
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # 5. Aggregation
    print("Aggregating predictions...")
    df_results = pd.DataFrame(results)

    # Group by prediction_id and take the MAX probability
    # This aligns with the strategy: robust single-instance predictions aggregated via max-pooling
    submission_df = df_results.groupby("prediction_id")["cancer"].max().reset_index()

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    print(f"Total predictions: {len(submission_df)}")
    print("Head of submission:")
    print(submission_df.head())
