import os
import torch
import pandas as pd
import numpy as np
from library.config import Config, set_seed
from library.model import AttentivePyramidSiamese
from library.data import get_dataloaders


def generate_submission(load_cached_data=True):
    """
    Generates the submission file by running inference on the test set.

    Args:
        load_cached_data (bool): Whether to use cached metadata/stats for data loading.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Ensure output directory exists
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # get_dataloaders returns (train, val, test). We only need test.
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)
    print(f"Test batches: {len(test_loader)}")

    # 3. Load Model
    model = AttentivePyramidSiamese(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # No need to download pretrained weights, we load checkpoint
    )

    checkpoint_path = Config.CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading model weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    print("Running inference...")
    with torch.no_grad():
        for target_input, contra_input, prediction_ids in test_loader:
            target_input = target_input.to(device)
            contra_input = contra_input.to(device)

            # Mixed precision inference if enabled
            if Config.USE_AMP and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(target_input, contra_input)
            else:
                logits = model(target_input, contra_input)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Store results
            # prediction_ids is a tuple of strings from the batch
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # 5. Aggregation
    # Convert to DataFrame
    df_pred = pd.DataFrame(results)

    # Group by prediction_id and take the MAX probability across views (CC/MLO)
    # as per task description logic.
    df_agg = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # 6. Formatting against Sample Submission
    sample_sub_path = Config.SAMPLE_SUBMISSION_PATH
    if os.path.exists(sample_sub_path):
        df_sample = pd.read_csv(sample_sub_path)
        # Keep only prediction_ids from sample submission to ensure correct order/rows
        # Merge with our predictions
        df_final = df_sample[["prediction_id"]].merge(
            df_agg, on="prediction_id", how="left"
        )

        # Fill missing values (if any prediction_ids were missing in test set processing)
        # Ideally this shouldn't happen if test.csv matches sample_submission.csv
        if df_final["cancer"].isnull().any():
            print(
                "Warning: Some prediction_ids missing in inference results. Filling with 0."
            )
            df_final["cancer"] = df_final["cancer"].fillna(0.0)
    else:
        # Fallback if sample submission not found (unlikely)
        print("Warning: Sample submission not found. Using raw aggregated predictions.")
        df_final = df_agg

    # 7. Save
    df_final.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(df_final.head())
