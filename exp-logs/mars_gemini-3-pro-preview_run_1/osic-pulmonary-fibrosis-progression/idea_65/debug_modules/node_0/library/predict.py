import os
import torch
import numpy as np
import pandas as pd
from library.model import TSCPNet
from library.data import get_dataloaders
from library.utils import seed_everything


def generate_submission(
    model_path="./working/best_model.pth", batch_size=16, debug=False
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a subset of data for debugging.
    """
    # Set seed for reproducibility
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    # get_dataloaders handles caching and preprocessing internally via OSICDataset
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cache=True, debug=debug
    )

    # Load Model
    model = TSCPNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: Model path {model_path} does not exist. Using random initialized weights."
        )

    model.eval()

    all_fvc_preds = []
    all_sigma_preds = []

    # Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)  # Relative weeks from baseline
            base_fvc = batch["base_fvc"].to(device)  # Baseline FVC

            # Predict trajectory parameters
            # alpha: slope of decline/incline
            # sigma_base: uncertainty at baseline
            # sigma_growth: growth of uncertainty over time
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate FVC Prediction: Base + Slope * Time
            fvc_pred = base_fvc + alpha * weeks

            # Calculate Confidence Prediction: Base_Sigma + Growth_Sigma * |Time|
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Collect results
            all_fvc_preds.append(fvc_pred.cpu().numpy())
            all_sigma_preds.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    if len(all_fvc_preds) > 0:
        all_fvc_preds = np.concatenate(all_fvc_preds)
        all_sigma_preds = np.concatenate(all_sigma_preds)
    else:
        all_fvc_preds = np.array([])
        all_sigma_preds = np.array([])

    # Load test metadata to align with Patient_Week IDs
    # The test_loader iterates sequentially over metadata/test.csv (shuffle=False)
    test_df = pd.read_csv("./metadata/test.csv")

    # Handle debug case where loader returns fewer samples than full metadata
    if len(all_fvc_preds) != len(test_df):
        if debug:
            test_df = test_df.iloc[: len(all_fvc_preds)]
        else:
            print(
                f"Error: Mismatch between predictions ({len(all_fvc_preds)}) and metadata ({len(test_df)})"
            )

    # Assign predictions
    test_df["FVC"] = all_fvc_preds
    test_df["Confidence"] = all_sigma_preds

    # Clip confidence at 70ml as per metric requirement
    test_df["Confidence"] = test_df["Confidence"].clip(lower=70)

    # Prepare submission dataframe
    submission = test_df[["Patient_Week", "FVC", "Confidence"]]

    # Save to file
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
