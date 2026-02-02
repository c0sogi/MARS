import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided libraries
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import LungDataset, LungDataProcessor, get_transforms
from library.model import HighFidelityDualNet, predict
from library.train import run_training

# Setup environment
warnings.filterwarnings("ignore")
seed_everything(42)


def main():
    # Configuration and Paths
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    CACHE_DIR = "./working/idea_14"
    MODEL_SAVE_PATH = "./working/best_model_runfile.pth"

    # 1. Train the model
    # We use 15 epochs for a fast baseline execution.
    # The dataset is small, so this is sufficient to verify the idea.
    print("Starting training pipeline...")
    run_training(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        cache_dir=CACHE_DIR,
        epochs=15,
        batch_size=16,
        lr=3e-4,
        weight_decay=1e-5,
        patience=5,
        save_path=MODEL_SAVE_PATH,
    )

    # 2. Validation & Failure Analysis
    print("\nStarting validation analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model weights
    model = HighFidelityDualNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model file not found. Training may have failed.")
        return

    model.eval()

    # Prepare validation data loader
    val_df = pd.read_csv(VAL_PATH)
    processor = LungDataProcessor(cache_dir=CACHE_DIR)
    val_transforms = get_transforms("val")
    val_dataset = LungDataset(val_df, processor, transforms=val_transforms, mode="val")
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    all_targets = []
    all_preds = []
    all_sigmas = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab_vec = batch["tab_vec"].to(device)
            rel_week = batch["rel_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            baseline_fvc_sc = batch["baseline_fvc_sc"].to(device)
            targets = batch["target"].to(device)

            pred_fvc, pred_conf = model(
                img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
            )

            all_targets.append(targets.cpu())
            all_preds.append(pred_fvc.cpu())
            all_sigmas.append(pred_conf.cpu())

    # Concatenate results
    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_preds)
    sigma = torch.cat(all_sigmas)

    # Compute Metric
    final_metric = laplace_log_likelihood_metric(y_true, y_pred, sigma).item()

    # Print EXACTLY as required
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error for correlation analysis
    errors = torch.abs(y_true - y_pred).numpy()

    # Create analysis dataframe (order is preserved since shuffle=False)
    analysis_df = val_df.copy()
    analysis_df["Error"] = errors

    # Encode categorical variables for correlation
    sex_map = {"Male": 0, "Female": 1}
    smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    analysis_df["Sex_Code"] = analysis_df["Sex"].map(sex_map)
    analysis_df["Smoke_Code"] = analysis_df["SmokingStatus"].map(smoke_map)

    # Compute correlations
    features = ["Weeks", "Percent", "Age", "Sex_Code", "Smoke_Code"]
    correlations = analysis_df[features].corrwith(analysis_df["Error"])

    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # 3. Submission Generation
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        test_df = pd.read_csv(TEST_PATH)
        predict(test_df, model_path=MODEL_SAVE_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
