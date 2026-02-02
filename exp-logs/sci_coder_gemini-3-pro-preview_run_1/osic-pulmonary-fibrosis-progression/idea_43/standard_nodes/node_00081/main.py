import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings("ignore")

# Monkeypatch tqdm to disable progress bars globally
import tqdm


class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    @classmethod
    def write(cls, msg):
        sys.stdout.write(msg + "\n")


tqdm.tqdm = SilentTqdm

# Import library modules
from library.config import Config
from library.train import train_model, generate_submission
from library.utils import calculate_metric, seed_everything
from library.data import get_dataloaders


def main():
    # 1. Configuration for Fast Baseline
    # Limit epochs to ensure quick execution while allowing convergence
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32
    Config.DEBUG = False  # Use full dataset (approx 1100 samples is small enough)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 2. Train Model
    # This trains, saves best model, and returns the model with best weights loaded
    model = train_model(debug=Config.DEBUG)

    # 3. Validation Inference
    device = torch.device(Config.DEVICE)
    model.eval()

    # Get validation loader (shuffle=False ensures alignment with metadata)
    _, val_loader = get_dataloaders(batch_size=Config.BATCH_SIZE, debug=Config.DEBUG)

    all_fvc_true = []
    all_fvc_pred = []
    all_sigma_pred = []

    # Run inference without gradients
    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            outputs = model(img_axial, img_coronal, tabular, delta_week, baseline_fvc)

            all_fvc_true.append(target.cpu().numpy())
            all_fvc_pred.append(outputs["fvc_pred"].cpu().numpy())
            all_sigma_pred.append(outputs["sigma_pred"].cpu().numpy())

    y_true = np.concatenate(all_fvc_true)
    y_pred = np.concatenate(all_fvc_pred)
    sigma_pred = np.concatenate(all_sigma_pred)

    # 4. Compute and Print Metric
    metric = calculate_metric(y_true, y_pred, sigma_pred)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    # Load validation metadata
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # If debug mode was used, slice the dataframe to match
    if Config.DEBUG:
        val_df = val_df.head(Config.DEBUG_SIZE)

    # Ensure alignment
    if len(val_df) == len(y_true):
        val_df["Error_Mag"] = np.abs(y_true - y_pred)

        # Encode categoricals for correlation analysis
        le = LabelEncoder()
        if "Sex" in val_df.columns:
            val_df["Sex_Enc"] = le.fit_transform(val_df["Sex"].astype(str))
        if "SmokingStatus" in val_df.columns:
            val_df["Smoking_Enc"] = le.fit_transform(
                val_df["SmokingStatus"].astype(str)
            )

        # Select features for correlation
        features = ["Weeks", "Percent", "Age", "Sex_Enc", "Smoking_Enc"]
        features = [f for f in features if f in val_df.columns]

        print("Failure Analysis (Correlation with Error Magnitude):")
        correlations = val_df[features].corrwith(val_df["Error_Mag"])
        print(correlations)
    else:
        print(
            "Warning: Validation dataframe length mismatch. Skipping detailed failure analysis."
        )

    # 6. Submission
    # Threshold defined in the task
    threshold = -6.510164260864258

    if metric > threshold:
        generate_submission(model)
    else:
        print(
            f"Metric {metric} is not higher than threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
