import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import seed_everything
from library.dataset import AnimalDataset, get_transforms
from library.model import MultiTaskConvNeXt
from library.engine import train_model, generate_submission


def run():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 1. Train Model
    # We limit to 2 epochs to ensure execution finishes within the 2-hour limit
    # while utilizing the full dataset for maximum performance.
    print("=== Starting Training Phase ===")
    train_model(epochs=2)

    # 2. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")
    device = torch.device(Config.DEVICE)

    # Load Validation Dataset
    val_dataset = AnimalDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model (EMA weights saved during training)
    model = MultiTaskConvNeXt(pretrained=False).to(device)
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")

    try:
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from {Config.BEST_MODEL_PATH}")
    except FileNotFoundError:
        print("Error: Best model file not found. Training might have failed.")
        return

    model.eval()

    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["species_label"].to(device)

            outputs = model(images)
            # Use species_logits for final classification
            preds = torch.argmax(outputs["species_logits"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate Metric
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"Final Validation Metric: {val_f1}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    df_val = val_dataset.df.copy()

    # Ensure dataframe length matches predictions
    if len(df_val) == len(all_preds):
        df_val["pred"] = all_preds
        df_val["target"] = all_targets
        df_val["is_error"] = (df_val["pred"] != df_val["target"]).astype(int)

        # 1. Correlation with Class Frequency
        try:
            # Load train metadata to get true training distribution
            train_df = pd.read_csv(Config.TRAIN_META)
            class_counts = train_df["Category"].value_counts().to_dict()

            # Map counts to validation samples based on their true category
            df_val["train_class_freq"] = df_val["target"].map(class_counts)

            # Calculate correlation
            corr = df_val["is_error"].corr(df_val["train_class_freq"])
            print(f"Correlation between Error and Training Class Frequency: {corr}")
            print("(Negative correlation implies rare classes have higher error rates)")

        except Exception as e:
            print(f"Could not calculate frequency correlation: {e}")

        # 2. Worst Performing Classes
        print("\nTop 5 Classes with Highest Error Rates:")
        class_error_rate = df_val.groupby("target")["is_error"].mean()
        print(class_error_rate.sort_values(ascending=False).head(5))

    else:
        print(
            "Warning: Mismatch between validation set size and prediction count. Skipping detailed analysis."
        )

    # 3. Submission
    print("\n=== Submission Generation ===")
    threshold = 0.6423934219391719

    if val_f1 > threshold:
        print(f"Validation metric {val_f1} exceeds threshold {threshold}.")
        generate_submission()
    else:
        print(
            f"Validation metric {val_f1} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
