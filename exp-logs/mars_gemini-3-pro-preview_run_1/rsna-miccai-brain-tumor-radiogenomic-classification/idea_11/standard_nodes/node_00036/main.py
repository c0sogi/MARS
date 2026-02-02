import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import and override Config
from library.config import Config

# Increased epochs to compensate for reduced dataset size (1 slice vs 3)
Config.NUM_EPOCHS = 30

# Import library modules
from library.utils import seed_everything, get_device
from library.train import run_training
from library.inference import predict_and_submit
from library.model import GLiClassModel
from library.data import get_loaders


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Training
    print("\n=== Starting Training ===")
    run_training(fold=None, num_epochs=Config.NUM_EPOCHS)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load the best model
    model = GLiClassModel(pretrained=False)
    model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print("Error: Model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get validation loader
    _, val_loader = get_loaders(fold=None)

    # Inference
    all_probs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, subject_ids in val_loader:
            images = images.to(device)
            # targets are (B,) or (B,1) depending on loader, model outputs (B,1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_ids.append(subject_ids.numpy().flatten())

    # Concatenate
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_ids = np.concatenate(all_ids)

    # Aggregate to Subject Level
    df_val = pd.DataFrame(
        {"BraTS21ID": all_ids, "prob": all_probs, "target": all_targets}
    )

    # Mean aggregation per subject
    df_agg = df_val.groupby("BraTS21ID").mean()

    # Calculate Metric
    final_metric = roc_auc_score(df_agg["target"], df_agg["prob"])
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Calculate error magnitude
    df_agg["error"] = (df_agg["target"] - df_agg["prob"]).abs()

    # Feature Extraction: T2w Slice Count (Proxy for brain volume/scan depth)
    # Load metadata to find paths
    df_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    id_to_path = pd.Series(df_meta.t2w_path.values, index=df_meta.BraTS21ID).to_dict()

    t2w_counts = []
    for sid in df_agg.index:
        rel_path = id_to_path.get(sid)
        count = 0
        if rel_path:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            if os.path.exists(full_path):
                # Fast count of files
                try:
                    count = len(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                    )
                except OSError:
                    count = 0
        t2w_counts.append(count)

    df_agg["t2w_count"] = t2w_counts

    # Calculate Correlation
    if len(df_agg) > 1 and df_agg["t2w_count"].std() > 0:
        correlation = np.corrcoef(df_agg["error"], df_agg["t2w_count"])[0, 1]
        print(f"Correlation between Error and T2w Slice Count: {correlation}")
    else:
        print("Correlation analysis skipped (insufficient data or variance).")

    # 5. Submission
    THRESHOLD = 0.6705454545454544
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
