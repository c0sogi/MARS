import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add current directory to path
sys.path.append(".")

# Import from library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import prepare_data
from library.model import WIISNet
from library.train import run_training
from library.predict import predict_and_submit


def main():
    # 1. Configuration & Setup
    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 10  # Reduced epochs for speed

    seed_everything(Config.SEED)
    device = get_device()

    # 2. Training
    print("--- Executing Training Pipeline ---")
    run_training()

    # 3. Validation & Evaluation
    print("\n--- Executing Validation Pipeline ---")

    # Load validation data
    # prepare_data returns (train, val, test)
    _, val_dataset, _ = prepare_data(load_cached_data=Config.LOAD_CACHED_DATA)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model saved during training
    model = WIISNet()
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(f"Critical Error: Model checkpoint not found at {Config.BEST_MODEL_PATH}")
        return

    model = model.to(device)
    model.eval()

    # Inference on Validation Set
    all_targets = []
    all_preds = []

    # We disable gradients for validation inference
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # labels are tensors

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(labels.numpy())

    # Compute Final Metric
    # Aggregate slab predictions by Subject ID before calculating AUC
    subject_ids = val_dataset.ids

    df_val_preds = pd.DataFrame(
        {"BraTS21ID": subject_ids, "prob": all_preds, "target": all_targets}
    )

    # Group by Subject ID: Mean of probabilities, First of target
    df_agg = df_val_preds.groupby("BraTS21ID").agg({"prob": "mean", "target": "first"})

    final_auc = roc_auc_score(df_agg["target"], df_agg["prob"])
    print(f"Final Validation Metric (Subject-Level): {final_auc}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate absolute errors
    targets_arr = df_agg["target"].values
    preds_arr = df_agg["prob"].values
    errors = np.abs(targets_arr - preds_arr)
    agg_ids = df_agg.index.values

    # Create a DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "BraTS21ID": agg_ids,
            "Target": targets_arr,
            "Prediction": preds_arr,
            "Error": errors,
        }
    )

    # Correlation with Target (Class Bias)
    corr_target = df_analysis["Error"].corr(df_analysis["Target"])
    print(f"Correlation between Error and Target Class: {corr_target}")

    # Correlation with Subject ID (Potential Batch/Temporal Drift)
    corr_id = df_analysis["Error"].corr(df_analysis["BraTS21ID"])
    print(f"Correlation between Error and Subject ID: {corr_id}")

    # 5. Conditional Submission
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        print(f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        predict_and_submit()
    else:
        print(
            f"\nValidation AUC ({final_auc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
