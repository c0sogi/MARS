import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, get_device, compute_roc_auc
from library.data_loader import get_train_val_datasets
from library.model import MILEfficientNet
from library.train_eval import run_training, predict_and_submit


def perform_failure_analysis(val_ids, val_targets, val_probs):
    """
    Analyzes model errors against input features (Slice Count).
    """
    print("\n--- Failure Analysis ---")

    # Load metadata to link IDs to file paths
    try:
        val_df = pd.read_csv(Config.VAL_METADATA)
    except FileNotFoundError:
        print("Metadata file not found, skipping detailed feature correlation.")
        return

    # Calculate absolute errors
    errors = np.abs(np.array(val_targets) - np.array(val_probs))

    analysis_data = []

    # Iterate through validation samples
    for pid, err, target in zip(val_ids, errors, val_targets):
        # Find corresponding metadata row
        row = val_df[val_df["BraTS21ID"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]

        # Feature Extraction: Count slices in FLAIR directory
        # This serves as a proxy for "Scan Volume" or "Data Quantity"
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        num_slices = 0
        if os.path.exists(flair_path):
            num_slices = len([f for f in os.listdir(flair_path) if f.endswith(".dcm")])

        analysis_data.append(
            {"BraTS21ID": pid, "error": err, "target": target, "num_slices": num_slices}
        )

    if not analysis_data:
        print("No analysis data could be aggregated.")
        return

    df_analysis = pd.DataFrame(analysis_data)

    # 1. Correlation with Slice Count
    if df_analysis["num_slices"].std() > 0:
        slice_corr = df_analysis["error"].corr(df_analysis["num_slices"])
        print(f"Correlation between Error and Slice Count: {slice_corr:.4f}")
    else:
        print("Slice count is constant, cannot compute correlation.")

    # 2. Correlation with Target Class
    if df_analysis["target"].std() > 0:
        target_corr = df_analysis["error"].corr(df_analysis["target"])
        print(f"Correlation between Error and Target Class: {target_corr:.4f}")
    else:
        print("Target class is constant, cannot compute correlation.")

    print("------------------------\n")


def main():
    # 1. Setup
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    # 2. Train Model
    # This will train, validate per epoch, and save the best model to Config.MODEL_SAVE_PATH
    print("Starting training pipeline...")
    run_training(epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)

    # 3. Final Validation Assessment
    print("Loading best model for final validation...")
    model = MILEfficientNet().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Best model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    # We ignore the train dataset here
    _, val_ds = get_train_val_datasets(load_cached=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_targets = []
    all_probs = []
    all_ids = []

    # Inference Loop
    with torch.no_grad():
        for data, target, ids in val_loader:
            data = data.to(device)
            target = target.to(device)

            # Forward Pass
            logits = model(data)
            logits = logits.squeeze(1)

            probs = torch.sigmoid(logits).cpu().numpy()

            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs)
            all_ids.extend(ids.numpy())

    # Compute Metric
    final_auc = compute_roc_auc(all_targets, all_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    perform_failure_analysis(all_ids, all_targets, all_probs)

    # 5. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.6321818181818182

    if final_auc > THRESHOLD:
        print(
            f"Validation metric ({final_auc:.4f}) > Threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        predict_and_submit(batch_size=Config.BATCH_SIZE)
    else:
        print(
            f"Validation metric ({final_auc:.4f}) <= Threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
