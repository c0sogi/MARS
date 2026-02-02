import os
import torch
import numpy as np
import pandas as pd
from library.config import SEED, VAL_META_PATH, CACHE_DIR, NUM_EPOCHS
from library.utils import seed_everything, compute_roc_auc, get_device
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.model import MSSHDNetwork
from library.inference import predict_and_submit


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Use cached data to ensure fast loading
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    # The dataset is small (approx 400 samples), so 15 epochs is very fast (minutes).
    # We use the full dataset to ensure a robust baseline.
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    trainer = Trainer(train_loader, val_loader)
    trainer.run()

    # 4. Evaluation on Validation Set (Best Model)
    print("Evaluating best model on validation set...")
    model = MSSHDNetwork()
    model.to(device)

    # Load best weights saved by the Trainer
    checkpoint_path = os.path.join(CACHE_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Checkpoint not found, using last model state.")
        model = trainer.model

    model.eval()

    all_targets = []
    all_preds = []
    all_ids = []

    # Inference loop without gradient calculation for speed
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            ids = batch["BraTS21ID"]

            logits = model(imgs)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()

            all_targets.extend(targets.cpu().numpy().flatten())
            all_preds.extend(preds)
            all_ids.extend(ids)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Compute and print the mandatory metric
    val_auc = compute_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Load metadata to get input features (slice counts)
    if os.path.exists(VAL_META_PATH):
        val_df = pd.read_parquet(VAL_META_PATH)

        # Map IDs to errors
        id_to_error = dict(zip(all_ids, errors))

        analysis_data = []
        for idx, row in val_df.iterrows():
            pid = row["BraTS21ID"]
            if pid in id_to_error:
                # Extract meta-features: slice counts per modality
                flair_count = len(row.get("flair_paths", []))
                t1w_count = len(row.get("t1w_paths", []))
                t1wce_count = len(row.get("t1wce_paths", []))
                t2w_count = len(row.get("t2w_paths", []))
                total_slices = flair_count + t1w_count + t1wce_count + t2w_count

                analysis_data.append(
                    {
                        "error": id_to_error[pid],
                        "flair_count": flair_count,
                        "t1w_count": t1w_count,
                        "t1wce_count": t1wce_count,
                        "t2w_count": t2w_count,
                        "total_slices": total_slices,
                    }
                )

        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            correlations = analysis_df.corr()["error"].drop("error")
            print(
                "Correlation between Error Magnitude and Input Features (Slice Counts):"
            )
            print(correlations)
        else:
            print("Could not align validation IDs for failure analysis.")
    else:
        print(
            f"Validation metadata not found at {VAL_META_PATH}. Skipping failure analysis."
        )

    # 6. Submission
    # Check against the specific threshold
    threshold = 0.6978181818181817
    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc:.16f}) > Threshold ({threshold:.16f}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"\nValidation AUC ({val_auc:.16f}) <= Threshold ({threshold:.16f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
