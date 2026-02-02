import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.train import Trainer
from library.model import RTICNN
from library.data import load_data, get_loaders
from library.utils import set_seed, get_device
from library.config import SEED, CHECKPOINT_DIR, SUBMISSION_PATH


def main():
    # 1. Setup
    set_seed(SEED)
    device = get_device()

    # 2. Train
    # Using 50 epochs ensures the model converges on this small dataset
    # while keeping execution time well within limits.
    trainer = Trainer(epochs=50)
    trainer.train()

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load raw data to get IDs and reconstruct folds
    (X_train, angles_train, y_train, ids_train), _ = load_data(load_cached_data=True)

    # Reconstruct StratifiedKFold splits to match training logic
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y_train)), y_train))

    # Storage for OOF predictions
    # Map ID -> (Prediction, Label, Angle, B1_Mean, B2_Mean)
    oof_data = {}

    # Iterate folds to generate OOF predictions
    for fold_idx in range(5):
        # Get validation loader for this fold
        # Note: get_loaders handles the split internally, but we need the IDs manually
        _, val_loader = get_loaders(fold_idx, batch_size=32, load_cached_data=True)

        # Identify IDs for this fold's validation set
        _, val_indices = splits[fold_idx]
        fold_ids = ids_train[val_indices]

        # Load Model
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint {model_path} not found. Skipping fold.")
            continue

        model = RTICNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        labels = []
        angles_list = []
        b1_means = []
        b2_means = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward
                logits = model(images, angles)
                probs = torch.sigmoid(logits)

                preds.extend(probs.cpu().numpy())
                labels.extend(targets.numpy())
                angles_list.extend(angles.cpu().numpy())

                # Compute image stats for failure analysis
                # images: (B, 3, 75, 75) -> Channel 0 is HH (Band 1), Channel 1 is HV (Band 2)
                # We compute mean per image
                imgs_np = images.cpu().numpy()
                b1_means.extend(np.mean(imgs_np[:, 0, :, :], axis=(1, 2)))
                b2_means.extend(np.mean(imgs_np[:, 1, :, :], axis=(1, 2)))

        # Store in dictionary
        for i, uid in enumerate(fold_ids):
            oof_data[uid] = {
                "pred": preds[i],
                "label": labels[i],
                "angle": angles_list[i],
                "b1_mean": b1_means[i],
                "b2_mean": b2_means[i],
            }

    # 4. Filter using Metadata (val.csv)
    # The task requires loading the hold-out validation dataset using the metadata.
    val_meta_path = "./metadata/val.csv"
    val_df = pd.read_csv(val_meta_path)

    # Extract data for the validation subset
    val_subset_data = []
    for uid in val_df["id"]:
        if uid in oof_data:
            val_subset_data.append(oof_data[uid])
        else:
            # Should not happen if splits are consistent
            print(f"Warning: ID {uid} not found in OOF predictions.")

    df_res = pd.DataFrame(val_subset_data)

    # 5. Calculate Metric
    y_true = df_res["label"].values
    y_pred = df_res["pred"].values

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_res["error"] = np.abs(df_res["label"] - df_res["pred"])

    # Correlations
    features_to_corr = ["angle", "b1_mean", "b2_mean"]
    correlations = df_res[features_to_corr].corrwith(df_res["error"])

    print("Correlation with Error Magnitude:")
    print(correlations)

    # 7. Submission
    threshold = 0.17174082291273365
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
