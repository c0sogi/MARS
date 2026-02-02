import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything, get_device, log_message
from library.dataset import get_dataloaders, get_test_loader
from library.model import BirdResNetSPP
from library.trainer import Trainer


def main():
    # 1. Setup
    Config.setup()
    seed_everything()
    device = get_device()

    log_message(f"Device: {device}")
    log_message("Starting 5-Fold Cross-Validation Training...")

    # Dictionary to store Out-Of-Fold predictions: rec_id -> probability vector
    oof_preds_dict = {}

    # 2. Training Loop
    for fold_idx in range(Config.N_FOLDS):
        log_message(f"\n=== Training Fold {fold_idx} ===")

        # Get DataLoaders (cached data will be generated if missing)
        train_loader, val_loader = get_dataloaders(fold_idx, load_cached_data=True)

        # Initialize Model
        model = BirdResNetSPP()

        # Initialize Trainer
        trainer = Trainer(model, device)

        # Train
        best_auc = trainer.fit(train_loader, val_loader, fold_idx)

        # Generate predictions for the validation fold (OOF)
        # We use the best model state which was loaded at the end of fit()
        ids, probs = trainer.predict(val_loader)

        # Store predictions
        for i, rec_id in enumerate(ids):
            oof_preds_dict[rec_id] = probs[i]

    log_message("\nTraining Complete.")

    # 3. Validation Assessment
    # Load the specific hold-out validation dataset metadata
    val_df = pd.read_csv(Config.VAL_METADATA)

    y_true = []
    y_pred = []

    # Align OOF predictions with the validation metadata
    # We only evaluate on the samples present in val.csv
    missing_ids = 0
    for idx, row in val_df.iterrows():
        rec_id = row["rec_id"]

        if rec_id in oof_preds_dict:
            # Parse ground truth labels
            lbl_str = str(row["labels"])
            label_vec = np.zeros(Config.NUM_CLASSES)
            if lbl_str != "?" and lbl_str != "nan":
                try:
                    indices = [int(x) for x in lbl_str.split()]
                    label_vec[indices] = 1.0
                except ValueError:
                    pass

            y_true.append(label_vec)
            y_pred.append(oof_preds_dict[rec_id])
        else:
            missing_ids += 1

    if missing_ids > 0:
        log_message(
            f"Warning: {missing_ids} validation samples missing from OOF predictions."
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute AUC
    # We calculate AUC per class and average, handling classes with no positive/negative samples
    aucs = []
    for i in range(Config.NUM_CLASSES):
        # Only calculate if both classes are present
        if len(np.unique(y_true[:, i])) > 1:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(score)

    final_val_metric = np.mean(aucs) if aucs else 0.5

    # Print required metric
    print(f"Final Validation Metric: {final_val_metric}")

    # 4. Failure Analysis
    # Compute Binary Cross Entropy per sample
    # Clip predictions to avoid log(0)
    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # BCE = - (y * log(p) + (1-y) * log(1-p))
    # We average over the classes to get a scalar loss per sample
    sample_losses = -np.mean(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped),
        axis=1,
    )

    # Get number of labels per sample
    num_labels = np.sum(y_true, axis=1)

    # Calculate correlation
    if len(sample_losses) > 1:
        correlation = np.corrcoef(sample_losses, num_labels)[0, 1]
    else:
        correlation = 0.0

    print("Failure Analysis:")
    print(f"Correlation between Error (BCE) and Number of Labels: {correlation}")

    # 5. Submission
    threshold = 0.9072993371210134

    if final_val_metric > threshold:
        log_message(
            f"\nMetric ({final_val_metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        test_loader = get_test_loader(load_cached_data=True)

        # Ensemble Inference
        ensemble_probs = None
        test_ids = None

        for fold_idx in range(Config.N_FOLDS):
            log_message(f"Inference with model from Fold {fold_idx}...")

            # Load Model
            model = BirdResNetSPP()
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth"
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            # Predict
            trainer = Trainer(model, device)
            ids, probs = trainer.predict(test_loader)

            if ensemble_probs is None:
                ensemble_probs = probs
                test_ids = ids
            else:
                ensemble_probs += probs

        # Average
        ensemble_probs /= Config.N_FOLDS

        # Format Submission
        submission_rows = []
        for i, rec_id in enumerate(test_ids):
            for species_idx in range(Config.NUM_CLASSES):
                # Construct Id: rec_id * 100 + species
                row_id = int(rec_id * 100 + species_idx)
                prob = ensemble_probs[i, species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        # Save
        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        log_message(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        log_message(
            f"\nMetric ({final_val_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
