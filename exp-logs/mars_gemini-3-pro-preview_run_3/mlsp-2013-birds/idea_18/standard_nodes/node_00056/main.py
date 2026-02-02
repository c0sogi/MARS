import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, average_checkpoints
from library.data import get_dataloaders, get_folds
from library.models import ModelFactory
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Training Loop (5 Folds x 3 Archs)
    # Map: rec_id -> { 'y_true': ..., 'y_pred_accum': ..., 'count': ... }
    oof_data = {}

    # Get fold info to initialize oof_data keys
    # load_cached_data=False ensures we have the dataframe in memory even if we don't use cache for training
    df_folds = get_folds(load_cached_data=True)
    all_rec_ids = df_folds["rec_id"].unique()

    # Initialize storage for OOF predictions
    for rid in all_rec_ids:
        oof_data[rid] = {
            "y_true": None,
            "y_pred_accum": np.zeros(Config.NUM_CLASSES),
            "count": 0,
        }

    # Iterate over folds and architectures
    for fold in range(Config.NUM_FOLDS):
        print(f"\n=== Fold {fold}/{Config.NUM_FOLDS - 1} ===")

        # Load DataLoaders for this fold
        train_loader, val_loader = get_dataloaders(fold, load_cached_data=True)

        # Capture Ground Truth for this fold's validation set
        # We access the dataframe directly from the dataset to ensure alignment with rec_ids
        val_df = val_loader.dataset.df
        val_rec_ids = val_df["rec_id"].values

        # Collect true labels from the loader (which handles parsing)
        y_trues_fold = []
        for _, labels in val_loader:
            y_trues_fold.append(labels.numpy())
        y_trues_fold = np.vstack(y_trues_fold)

        # Store y_true in the global dictionary
        for i, rid in enumerate(val_rec_ids):
            oof_data[rid]["y_true"] = y_trues_fold[i]

        for arch in Config.ARCHITECTURES:
            print(f"--- Training {arch} ---")

            # Initialize Model
            model = ModelFactory.create_model(
                arch, num_classes=Config.NUM_CLASSES, pretrained=True
            )
            model.to(device)

            # Optimizer & Scheduler
            # High weight decay as per strategy
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.NUM_EPOCHS
            )

            # Initialize Trainer
            trainer = Trainer(model, optimizer, scheduler, device, fold, arch)

            # Train and get Top-K checkpoints
            top_k_paths = trainer.fit(
                train_loader, val_loader, num_epochs=Config.NUM_EPOCHS
            )

            # Average Checkpoints
            avg_path = os.path.join(
                Config.CHECKPOINT_DIR, f"avg_{arch}_fold_{fold}.pth"
            )
            average_checkpoints(top_k_paths, avg_path)

            # --- Generate OOF Predictions for this Architecture ---
            # Load the averaged weights
            model.load_state_dict(torch.load(avg_path))
            model.eval()

            preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    # Standard inference for validation (TTA is used for Test)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    preds.append(probs.cpu().numpy())

            preds = np.vstack(preds)

            # Accumulate predictions in global dictionary
            for i, rid in enumerate(val_rec_ids):
                oof_data[rid]["y_pred_accum"] += preds[i]
                oof_data[rid]["count"] += 1

            # Cleanup to free memory
            del model, optimizer, scheduler, trainer
            torch.cuda.empty_cache()

    # 3. Compute Final Validation Metric
    print("\nComputing Final Validation Metric...")
    final_y_true = []
    final_y_pred = []

    # Prepare list for failure analysis
    analysis_data = []

    for rid, data in oof_data.items():
        if data["y_true"] is None:
            continue

        y_t = data["y_true"]
        # Average the predictions across the ensemble (3 architectures)
        y_p = data["y_pred_accum"] / data["count"]

        final_y_true.append(y_t)
        final_y_pred.append(y_p)

        # Compute Mean Absolute Error for this sample
        mae = np.mean(np.abs(y_t - y_p))
        analysis_data.append({"rec_id": rid, "error": mae})

    final_y_true = np.array(final_y_true)
    final_y_pred = np.array(final_y_pred)

    # Calculate Macro-Averaged AUC
    auc_scores = []
    for i in range(Config.NUM_CLASSES):
        # Only calculate AUC if the class is present in the ground truth
        if len(np.unique(final_y_true[:, i])) > 1:
            auc_scores.append(roc_auc_score(final_y_true[:, i], final_y_pred[:, i]))

    final_metric = np.mean(auc_scores) if auc_scores else 0.0

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Merge error data with file paths
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.merge(df_folds[["rec_id", "file_path"]], on="rec_id")

    energies = []
    stds = []

    # Extract audio features
    for idx, row in analysis_df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.flatten()

            # Compute signal stats
            energies.append(np.mean(data**2))
            stds.append(np.std(data))
        except Exception as e:
            energies.append(0)
            stds.append(0)

    analysis_df["energy"] = energies
    analysis_df["std"] = stds

    # Compute Correlations
    corr_energy = analysis_df["error"].corr(analysis_df["energy"])
    corr_std = analysis_df["error"].corr(analysis_df["std"])

    print(f"Correlation (Error vs Signal Energy): {corr_energy}")
    print(f"Correlation (Error vs Signal Std): {corr_std}")

    # 5. Submission
    threshold = 0.9479806884980326

    if final_metric > threshold:
        print("\nMetric threshold passed. Generating submission...")
        engine = InferenceEngine()
        engine.generate_submission()
    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
