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
from library.data import get_dataloaders, get_folds, get_fixed_val_dataloader
from library.models import ModelFactory
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Training Loop (5 Folds x 3 Archs)
    # We switch to Hold-Out Validation Strategy (Cite solution_lesson_node_00056)
    # We train on folds of train.csv, but evaluate the FULL ensemble on val.csv

    # Load Fixed Validation Set
    fixed_val_loader = get_fixed_val_dataloader()

    # Get ground truth for fixed validation set
    val_df = fixed_val_loader.dataset.df
    val_rec_ids = val_df["rec_id"].values

    y_trues_val = []
    for _, labels in fixed_val_loader:
        y_trues_val.append(labels.numpy())
    y_trues_val = np.vstack(y_trues_val)

    # Initialize accumulator for ensemble predictions on fixed validation set
    # Shape: (Num_Val_Samples, Num_Classes)
    ensemble_val_preds = np.zeros(
        (len(val_rec_ids), Config.NUM_CLASSES), dtype=np.float32
    )
    model_count = 0

    # Iterate over folds and architectures
    for fold in range(Config.NUM_FOLDS):
        print(f"\n=== Fold {fold}/{Config.NUM_FOLDS - 1} ===")

        # Load DataLoaders for this fold (Internal CV for model selection)
        train_loader, internal_val_loader = get_dataloaders(fold, load_cached_data=True)

        for arch in Config.ARCHITECTURES:
            print(f"--- Training {arch} ---")

            # Initialize Model
            model = ModelFactory.create_model(
                arch, num_classes=Config.NUM_CLASSES, pretrained=True
            )
            model.to(device)

            # Optimizer & Scheduler
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
                train_loader, internal_val_loader, num_epochs=Config.NUM_EPOCHS
            )

            # Average Checkpoints
            avg_path = os.path.join(
                Config.CHECKPOINT_DIR, f"avg_{arch}_fold_{fold}.pth"
            )
            average_checkpoints(top_k_paths, avg_path)

            # --- Evaluate on Fixed Validation Set (Ensemble Contribution) ---
            # Load the averaged weights
            model.load_state_dict(torch.load(avg_path))
            model.eval()

            preds = []
            with torch.no_grad():
                for images, _ in fixed_val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    preds.append(probs.cpu().numpy())

            preds = np.vstack(preds)
            ensemble_val_preds += preds
            model_count += 1

            # Cleanup to free memory
            del model, optimizer, scheduler, trainer
            torch.cuda.empty_cache()

    # 3. Compute Final Validation Metric on Fixed Set
    print("\nComputing Final Validation Metric on Fixed Validation Set...")

    # Average predictions
    final_y_pred = ensemble_val_preds / model_count
    final_y_true = y_trues_val

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

    # Calculate per-sample error
    mae_per_sample = np.mean(np.abs(final_y_true - final_y_pred), axis=1)

    analysis_data = []
    for i, rid in enumerate(val_rec_ids):
        analysis_data.append({"rec_id": rid, "error": mae_per_sample[i]})

    analysis_df = pd.DataFrame(analysis_data)
    # Merge with file paths from val_df
    analysis_df = analysis_df.merge(val_df[["rec_id", "file_path"]], on="rec_id")

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
