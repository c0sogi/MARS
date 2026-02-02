import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, load_checkpoint
from library.train import run_fold, get_binary_labels
from library.inference import predict_ensemble
from library.dataset import get_dataloaders
from library.model import BirdResNet

# Try importing IterativeStratification
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_ITERATIVE = True
except ImportError:
    from sklearn.model_selection import KFold

    HAS_ITERATIVE = False


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Adjust Config for Fast Baseline
    Config.EPOCHS = 30
    print(f"Configured for fast baseline: EPOCHS={Config.EPOCHS}")

    # 2. Load Data
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        print("Metadata not found. Exiting.")
        return

    train_orig = pd.read_csv(Config.TRAIN_CSV)
    val_orig = pd.read_csv(Config.VAL_CSV)

    # Combine for CV
    dev_df = pd.concat([train_orig, val_orig], ignore_index=True)

    # 3. Stratified Split
    X = dev_df["rec_id"].values.reshape(-1, 1)
    y = get_binary_labels(dev_df)

    folds = []
    if HAS_ITERATIVE:
        print("Using IterativeStratification for splitting.")
        k_fold = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)
        for train_idx, val_idx in k_fold.split(X, y):
            folds.append((train_idx, val_idx))
    else:
        print("Using KFold for splitting.")
        k_fold = KFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        for train_idx, val_idx in k_fold.split(X):
            folds.append((train_idx, val_idx))

    # Storage for OOF analysis
    oof_preds = []
    oof_targets = []
    oof_rec_ids = []
    oof_indices = []

    # 4. Train Folds
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        train_fold_df = dev_df.iloc[train_indices]
        val_fold_df = dev_df.iloc[val_indices]

        # Run training for the fold
        run_fold(fold_idx, train_fold_df, val_fold_df, load_cached_data=True)

        # Load best model for inference on validation set
        device = Config.DEVICE
        model = BirdResNet(pretrained=False, num_classes=Config.NUM_CLASSES)
        model.to(device)

        checkpoint_name = f"fold_{fold_idx}_best.pth"
        if not load_checkpoint(checkpoint_name, model, device=device):
            print(f"Failed to load checkpoint for fold {fold_idx}")
            continue

        model.eval()

        # Get Val Loader
        _, val_loader, _ = get_dataloaders(
            pd.DataFrame(), val_fold_df, pd.DataFrame(), load_cached_data=True
        )

        fold_probs = []
        fold_targets = []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)

                fold_probs.append(probs.cpu().numpy())
                fold_targets.append(targets.cpu().numpy())

        fold_probs = np.concatenate(fold_probs, axis=0)
        fold_targets = np.concatenate(fold_targets, axis=0)

        oof_preds.append(fold_probs)
        oof_targets.append(fold_targets)
        oof_rec_ids.extend(val_fold_df["rec_id"].values)
        oof_indices.extend(val_indices)

    # 5. Global Validation Metric
    if not oof_preds:
        print("No predictions generated.")
        return

    all_preds = np.concatenate(oof_preds, axis=0)
    all_targets = np.concatenate(oof_targets, axis=0)

    # Calculate Macro AUC
    final_metric = calculate_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")

    # Reconstruct DataFrame for OOF samples to map back to files
    # We used indices from dev_df
    analysis_df = dev_df.iloc[oof_indices].copy()
    analysis_df["error_magnitude"] = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Feature 1: Number of Labels
    analysis_df["num_labels"] = analysis_df["labels"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and x != "?" else 0
    )

    # Feature 2 & 3: Signal Energy and Std
    # We need to read audio files. Since dataset is small, we can do this quickly.
    energies = []
    stds = []

    for _, row in analysis_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            audio, _ = sf.read(full_path)
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Normalize length roughly to avoid bias if any (though all are 10s)
            energies.append(np.mean(audio**2))
            stds.append(np.std(audio))
        except Exception:
            energies.append(0)
            stds.append(0)

    analysis_df["signal_energy"] = energies
    analysis_df["signal_std"] = stds

    # Calculate Correlations
    features = ["num_labels", "signal_energy", "signal_std"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if analysis_df[feat].std() > 0:
            corr, _ = pearsonr(analysis_df["error_magnitude"], analysis_df[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: N/A (Constant)")

    # 7. Conditional Submission
    threshold = 0.9072993371210134
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        predict_ensemble(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
