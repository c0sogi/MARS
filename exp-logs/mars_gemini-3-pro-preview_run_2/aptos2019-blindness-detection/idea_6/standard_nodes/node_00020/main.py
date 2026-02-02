import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# Import library modules
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import RetinopathyDataset, get_transforms
from library.engine import run_fold
from library.model import DRModel
from library.inference import predict_ensemble


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We use 5 epochs to ensure convergence while staying within time limits.
    # We use 5 folds as required by the task description to build the full ensemble.
    Config.override(EPOCHS=5, NUM_FOLDS=5, DEBUG=False)

    print("=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Folds: {Config.NUM_FOLDS}")
    print(f"Working Dir: {Config.WORKING_DIR}")

    # 2. Prepare Data for Training (K-Fold)
    # Load the full training set
    train_full_df = pd.read_csv(Config.TRAIN_CSV)

    # Define Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Stratify based on diagnosis
    y = train_full_df["diagnosis"].values

    # 3. Training Loop
    model_configs = [Config.MODEL_CNN, Config.MODEL_TRANS]

    for model_cfg in model_configs:
        print(f"\n\n>>> Training Architecture: {model_cfg['name']} <<<")

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_full_df, y)):
            print(f"\n--- Fold {fold} ---")

            # Split Data
            fold_train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_full_df.iloc[val_idx].reset_index(drop=True)

            # Create Datasets
            # Note: For training, we use 'train' mode (augments).
            # For fold-validation (early stopping), we use 'val' mode.
            train_dataset = RetinopathyDataset(
                fold_train_df,
                transforms=get_transforms(model_cfg["img_size"], mode="train"),
                mode="train",
                input_dir=Config.INPUT_DIR,
            )

            val_dataset = RetinopathyDataset(
                fold_val_df,
                transforms=get_transforms(model_cfg["img_size"], mode="val"),
                mode="val",  # Returns image, label
                input_dir=Config.INPUT_DIR,
            )

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_cfg["batch_size"],
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=model_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
            )

            # Run Training for this Fold
            run_fold(fold, model_cfg, train_loader, val_loader)

    # 4. Final Validation Assessment (Ensemble)
    print("\n\n=== Final Validation Assessment ===")
    val_holdout_df = pd.read_csv(Config.VAL_CSV)

    # Perform Ensemble Inference on Hold-out Set
    # We need to manually run inference using all 10 models

    ensemble_preds = np.zeros(len(val_holdout_df), dtype=np.float32)
    model_count = 0

    device = Config.DEVICE

    for model_cfg in model_configs:
        img_size = model_cfg["img_size"]
        model_name = model_cfg["name"]
        prefix = model_cfg["checkpoint_prefix"]
        batch_size = model_cfg["batch_size"]

        # DataLoader for Validation
        val_ds = RetinopathyDataset(
            val_holdout_df,
            transforms=get_transforms(img_size, mode="val"),
            mode="test",  # We just want images, we have labels in df separately
            input_dir=Config.INPUT_DIR,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        for fold in range(Config.NUM_FOLDS):
            # Load Checkpoint (Prefer SWA, fallback to Best)
            swa_path = os.path.join(Config.WORKING_DIR, f"{prefix}_fold_{fold}_swa.pth")
            best_path = os.path.join(
                Config.WORKING_DIR, f"{prefix}_fold_{fold}_best.pth"
            )

            ckpt_path = swa_path if os.path.exists(swa_path) else best_path
            if not os.path.exists(ckpt_path):
                continue

            # Load Model
            model = DRModel(model_name=model_name, pretrained=False)
            state_dict = torch.load(ckpt_path, map_location=device)

            # Clean state dict (handle DataParallel or SWA wrappers)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k == "n_averaged":
                    continue
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images in val_loader:
                    images = images.to(device)
                    # TTA: Original + Flip
                    out1 = model(images)
                    out2 = model(torch.flip(images, dims=[3]))
                    batch_preds = (out1 + out2) / 2.0
                    fold_preds.append(batch_preds.cpu().numpy())

            ensemble_preds += np.concatenate(fold_preds).flatten()
            model_count += 1

            del model
            torch.cuda.empty_cache()

    # Calculate Final Metric
    if model_count > 0:
        avg_preds = ensemble_preds / model_count
        final_preds = np.rint(avg_preds).clip(0, 4).astype(int)
    else:
        final_preds = np.zeros(len(val_holdout_df), dtype=int)

    y_true = val_holdout_df["diagnosis"].values.astype(int)
    final_qwk = quadratic_weighted_kappa(y_true, final_preds)

    print(f"Final Validation Metric: {final_qwk}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_true - final_preds)

    # Extract metadata features for validation set
    print("Extracting metadata for failure analysis...")
    meta_stats = []

    for idx, row in val_holdout_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                mean_int = img.mean()
                std_int = img.std()
                fsize = os.path.getsize(full_path)
                ar = w / h if h > 0 else 0
                meta_stats.append(
                    {
                        "width": w,
                        "height": h,
                        "aspect_ratio": ar,
                        "file_size": fsize,
                        "mean_intensity": mean_int,
                        "std_intensity": std_int,
                        "error": errors[idx],
                    }
                )
        except Exception:
            pass

    if meta_stats:
        df_stats = pd.DataFrame(meta_stats)
        print("\nCorrelation between Error Magnitude and Image Features:")
        for col in [
            "width",
            "height",
            "aspect_ratio",
            "file_size",
            "mean_intensity",
            "std_intensity",
        ]:
            if col in df_stats.columns:
                corr, _ = spearmanr(df_stats[col], df_stats["error"])
                print(f"{col}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.9207435978935975
    if final_qwk > THRESHOLD:
        print(
            f"\nMetric ({final_qwk}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_ensemble(load_cached_data=False)
    else:
        print(
            f"\nMetric ({final_qwk}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
