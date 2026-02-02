import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, calculate_metric, rank_normalize
from library.dataset import load_data, AppleDataset, get_transforms
from library.trainer import SWATrainer
from library.model import DiseaseClassifier


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print(f"Starting execution for experiment: {Config.EXP_NAME}")

    # 2. Load Data
    # We load cached data if available to save time
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Combine train and val for 5-Fold CV
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    # 3. Initialize Trainer and CV
    trainer = SWATrainer()
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_aucs = []

    # Containers for global failure analysis
    all_oof_preds = []
    all_oof_targets = []
    all_oof_ids = []

    # 4. Training and Validation Loop
    # We iterate through folds first, then models, to build the ensemble for each fold
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["stratify_label"])
    ):
        print(
            f"\n================ Processing Fold {fold}/{Config.N_FOLDS - 1} ================"
        )

        train_sub = full_df.iloc[train_idx].reset_index(drop=True)
        val_sub = full_df.iloc[val_idx].reset_index(drop=True)

        fold_model_ranks = []

        # Train and Predict with each model architecture
        for model_conf in Config.MODEL_CONFIGS:
            # A. Train
            # This saves the SWA model checkpoint to disk
            trainer.train_one_fold(fold, train_sub, val_sub, model_conf)

            # B. Inference on Validation Fold (for Ensembling)
            print(f"Generating validation predictions for {model_conf['name']}...")

            # Load SWA Model
            model = DiseaseClassifier(model_conf["name"], pretrained=False)
            ckpt_path = os.path.join(
                Config.WORKING_DIR, f"swa_model_{model_conf['name']}_fold_{fold}.pth"
            )
            model.load_weights(ckpt_path, device=Config.DEVICE)
            model.to(Config.DEVICE)
            model.eval()

            # Create Validation Loader
            val_ds = AppleDataset(
                val_sub,
                transform=get_transforms(model_conf["img_size"], "val"),
                output_label=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=model_conf["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Predict
            preds_list = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(Config.DEVICE)
                    with autocast():
                        logits = model(imgs)
                        probs = torch.sigmoid(logits)
                    preds_list.append(probs.cpu().numpy())

            raw_probs = np.concatenate(preds_list, axis=0)  # Shape: (N_val, 2)

            # Rank Normalize Predictions for this model
            # This aligns the distributions of different models before averaging
            ranks = rank_normalize(raw_probs)
            fold_model_ranks.append(ranks)

        # C. Heterogeneous Ensemble (Rank Averaging)
        avg_ranks = np.mean(fold_model_ranks, axis=0)  # Shape: (N_val, 2)

        # D. Reconstruct 4-Class Probabilities
        r_rank = avg_ranks[:, 0]
        s_rank = avg_ranks[:, 1]

        # Reconstruct: [Healthy, Multiple, Rust, Scab]
        # Using the trainer's helper function which handles the math
        final_probs = trainer.reconstruct_probs(r_rank, s_rank)

        # E. Reconstruct Ground Truth Targets
        # val_sub has 'target_rust' and 'target_scab' (binary indicators)
        t_r = val_sub["target_rust"].values
        t_s = val_sub["target_scab"].values

        # Logic:
        # Healthy: (1-tr)(1-ts)
        # Multiple: tr*ts
        # Rust Only: tr*(1-ts)
        # Scab Only: (1-tr)*ts
        targets = np.stack(
            [(1 - t_r) * (1 - t_s), t_r * t_s, t_r * (1 - t_s), (1 - t_r) * t_s], axis=1
        )

        # F. Calculate Fold Metric
        auc = calculate_metric(targets, final_probs)
        fold_aucs.append(auc)
        print(f"Fold {fold} Ensemble ROC AUC: {auc:.6f}")

        # Store for global failure analysis
        all_oof_preds.append(final_probs)
        all_oof_targets.append(targets)
        all_oof_ids.extend(val_sub["image_id"].tolist())

    # 5. Global Validation Metric
    # We report the mean of the fold AUCs as per the task description/idea
    final_metric = np.mean(fold_aucs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Concatenate all OOF data
    oof_preds = np.concatenate(all_oof_preds, axis=0)
    oof_targets = np.concatenate(all_oof_targets, axis=0)

    # Calculate Error Magnitude
    # We use Mean Absolute Error across the 4 classes
    errors = np.mean(np.abs(oof_preds - oof_targets), axis=1)

    # Extract Input Features (File Size, Dimensions)
    # We need to read the images corresponding to the OOF IDs
    print("Extracting image features for correlation analysis...")
    path_map = pd.Series(full_df.file_path.values, index=full_df.image_id).to_dict()

    file_sizes = []
    widths = []
    heights = []

    for img_id in all_oof_ids:
        rel_path = path_map.get(img_id)
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            # File Size
            file_sizes.append(os.path.getsize(full_path))

            # Dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Calculate Correlations
    if len(errors) > 1:
        corr_size, _ = pearsonr(errors, file_sizes)
        corr_w, _ = pearsonr(errors, widths)
        corr_h, _ = pearsonr(errors, heights)

        print(f"Correlation (Error vs File Size): {corr_size:.4f}")
        print(f"Correlation (Error vs Width): {corr_w:.4f}")
        print(f"Correlation (Error vs Height): {corr_h:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 7. Submission
    threshold = 0.9954104122251848
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold}). Generating submission..."
        )
        # This function handles TTA, Rank Normalization, and CSV generation
        trainer.predict_and_submit(test_df)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
