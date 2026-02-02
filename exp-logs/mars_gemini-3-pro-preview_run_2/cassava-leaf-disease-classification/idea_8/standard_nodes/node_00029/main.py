import os
import time
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint
from library.transforms import get_transforms, get_mixup_fn
from library.dataset import CassavaDataset
from library.model import get_model
from library.engine import train_one_epoch, validate


def main():
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger(os.path.join(Config.working_dir, "train.log"))
    logger.info("Starting runfile.py execution...")

    # 2. Data Preparation
    # Load metadata
    df_train_meta = pd.read_csv(Config.train_metadata_path)
    df_val_meta = pd.read_csv(Config.val_metadata_path)

    # Combine for CV
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    df_full["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full["label"])):
        df_full.loc[val_idx, "fold"] = fold

    logger.info(
        f"Data prepared. Total samples: {len(df_full)}. Folds: {Config.n_folds}"
    )

    # Placeholders for OOF predictions
    oof_preds = np.zeros((len(df_full), Config.num_classes))
    oof_targets = np.zeros((len(df_full),))

    # 3. Cross-Validation Loop
    for fold in range(Config.n_folds):
        fold_start = time.time()
        logger.info(f"\n{'='*20} Fold {fold+1}/{Config.n_folds} {'='*20}")

        # Split Data
        df_train = df_full[df_full["fold"] != fold].reset_index(drop=True)
        df_valid = df_full[df_full["fold"] == fold].reset_index(drop=True)

        # Initialize Model
        model, model_ema = get_model(
            Config.model_name,
            Config.num_classes,
            pretrained=True,
            use_ema=Config.use_ema,
        )

        # Define Phases (Progressive Resizing)
        phases = [1, 2]

        for phase in phases:
            phase_cfg = Config.get_phase_config(phase)
            logger.info(
                f"--- Phase {phase}: Res {phase_cfg['image_size']}x{phase_cfg['image_size']}, Epochs {phase_cfg['epochs']} ---"
            )

            # Prepare DataLoaders
            train_dataset = CassavaDataset(
                df_train, transform=get_transforms("train", phase_cfg["image_size"])
            )
            valid_dataset = CassavaDataset(
                df_valid, transform=get_transforms("valid", phase_cfg["image_size"])
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=phase_cfg["batch_size"],
                shuffle=True,
                num_workers=Config.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=phase_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            # Optimizer & Scheduler
            # Re-initialize for each phase to adapt to new batch dynamics
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.learning_rate,
                weight_decay=Config.weight_decay,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase_cfg["epochs"], eta_min=Config.min_lr
            )

            loss_fn = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
            mixup_fn = get_mixup_fn(phase_cfg)

            # Training Loop
            for epoch in range(phase_cfg["epochs"]):
                avg_loss = train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    device=torch.device(Config.device),
                    epoch=epoch,
                    loss_fn=loss_fn,
                    max_norm=None,
                    model_ema=model_ema,
                    mixup_fn=mixup_fn,
                    accum_iter=phase_cfg["accum_iter"],
                )
                scheduler.step()

            # Validate at end of phase
            current_model = model_ema.module if model_ema else model
            val_metrics = validate(
                model=current_model,
                data_loader=valid_loader,
                loss_fn=loss_fn,
                device=torch.device(Config.device),
            )
            logger.info(
                f"Phase {phase} finished. Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}"
            )

        # Save Best Model (EMA) for this Fold
        final_model = model_ema.module if model_ema else model
        save_path = os.path.join(Config.working_dir, f"fold_{fold}_best.pth")
        torch.save(final_model.state_dict(), save_path)

        # Generate OOF Predictions (using best resolution)
        final_valid_dataset = CassavaDataset(
            df_valid, transform=get_transforms("valid", Config.phase2_image_size)
        )
        final_valid_loader = DataLoader(
            final_valid_dataset,
            batch_size=Config.phase2_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        preds = []
        targets = []
        final_model.eval()
        with torch.no_grad():
            for imgs, lbls in final_valid_loader:
                imgs = imgs.to(Config.device)
                output = final_model(imgs)
                preds.append(output.softmax(dim=1).cpu().numpy())
                targets.append(lbls.numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        # Store OOF
        val_indices = df_full[df_full["fold"] == fold].index
        oof_preds[val_indices] = preds
        oof_targets[val_indices] = targets

        logger.info(f"Fold {fold} completed in {time.time() - fold_start:.0f}s")

    # 4. Overall Validation & Failure Analysis
    # Compute accuracy
    oof_predictions_indices = oof_preds.argmax(axis=1)
    oof_acc = (oof_predictions_indices == oof_targets).mean()

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {oof_acc}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate error magnitude (1 - probability assigned to the true class)
    # oof_targets are indices of true class
    true_class_probs = oof_preds[np.arange(len(oof_preds)), oof_targets.astype(int)]
    error_magnitudes = 1.0 - true_class_probs

    # Get file sizes for correlation analysis
    file_sizes = []
    for idx, row in df_full.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])
        try:
            st = os.stat(full_path)
            file_sizes.append(st.st_size)
        except:
            file_sizes.append(0)

    # Calculate correlation
    if len(file_sizes) == len(error_magnitudes):
        try:
            corr, _ = pearsonr(error_magnitudes, file_sizes)
            print(f"Correlation between Error Magnitude and File Size: {corr:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate correlation: {e}")

    # 5. Submission
    if oof_acc > 0.9076:
        logger.info("Validation metric passed threshold. Generating submission...")
        generate_submission()
    else:
        logger.info(
            f"Validation metric {oof_acc} did not pass threshold 0.9076. Submission skipped."
        )


def generate_submission():
    # Load Test Metadata
    df_test = pd.read_csv(Config.test_metadata_path)

    # Load all fold models
    models = []
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.working_dir, f"fold_{fold}_best.pth")
        # Initialize model structure (no pretrained weights needed as we load state_dict)
        model, _ = get_model(
            Config.model_name, Config.num_classes, pretrained=False, use_ema=False
        )
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.eval()
        models.append(model)

    # Prepare Test Loader
    test_ds = CassavaDataset(
        df_test,
        transform=get_transforms("test", Config.phase2_image_size),
        output_label=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.phase2_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    final_preds = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(Config.device)

            # Accumulate predictions from all models and TTA views
            batch_preds = torch.zeros(
                (imgs.size(0), Config.num_classes), device=Config.device
            )

            for model in models:
                # 1. Standard View
                out = model(imgs)
                batch_preds += out.softmax(dim=1)

                # 2. Horizontal Flip (TTA)
                if Config.tta_flips:
                    out_flip = model(torch.flip(imgs, dims=[3]))
                    batch_preds += out_flip.softmax(dim=1)

            # Average predictions
            # Divisor: num_folds * (1 original + 1 flip)
            div_factor = len(models) * (2 if Config.tta_flips else 1)
            batch_preds /= div_factor

            final_preds.append(batch_preds.argmax(dim=1).cpu().numpy())

    final_preds = np.concatenate(final_preds)

    # Create submission file
    df_sub = pd.DataFrame({"image_id": df_test["image_id"], "label": final_preds})

    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


if __name__ == "__main__":
    main()
