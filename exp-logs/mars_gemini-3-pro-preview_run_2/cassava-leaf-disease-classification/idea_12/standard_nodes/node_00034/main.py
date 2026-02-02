import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    save_checkpoint,
    load_checkpoint,
    accuracy,
)
from library.data import get_dataloaders, get_test_loader, get_mixup_fn, get_metadata
from library.model import get_model, ModelEMA
from library.engine import train_one_epoch, evaluate, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    # Ensure output directories exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    logger = get_logger(os.path.join(Config.OUTPUT_DIR, "training.log"))
    logger.info("Starting Runfile Execution")

    # Override Config for Fast Baseline Execution within 2 hours
    # 5 Folds * (6 + 4) epochs = 50 epochs total.
    # Estimated time on A100: ~1.5 hours.
    Config.P1_EPOCHS = 6
    Config.P2_EPOCHS = 4
    logger.info(
        f"Configuration: {Config.N_FOLDS} Folds, P1={Config.P1_EPOCHS} eps, P2={Config.P2_EPOCHS} eps"
    )

    # Storage for Out-Of-Fold (OOF) predictions
    oof_preds = []
    oof_targets = []
    oof_image_ids = []

    # 2. 5-Fold Stratified Training Loop
    for fold in range(Config.N_FOLDS):
        logger.info(f"\n{'='*20} Fold {fold} / {Config.N_FOLDS - 1} {'='*20}")

        # ==========================
        # Phase 1: Coarse Learning
        # ==========================
        logger.info(
            f"--- Phase 1: {Config.P1_IMG_SIZE}x{Config.P1_IMG_SIZE}, MixUp={Config.P1_MIXUP_PROB} ---"
        )

        # Data Loaders for Phase 1
        train_loader, val_loader = get_dataloaders(
            fold_idx=fold,
            img_size=Config.P1_IMG_SIZE,
            batch_size=Config.P1_BATCH_SIZE,
            load_cached_data=True,
        )

        # Model & Optimizer
        model = get_model(pretrained=True)
        model_ema = ModelEMA(model)

        optimizer = AdamW(
            model.parameters(), lr=Config.P1_LR_MAX, weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.P1_EPOCHS, eta_min=Config.MIN_LR
        )
        mixup_fn = get_mixup_fn(Config.P1_MIXUP_PROB)

        # Train Phase 1
        for epoch in range(Config.P1_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                epoch,
                model,
                optimizer,
                train_loader,
                Config.DEVICE,
                scheduler=scheduler,
                mixup_fn=mixup_fn,
                model_ema=model_ema,
                accum_iter=Config.P1_ACCUM_ITER,
            )
            # Optional: Log validation during P1 (using EMA)
            val_loss, val_acc = evaluate(model_ema.ema_model, val_loader, Config.DEVICE)
            logger.info(
                f"Fold {fold} P1 Epoch {epoch}: Train Loss {train_loss:.4f}, Val Acc {val_acc:.4f}"
            )

        # ==========================
        # Phase Reset
        # ==========================
        logger.info("--- Phase Reset: Synchronizing EMA weights ---")
        model_ema.set_weights(model)

        # ==========================
        # Phase 2: Fine-Grained Refinement
        # ==========================
        logger.info(
            f"--- Phase 2: {Config.P2_IMG_SIZE}x{Config.P2_IMG_SIZE}, Fine-tuning ---"
        )

        # Data Loaders for Phase 2
        train_loader, val_loader = get_dataloaders(
            fold_idx=fold,
            img_size=Config.P2_IMG_SIZE,
            batch_size=Config.P2_BATCH_SIZE,
            load_cached_data=True,
        )

        # Re-initialize Optimizer for fine-tuning (lower LR)
        optimizer = AdamW(
            model.parameters(), lr=Config.P2_LR_MAX, weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.P2_EPOCHS, eta_min=Config.MIN_LR
        )
        # Disable MixUp, Enable Label Smoothing
        mixup_fn = get_mixup_fn(
            Config.P2_MIXUP_PROB, label_smoothing=Config.P2_LABEL_SMOOTHING
        )

        best_acc = 0.0
        fold_save_dir = os.path.join(Config.OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # Train Phase 2
        for epoch in range(Config.P2_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                epoch,
                model,
                optimizer,
                train_loader,
                Config.DEVICE,
                scheduler=scheduler,
                mixup_fn=mixup_fn,
                model_ema=model_ema,
                accum_iter=Config.P2_ACCUM_ITER,
            )
            val_loss, val_acc = evaluate(model_ema.ema_model, val_loader, Config.DEVICE)
            logger.info(f"Fold {fold} P2 Epoch {epoch}: Val Acc {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model_ema.ema_model.state_dict(),
                    is_best=True,
                    output_dir=fold_save_dir,
                    filename="best_model.pth",
                )

        # ==========================
        # Inference for OOF
        # ==========================
        logger.info(
            f"Fold {fold} Completed. Best Acc: {best_acc:.4f}. Generating OOF predictions..."
        )

        # Load best model
        best_model_path = os.path.join(fold_save_dir, "model_best.pth")
        load_checkpoint(model, best_model_path, Config.DEVICE)

        # Predict
        preds = inference(model, val_loader, Config.DEVICE, tta=Config.TTA_FLIP)

        # Collect Targets and IDs
        targets = []
        for _, t in val_loader:
            targets.append(t)
        targets = torch.cat(targets).cpu()

        # val_loader.dataset is CassavaDataset, which has .df
        ids = val_loader.dataset.df["image_id"].tolist()

        oof_preds.append(preds)
        oof_targets.append(targets)
        oof_image_ids.extend(ids)

    # 3. Global Evaluation
    all_preds = torch.cat(oof_preds)
    all_targets = torch.cat(oof_targets)

    acc_res = accuracy(all_preds, all_targets, topk=(1,))
    final_metric = acc_res[0].item()

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    results_df = pd.DataFrame(
        {
            "image_id": oof_image_ids,
            "target": all_targets.numpy(),
            "pred_prob_true": [p[t].item() for p, t in zip(all_preds, all_targets)],
            "pred_label": all_preds.argmax(dim=1).numpy(),
        }
    )

    # Calculate Error Magnitude (1 - probability of correct class)
    results_df["error_magnitude"] = 1.0 - results_df["pred_prob_true"]

    # Helper to get file size
    def get_file_size(img_id):
        path = os.path.join(Config.INPUT_DIR, "train_images", img_id)
        try:
            return os.path.getsize(path)
        except:
            return 0

    results_df["file_size"] = results_df["image_id"].apply(get_file_size)

    # Calculate Correlation
    corr = results_df["error_magnitude"].corr(results_df["file_size"])
    print(f"Correlation between Error Magnitude and File Size: {corr:.4f}")

    # 5. Submission
    if final_metric > 0.9076:
        logger.info("Validation metric passed threshold. Generating Submission...")

        test_loader = get_test_loader(Config.P2_IMG_SIZE, Config.P2_BATCH_SIZE)
        fold_preds = []

        for fold in range(Config.N_FOLDS):
            model = get_model(pretrained=False)
            path = os.path.join(Config.OUTPUT_DIR, f"fold_{fold}", "model_best.pth")
            load_checkpoint(model, path, Config.DEVICE)

            logger.info(f"Predicting Test Set with Fold {fold}...")
            preds = inference(model, test_loader, Config.DEVICE, tta=Config.TTA_FLIP)
            fold_preds.append(preds)

        # Average predictions
        avg_preds = torch.stack(fold_preds).mean(dim=0)
        final_labels = avg_preds.argmax(dim=1).cpu().numpy()

        # Create Submission DataFrame
        sub_df = pd.read_csv(Config.TEST_METADATA_PATH)
        sub_df["label"] = final_labels

        # Ensure correct format
        sub_df = sub_df[["image_id", "label"]]
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        logger.info(
            f"Validation metric {final_metric:.4f} <= 0.9076. Submission skipped."
        )


if __name__ == "__main__":
    main()
