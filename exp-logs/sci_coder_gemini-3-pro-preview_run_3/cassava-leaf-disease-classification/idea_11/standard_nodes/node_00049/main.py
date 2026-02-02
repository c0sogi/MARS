import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from timm.loss import SoftTargetCrossEntropy

# Ensure local library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_manager import (
    get_dataloaders,
    get_test_dataloader,
    process_data,
    CassavaDataset,
    get_transforms,
)
from library.model_factory import get_model
from library.training_engine import train_one_epoch, valid_one_epoch, get_mixup_fn
from library.meta_learner import (
    fit_meta_learner,
    generate_submission,
    predict_meta_learner,
)


def run():
    # 1. Configuration & Setup
    Config.EPOCHS = 1  # Override for fast baseline execution
    seed_everything(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = get_logger(
        "Cassava_Pipeline", os.path.join(Config.OUTPUT_DIR, "pipeline.log")
    )

    logger.info("Starting End-to-End Stacking Pipeline...")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Architectures: {Config.MODEL_ARCHS}")

    # Ensure data is processed and folds assigned
    # This creates/loads 'train_with_folds.parquet'
    process_data(load_cached_data=True)

    # Reload dataframe to ensure we have the exact index alignment used by dataloaders
    train_df_with_folds = pd.read_parquet(
        os.path.join(Config.OUTPUT_DIR, "train_with_folds.parquet")
    )

    # Storage for Level 1 Training Data (OOF Predictions)
    # Structure: Dictionary { arch_name: np.array(N_samples, N_classes) }
    oof_preds_by_arch = {}

    # 2. Level 0: Train Base Models (Heterogeneous Stacking)
    for arch in Config.MODEL_ARCHS:
        logger.info(f"=== Processing Architecture: {arch} ===")

        # Initialize OOF array for this architecture
        num_train_samples = len(train_df_with_folds)
        arch_oof = np.zeros((num_train_samples, Config.NUM_CLASSES), dtype=np.float32)

        for fold in range(Config.N_FOLDS):
            logger.info(f"  [Fold {fold}/{Config.N_FOLDS - 1}] Training...")

            # Get DataLoaders for this fold
            train_loader, val_loader = get_dataloaders(
                fold_id=fold, load_cached_data=True
            )

            # Initialize Model
            model = get_model(arch, pretrained=True)
            model.to(Config.DEVICE)

            # Optimization Setup
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
            )
            scaler = torch.cuda.amp.GradScaler()

            # Loss Functions
            # SoftTargetCrossEntropy for MixUp training
            loss_fn_train = SoftTargetCrossEntropy()
            # Standard CrossEntropy for validation/OOF
            loss_fn_val = torch.nn.CrossEntropyLoss()

            mixup_fn = get_mixup_fn()

            # Training Loop
            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    epoch,
                    model,
                    train_loader,
                    optimizer,
                    Config.DEVICE,
                    loss_fn_train,
                    scaler,
                    mixup_fn,
                )

            # Validation & OOF Generation
            logger.info(f"  [Fold {fold}] Generating OOF predictions...")
            val_loss, val_acc, preds, targets = valid_one_epoch(
                0, model, val_loader, Config.DEVICE, loss_fn_val
            )

            # Align OOF predictions with the original dataframe
            # We identify the indices corresponding to this fold
            val_indices = train_df_with_folds[
                train_df_with_folds["fold"] == fold
            ].index.values

            # Handle potential size mismatch (though unlikely with proper logic)
            if len(val_indices) != len(preds):
                logger.warning(
                    f"  Size mismatch: Indices {len(val_indices)} vs Preds {len(preds)}. Adjusting."
                )
                min_len = min(len(val_indices), len(preds))
                val_indices = val_indices[:min_len]
                preds = preds[:min_len]

            # Store predictions
            arch_oof[val_indices] = preds

            # Save Model Checkpoint
            model_path = os.path.join(Config.OUTPUT_DIR, f"{arch}_fold{fold}.pth")
            torch.save(model.state_dict(), model_path)

            # Cleanup to save memory
            del model, optimizer, scaler, train_loader, val_loader
            torch.cuda.empty_cache()

        oof_preds_by_arch[arch] = arch_oof

    # 3. Level 1: Train Meta-Learner
    logger.info("=== Training Meta-Learner ===")
    # Prepare input features: Concatenate OOF probs from all architectures
    # Order must be consistent: [Arch1, Arch2, Arch3]
    oof_list = [oof_preds_by_arch[arch] for arch in Config.MODEL_ARCHS]
    train_targets = train_df_with_folds["label"].values

    meta_model = fit_meta_learner(oof_list, train_targets)

    # 4. Hold-out Validation (Final Metric Calculation)
    logger.info("=== Performing Hold-out Validation ===")

    # Load Hold-out Dataset
    val_meta_df = pd.read_csv(Config.VAL_METADATA)
    val_ds = CassavaDataset(
        val_meta_df, transforms=get_transforms("valid"), output_label=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate predictions on Hold-out set using the full ensemble
    holdout_preds_by_arch = []

    for arch in Config.MODEL_ARCHS:
        logger.info(f"  Predicting Hold-out with {arch}...")
        arch_preds = np.zeros((len(val_meta_df), Config.NUM_CLASSES), dtype=np.float32)

        # Bagging: Average predictions across all 5 folds
        for fold in range(Config.N_FOLDS):
            model = get_model(arch, pretrained=False)
            model_path = os.path.join(Config.OUTPUT_DIR, f"{arch}_fold{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)
            model.eval()

            fold_preds_list = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(Config.DEVICE)
                    out = model(imgs)
                    fold_preds_list.append(torch.softmax(out, dim=1).cpu().numpy())

            fold_preds = np.concatenate(fold_preds_list)
            arch_preds += fold_preds

            del model
            torch.cuda.empty_cache()

        arch_preds /= Config.N_FOLDS
        holdout_preds_by_arch.append(arch_preds)

    # Meta-Learner Prediction on Hold-out
    final_val_preds, final_val_probs = predict_meta_learner(
        holdout_preds_by_arch, meta_model
    )
    final_val_targets = val_meta_df["label"].values

    final_metric = accuracy_score(final_val_targets, final_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("=== Performing Failure Analysis ===")

    # Confusion Matrix
    cm = confusion_matrix(final_val_targets, final_val_preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Per-class Accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        logger.info(f"Class {i} Accuracy: {acc:.4f}")

    # Correlation between Error and Label (Systematic Bias)
    errors = (final_val_preds != final_val_targets).astype(int)
    corr_label = np.corrcoef(errors, final_val_targets)[0, 1]
    print(f"Correlation between Error and Label: {corr_label:.4f}")

    # 6. Submission Logic
    threshold = 0.9086782376502003
    if final_metric > threshold:
        logger.info(f"Metric {final_metric} > {threshold}. Generating Submission...")

        test_loader = get_test_dataloader()
        test_preds_by_arch = []

        for arch in Config.MODEL_ARCHS:
            logger.info(f"  Predicting Test with {arch}...")
            n_test = len(test_loader.dataset)
            arch_preds = np.zeros((n_test, Config.NUM_CLASSES), dtype=np.float32)

            for fold in range(Config.N_FOLDS):
                model = get_model(arch, pretrained=False)
                model_path = os.path.join(Config.OUTPUT_DIR, f"{arch}_fold{fold}.pth")
                model.load_state_dict(
                    torch.load(model_path, map_location=Config.DEVICE)
                )
                model.to(Config.DEVICE)
                model.eval()

                fold_preds_list = []
                with torch.no_grad():
                    for imgs, _ in test_loader:
                        imgs = imgs.to(Config.DEVICE)
                        out = model(imgs)
                        fold_preds_list.append(torch.softmax(out, dim=1).cpu().numpy())

                fold_preds = np.concatenate(fold_preds_list)
                arch_preds += fold_preds

                del model
                torch.cuda.empty_cache()

            arch_preds /= Config.N_FOLDS
            test_preds_by_arch.append(arch_preds)

        # Generate Submission File
        image_ids = pd.read_csv(Config.TEST_METADATA)["image_id"].values
        generate_submission(test_preds_by_arch, meta_model, image_ids)

    else:
        logger.info(
            f"Metric {final_metric} did not pass threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
