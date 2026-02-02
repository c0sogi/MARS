import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from timm.data import Mixup
from timm.scheduler import CosineLRScheduler
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint
from library.dataset import CassavaDataset
from library.transforms import get_transforms
from library.model import get_model
from library.engine import train_one_epoch, validate, inference


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    Config.setup_directories()
    logger = get_logger("run.log")

    # Override Config for Fast Baseline Execution while maintaining quality
    # We use full data but fewer epochs to fit in ~2 hours
    Config.PHASE1_EPOCHS = 5
    Config.PHASE2_EPOCHS = 3
    Config.BATCH_SIZE = 32  # Ensure this fits in A100 (40GB) with 384x384

    logger.info(f"Starting execution with {Config.NUM_FOLDS} folds.")
    logger.info(
        f"Phase 1: {Config.PHASE1_EPOCHS} epochs @ {Config.PHASE1_IMG_SIZE}x{Config.PHASE1_IMG_SIZE}"
    )
    logger.info(
        f"Phase 2: {Config.PHASE2_EPOCHS} epochs @ {Config.PHASE2_IMG_SIZE}x{Config.PHASE2_IMG_SIZE}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    # Load training metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Prepare Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # -------------------------------------------------------------------------
    # 3. Training Loop (5 Folds)
    # -------------------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        logger.info(f"--- Starting Fold {fold} ---")

        df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_valid = df_train_full.iloc[val_idx].reset_index(drop=True)

        # Initialize Model
        model, model_ema = get_model(
            model_name=Config.MODEL_BACKBONE,
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
            drop_path_rate=Config.DROP_PATH_RATE,
            use_ema=Config.USE_EMA,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # ==========================
        # Phase 1: Coarse Learning
        # ==========================
        logger.info(f"Fold {fold} | Phase 1 Start")

        # Datasets & Loaders
        train_dataset_p1 = CassavaDataset(
            df_train, transform=get_transforms("train", Config.PHASE1_IMG_SIZE)
        )
        valid_dataset_p1 = CassavaDataset(
            df_valid, transform=get_transforms("valid", Config.PHASE1_IMG_SIZE)
        )

        train_loader_p1 = torch.utils.data.DataLoader(
            train_dataset_p1,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader_p1 = torch.utils.data.DataLoader(
            valid_dataset_p1,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Mixup
        mixup_fn_p1 = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=Config.PHASE1_MIXUP_PROB,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=Config.LABEL_SMOOTHING,
            num_classes=Config.NUM_CLASSES,
        )

        # Loss & Scheduler
        criterion_p1 = SoftTargetCrossEntropy()
        scheduler_p1 = CosineLRScheduler(
            optimizer,
            t_initial=Config.PHASE1_EPOCHS,
            lr_min=1e-6,
            warmup_t=1,
            warmup_lr_init=1e-5,
            cycle_limit=1,
        )

        best_acc = 0.0

        for epoch in range(Config.PHASE1_EPOCHS):
            train_metrics = train_one_epoch(
                model,
                criterion_p1,
                train_loader_p1,
                optimizer,
                Config.DEVICE,
                epoch,
                model_ema,
                mixup_fn_p1,
                scheduler_p1,
            )
            scheduler_p1.step(epoch + 1)

            # Validate (using EMA if available)
            eval_model = model_ema.module if model_ema else model
            val_metrics = validate(
                eval_model, nn.CrossEntropyLoss(), valid_loader_p1, Config.DEVICE
            )

            if val_metrics["val_acc1"] > best_acc:
                best_acc = val_metrics["val_acc1"]
                save_checkpoint(
                    {"state_dict": eval_model.state_dict()}, is_best=True, fold=fold
                )

        # ==========================
        # Phase 2: Fine Tuning
        # ==========================
        logger.info(f"Fold {fold} | Phase 2 Start")

        # Datasets & Loaders (High Res)
        train_dataset_p2 = CassavaDataset(
            df_train, transform=get_transforms("train", Config.PHASE2_IMG_SIZE)
        )
        valid_dataset_p2 = CassavaDataset(
            df_valid, transform=get_transforms("valid", Config.PHASE2_IMG_SIZE)
        )

        train_loader_p2 = torch.utils.data.DataLoader(
            train_dataset_p2,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader_p2 = torch.utils.data.DataLoader(
            valid_dataset_p2,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # No Mixup in Phase 2
        mixup_fn_p2 = None

        # Loss: Label Smoothing
        criterion_p2 = LabelSmoothingCrossEntropy(smoothing=Config.LABEL_SMOOTHING)

        # Scheduler: Reset for Phase 2
        scheduler_p2 = CosineLRScheduler(
            optimizer,
            t_initial=Config.PHASE2_EPOCHS,
            lr_min=1e-7,
            warmup_t=0,
            cycle_limit=1,
        )

        for epoch in range(Config.PHASE2_EPOCHS):
            # Adjust epoch index for logging
            global_epoch = Config.PHASE1_EPOCHS + epoch

            train_metrics = train_one_epoch(
                model,
                criterion_p2,
                train_loader_p2,
                optimizer,
                Config.DEVICE,
                global_epoch,
                model_ema,
                mixup_fn_p2,
                scheduler_p2,
            )
            scheduler_p2.step(epoch + 1)

            # Validate
            eval_model = model_ema.module if model_ema else model
            val_metrics = validate(
                eval_model, nn.CrossEntropyLoss(), valid_loader_p2, Config.DEVICE
            )

            if val_metrics["val_acc1"] > best_acc:
                best_acc = val_metrics["val_acc1"]
                save_checkpoint(
                    {"state_dict": eval_model.state_dict()}, is_best=True, fold=fold
                )

        # Cleanup
        del (
            model,
            model_ema,
            optimizer,
            train_loader_p1,
            valid_loader_p1,
            train_loader_p2,
            valid_loader_p2,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. Ensemble Validation on Hold-out Set
    # -------------------------------------------------------------------------
    logger.info("--- Starting Ensemble Validation ---")

    df_val_holdout = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = CassavaDataset(
        df_val_holdout, transform=get_transforms("valid", Config.PHASE2_IMG_SIZE)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize arrays for ensemble predictions
    ensemble_probs = torch.zeros(
        (len(df_val_holdout), Config.NUM_CLASSES), device=Config.DEVICE
    )

    # Iterate over folds and aggregate predictions
    for fold in range(Config.NUM_FOLDS):
        logger.info(f"Loading Fold {fold} model...")
        model, _ = get_model(
            model_name=Config.MODEL_BACKBONE,
            num_classes=Config.NUM_CLASSES,
            pretrained=False,
            use_ema=False,
            checkpoint_path=os.path.join(
                Config.WORKING_DIR, f"best_model_fold_{fold}.pth"
            ),
        )
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(Config.DEVICE)

                # Standard Inference
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                # TTA (Flip)
                if Config.TTA_FLIP:
                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = model(images_flip)
                    probs_flip = torch.softmax(logits_flip, dim=1)
                    probs = (probs + probs_flip) / 2.0

                fold_probs.append(probs)

        fold_probs = torch.cat(fold_probs, dim=0)
        ensemble_probs += fold_probs

        del model
        torch.cuda.empty_cache()

    # Average probabilities
    ensemble_probs /= Config.NUM_FOLDS
    final_preds = torch.argmax(ensemble_probs, dim=1).cpu().numpy()
    ground_truth = df_val_holdout["label"].values

    # Calculate Metric
    accuracy = np.mean(final_preds == ground_truth)
    print(f"Final Validation Metric: {accuracy}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("--- Performing Failure Analysis ---")

    # Calculate error magnitude (1 - probability of correct class)
    correct_class_probs = (
        ensemble_probs[
            torch.arange(len(ground_truth)),
            torch.tensor(ground_truth, device=Config.DEVICE),
        ]
        .cpu()
        .numpy()
    )
    error_magnitude = 1.0 - correct_class_probs

    # Get file sizes
    file_sizes = []
    for rel_path in df_val_holdout["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        file_sizes.append(os.path.getsize(full_path))
    file_sizes = np.array(file_sizes)

    # Correlation
    correlation = np.corrcoef(error_magnitude, file_sizes)[0, 1]
    print(f"Correlation between Error Magnitude and File Size: {correlation:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    if accuracy > 0.9076:
        logger.info("Validation metric met threshold. Generating submission...")

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = CassavaDataset(
            df_test,
            transform=get_transforms("inference", Config.PHASE2_IMG_SIZE),
            return_id=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ensemble_probs = torch.zeros(
            (len(df_test), Config.NUM_CLASSES), device=Config.DEVICE
        )
        image_ids = []

        for fold in range(Config.NUM_FOLDS):
            model, _ = get_model(
                model_name=Config.MODEL_BACKBONE,
                num_classes=Config.NUM_CLASSES,
                pretrained=False,
                use_ema=False,
                checkpoint_path=os.path.join(
                    Config.WORKING_DIR, f"best_model_fold_{fold}.pth"
                ),
            )
            model.eval()

            fold_probs = []
            current_image_ids = []

            with torch.no_grad():
                for batch in test_loader:
                    images, _, batch_ids = batch
                    images = images.to(Config.DEVICE)

                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)

                    if Config.TTA_FLIP:
                        images_flip = torch.flip(images, dims=[3])
                        logits_flip = model(images_flip)
                        probs_flip = torch.softmax(logits_flip, dim=1)
                        probs = (probs + probs_flip) / 2.0

                    fold_probs.append(probs)
                    if fold == 0:
                        current_image_ids.extend(batch_ids)

            test_ensemble_probs += torch.cat(fold_probs, dim=0)
            if fold == 0:
                image_ids = current_image_ids

            del model
            torch.cuda.empty_cache()

        test_ensemble_probs /= Config.NUM_FOLDS
        test_preds = torch.argmax(test_ensemble_probs, dim=1).cpu().numpy()

        # Create submission DataFrame
        submission_df = pd.DataFrame({"image_id": image_ids, "label": test_preds})

        submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        logger.info(
            f"Validation metric {accuracy} did not meet threshold 0.9076. Skipping submission."
        )


if __name__ == "__main__":
    main()
