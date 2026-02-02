import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import (
    load_and_cache_split,
    CactusDataset,
    MixupCollate,
    get_transforms,
)
from library.models import CactusRepVGG, CactusResNet, CactusNeXt
from library.engine import train_one_epoch, validate, SWAHandler, save_checkpoint

logger = get_logger(name="train")


def get_model_instance(arch_name, device):
    """
    Factory function to instantiate models based on architecture name.
    """
    if arch_name == "RepVGG_FiLM":
        model = CactusRepVGG(num_classes=Config.NUM_CLASSES, deploy=False)
    elif arch_name == "ResNet_FiLM":
        model = CactusResNet(num_classes=Config.NUM_CLASSES)
    elif arch_name == "NeXt_FiLM":
        model = CactusNeXt(num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model.to(device)


def prepare_full_data():
    """
    Loads both train and validation splits from metadata and concatenates them
    to allow for proper N-Fold Cross Validation.
    """
    # Load Train Split
    t_imgs, t_labels, t_fs, _ = load_and_cache_split(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        Config.CACHE_TRAIN_FILESIZES,
        Config.INPUT_DIR,
        load_cached=True,
    )

    # Load Val Split
    v_imgs, v_labels, v_fs, _ = load_and_cache_split(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_IMGS,
        Config.CACHE_VAL_LABELS,
        Config.CACHE_VAL_FILESIZES,
        Config.INPUT_DIR,
        load_cached=True,
    )

    # Concatenate
    full_imgs = np.concatenate([t_imgs, v_imgs], axis=0)
    full_labels = np.concatenate([t_labels, v_labels], axis=0)
    full_fs = np.concatenate([t_fs, v_fs], axis=0)

    return full_imgs, full_labels, full_fs


def run_training(debug=False):
    """
    Main driver for 5-Fold Stratified Cross-Validation training.
    """
    seed_everything(Config.SEED)

    # 1. Prepare Data
    logger.info("Loading and preparing full dataset for Cross-Validation...")
    images, labels, filesizes = prepare_full_data()

    if debug:
        logger.info("DEBUG mode: Truncating dataset to 500 samples.")
        images = images[:500]
        labels = labels[:500]
        filesizes = filesizes[:500]

    # Normalize File Sizes (Z-score) globally
    fs_mean = np.mean(filesizes)
    fs_std = np.std(filesizes) + 1e-8
    logger.info(f"Global File Size Stats - Mean: {fs_mean}, Std: {fs_std}")

    filesizes_norm = (filesizes - fs_mean) / fs_std

    # 2. Setup Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # 3. Iterate Architectures
    for arch_name in Config.MODEL_ARCHS:
        logger.info(
            f"================ Starting Training for Architecture: {arch_name} ================"
        )

        # 4. Iterate Folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            logger.info(f"--- Fold {fold} / {Config.N_FOLDS - 1} ---")

            # Split Data
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            fs_train, fs_val = filesizes_norm[train_idx], filesizes_norm[val_idx]

            # Create Datasets
            train_dataset = CactusDataset(
                X_train, y_train, fs_train, transform=get_transforms("train")
            )
            val_dataset = CactusDataset(
                X_val, y_val, fs_val, transform=get_transforms("val")
            )

            # Create Loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                collate_fn=MixupCollate(alpha=Config.MIXUP_ALPHA),
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            device = torch.device(Config.DEVICE)
            model = get_model_instance(arch_name, device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # SWA Handler
            swa_handler = SWAHandler(model, optimizer, Config)

            # Training Loop Variables
            best_auc = 0.0
            patience = 8
            patience_counter = 0

            for epoch in range(Config.EPOCHS):
                # Train
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device, epoch, scheduler
                )

                # Validate
                val_loss, val_auc = validate(model, val_loader, device)

                # SWA Step
                swa_handler.step(epoch, model)

                logger.info(
                    f"Epoch {epoch+1}/{Config.EPOCHS} | "
                    f"Train Loss: {train_loss} | "
                    f"Val Loss: {val_loss} | "
                    f"Val AUC: {val_auc}"
                )

                # Checkpointing (Best Model)
                is_best = val_auc > best_auc
                if is_best:
                    best_auc = val_auc
                    patience_counter = 0
                    checkpoint_path = os.path.join(
                        Config.CHECKPOINT_DIR, f"{arch_name}_fold{fold}_best.pth"
                    )
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_auc": best_auc,
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best=True,
                        filepath=checkpoint_path,
                    )
                else:
                    patience_counter += 1

                # Early Stopping (only if SWA is not active or if we are way past SWA start)
                # We want to ensure SWA gets enough epochs.
                if patience_counter >= patience:
                    # If SWA is enabled, we should ensure we don't stop before SWA has collected samples
                    if Config.USE_SWA and epoch < (Config.SWA_START_EPOCH + 2):
                        pass  # Continue training to allow SWA
                    else:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

            # Finalize SWA
            if Config.USE_SWA and swa_handler.swa_model is not None:
                logger.info("Finalizing SWA model...")
                swa_handler.update_bn(train_loader)
                swa_model = swa_handler.get_model()

                # Validate SWA Model
                swa_loss, swa_auc = validate(swa_model, val_loader, device)
                logger.info(f"SWA Final Results - Loss: {swa_loss} | AUC: {swa_auc}")

                swa_path = os.path.join(
                    Config.CHECKPOINT_DIR, f"{arch_name}_fold{fold}_swa.pth"
                )
                save_checkpoint(
                    {"state_dict": swa_model.state_dict(), "auc": swa_auc},
                    is_best=False,
                    filepath=swa_path,
                )

            # Clean up
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
