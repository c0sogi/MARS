import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data import load_and_cache_split, CactusDataset
from library.models import CactusRepVGG, CactusResNet, CactusNeXt

logger = get_logger(name="inference")


def get_model_instance(arch_name, device):
    """
    Factory function to instantiate models based on architecture name.
    """
    if arch_name == "RepVGG_FiLM":
        # Initialize with deploy=False to load training weights correctly, then switch later
        model = CactusRepVGG(num_classes=Config.NUM_CLASSES, deploy=False)
    elif arch_name == "ResNet_FiLM":
        model = CactusResNet(num_classes=Config.NUM_CLASSES)
    elif arch_name == "NeXt_FiLM":
        model = CactusNeXt(num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model.to(device)


def load_checkpoint_weights(model, filepath, device):
    """
    Loads weights from a checkpoint file. Handles state_dict wrapping.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Checkpoint not found: {filepath}")
        return False

    logger.info(f"Loading weights from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    state_dict = None
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return True


def apply_tta_views(images):
    """
    Generates 4 TTA views for a batch of images:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. 180 Rotation (H + V Flip)
    """
    # images: (B, C, H, W)
    v1 = images
    v2 = torch.flip(images, dims=[3])  # H-Flip
    v3 = torch.flip(images, dims=[2])  # V-Flip
    v4 = torch.flip(images, dims=[2, 3])  # 180 Rot

    return [v1, v2, v3, v4]


def predict_loader(model, loader, device):
    """
    Runs inference on a DataLoader using 4-view TTA.
    Returns raw probabilities (after sigmoid).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _, fsizes in loader:
            images = images.to(device, non_blocking=True)
            fsizes = fsizes.to(device, non_blocking=True)

            # Ensure correct shape for model input
            fsizes = fsizes.view(-1, 1)

            # Get TTA views
            views = apply_tta_views(images)

            batch_preds = []
            for view in views:
                logits = model(view, fsizes)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs)

            # Stack and average across views: (4, B, 1) -> (B, 1)
            batch_preds = torch.stack(batch_preds)
            avg_preds = torch.mean(batch_preds, dim=0)

            all_preds.append(avg_preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0).flatten()


def prepare_data_for_inference(debug=False):
    """
    Loads Train (for OOF) and Test data.
    """
    # 1. Load Train Metadata
    t_imgs, t_labels, t_fs, _ = load_and_cache_split(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        Config.CACHE_TRAIN_FILESIZES,
        Config.INPUT_DIR,
        load_cached=True,
    )

    # 2. Load Test Data
    test_labels_cache = Config.CACHE_TEST_IDS.replace("ids.npy", "labels.npy")
    test_imgs, _, test_fs, test_ids = load_and_cache_split(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_IMGS,
        test_labels_cache,
        Config.CACHE_TEST_FILESIZES,
        Config.INPUT_DIR,
        load_cached=True,
    )

    if debug:
        logger.info("DEBUG mode: Truncating datasets.")
        t_imgs = t_imgs[:200]
        t_labels = t_labels[:200]
        t_fs = t_fs[:200]
        test_imgs = test_imgs[:200]
        test_fs = test_fs[:200]
        test_ids = test_ids[:200]

    # 3. Normalize File Sizes (Global Stats from Train)
    fs_mean = np.mean(t_fs)
    fs_std = np.std(t_fs) + 1e-8
    logger.info(
        f"Global File Size Stats (Train) - Mean: {fs_mean:.4f}, Std: {fs_std:.4f}"
    )

    train_fs_norm = (t_fs - fs_mean) / fs_std
    test_fs_norm = (test_fs - fs_mean) / fs_std

    return (
        t_imgs,
        t_labels,
        train_fs_norm,
        test_imgs,
        test_ids,
        test_fs_norm,
    )


def run_inference(debug=False):
    """
    Main inference pipeline:
    1. Generate OOF predictions for Meta-Learner training.
    2. Generate Test predictions using Base Models.
    3. Train Meta-Learner and predict final Test probabilities.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Loading data for inference...")
    (X_train, y_train, fs_train, X_test, test_ids, fs_test) = (
        prepare_data_for_inference(debug)
    )

    # Create Test Loader (used repeatedly)
    # Dummy labels for test dataset
    dummy_test_labels = np.zeros(len(X_test), dtype=np.float32)
    test_dataset = CactusDataset(X_test, dummy_test_labels, fs_test, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 2. Base Model Inference (OOF & Test)
    # -------------------------------------------------------------------------
    # Containers for predictions
    # OOF: (N_train_samples, N_archs)
    oof_preds_matrix = np.zeros((len(X_train), len(Config.MODEL_ARCHS)))
    # Test: (N_test_samples, N_archs) - averaged over folds
    test_preds_matrix = np.zeros((len(X_test), len(Config.MODEL_ARCHS)))

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for i, arch_name in enumerate(Config.MODEL_ARCHS):
        logger.info(f"Processing Architecture: {arch_name}")

        arch_test_preds_accum = np.zeros(len(X_test))

        # Iterate Folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"  Fold {fold}...")

            # Define Checkpoint Paths (Prioritize SWA)
            swa_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{arch_name}_fold{fold}_swa.pth"
            )
            best_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{arch_name}_fold{fold}_best.pth"
            )

            checkpoint_path = swa_path if os.path.exists(swa_path) else best_path

            # Load Model
            model = get_model_instance(arch_name, device)
            loaded = load_checkpoint_weights(model, checkpoint_path, device)

            if not loaded:
                logger.error(
                    f"    No checkpoint found for {arch_name} Fold {fold}. Skipping."
                )
                continue

            # Switch RepVGG to deploy mode (fuse layers)
            if hasattr(model, "switch_to_deploy"):
                model.switch_to_deploy()

            # --- OOF Prediction (Validation Chunk) ---
            # Create Val Loader for this fold
            val_dataset = CactusDataset(
                X_train[val_idx], y_train[val_idx], fs_train[val_idx], transform=None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            val_preds = predict_loader(model, val_loader, device)
            oof_preds_matrix[val_idx, i] = val_preds

            # --- Test Prediction ---
            fold_test_preds = predict_loader(model, test_loader, device)
            arch_test_preds_accum += fold_test_preds

            # Clean up
            del model, val_loader, val_dataset
            torch.cuda.empty_cache()

        # Average Test Preds for this architecture
        test_preds_matrix[:, i] = arch_test_preds_accum / Config.N_FOLDS

        # Evaluate OOF AUC for this architecture
        arch_auc = roc_auc_score(y_train, oof_preds_matrix[:, i])
        logger.info(f"  Architecture {arch_name} OOF AUC: {arch_auc:.6f}")

    # -------------------------------------------------------------------------
    # 3. Stacking (Meta-Learner)
    # -------------------------------------------------------------------------
    logger.info("Training Meta-Learner (Logistic Regression)...")

    # Check for any zero columns (failed loads)
    valid_archs = ~np.all(oof_preds_matrix == 0, axis=0)
    if not np.any(valid_archs):
        raise RuntimeError("No valid predictions generated. Check model paths.")

    X_meta_train = oof_preds_matrix[:, valid_archs]
    X_meta_test = test_preds_matrix[:, valid_archs]

    meta_model = LogisticRegression(random_state=Config.SEED, solver="liblinear")
    meta_model.fit(X_meta_train, y_train)

    # Evaluate Meta-Learner OOF
    meta_oof_preds = meta_model.predict_proba(X_meta_train)[:, 1]
    meta_auc = roc_auc_score(y_train, meta_oof_preds)
    logger.info(f"Meta-Learner OOF AUC: {meta_auc:.6f}")
    logger.info(f"Meta-Learner Coefficients: {meta_model.coef_}")

    # -------------------------------------------------------------------------
    # 4. Final Submission
    # -------------------------------------------------------------------------
    logger.info("Generating final submission...")
    final_test_preds = meta_model.predict_proba(X_meta_test)[:, 1]

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_test_preds})

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
