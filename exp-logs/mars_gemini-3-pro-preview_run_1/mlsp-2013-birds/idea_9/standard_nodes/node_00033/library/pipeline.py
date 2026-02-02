import os
import copy
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
import pandas as pd

from library.config import (
    WORKING_DIR,
    TEACHER_WIDTHS,
    STUDENT_WIDTH,
    IMG_HEIGHT,
    TEACHER_EPOCHS,
    STUDENT_EPOCHS,
    SWA_START_EPOCH,
    SWA_LR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MIXUP_ALPHA,
    AUGMENT,
    DEVICE,
    SEED,
    NUM_CLASSES,
)
from library.utils import set_seed, save_checkpoint, load_checkpoint, compute_roc_auc
from library.data import load_metadata, process_and_cache_data, get_loader
from library.model import get_bird_model
from library.engine import train_one_epoch, evaluate


def train_teachers(debug=False):
    """
    Stage 1: Train multiple teacher models at different resolutions.

    Args:
        debug (bool): If True, runs a shortened training loop for debugging.

    Returns:
        list: Paths to the saved best teacher models.
    """
    set_seed(SEED)

    teacher_paths = []

    # Load metadata
    df_train = load_metadata("train")
    df_val = load_metadata("val")

    epochs = 2 if debug else TEACHER_EPOCHS

    for i, width in enumerate(TEACHER_WIDTHS):
        print(
            f"\n--- Training Teacher {i+1}/{len(TEACHER_WIDTHS)} at Resolution {IMG_HEIGHT}x{width} ---"
        )

        # Prepare Data
        # Note: process_and_cache_data handles caching internally
        train_images, train_labels, train_ids = process_and_cache_data(
            df_train, "train", width, IMG_HEIGHT, load_cached_data=True
        )
        val_images, val_labels, val_ids = process_and_cache_data(
            df_val, "val", width, IMG_HEIGHT, load_cached_data=True
        )

        # Debug Slicing
        if debug:
            subset_size = 20
            train_images = train_images[:subset_size]
            train_labels = train_labels[:subset_size]
            train_ids = train_ids[:subset_size]
            val_images = val_images[:subset_size]
            val_labels = val_labels[:subset_size]
            val_ids = val_ids[:subset_size]

        train_loader = get_loader(
            train_images,
            train_labels,
            train_ids,
            BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            augment=AUGMENT,
        )
        val_loader = get_loader(
            val_images,
            val_labels,
            val_ids,
            BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            augment=False,
        )

        # Model & Optimizer
        model = get_bird_model(pretrained=True).to(DEVICE)
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        # Use index in filename to ensure unique files for ensemble members with same width
        best_model_path = os.path.join(WORKING_DIR, f"teacher_{i}_{width}.pth")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, DEVICE, epoch, mixup_alpha=MIXUP_ALPHA
            )
            scheduler.step()

            val_loss, val_auc, _, _ = evaluate(model, val_loader, DEVICE)

            print(
                f"Epoch {epoch+1}/{epochs} [Width {width}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.16f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, best_model_path)

        teacher_paths.append(best_model_path)
        print(f"Best Teacher {width} saved with AUC: {best_auc:.16f}")

    return teacher_paths


def generate_ensemble_pseudo_labels(teacher_paths, debug=False):
    """
    Stage 2: Generate pseudo-labels for the test set using the ensemble of teachers.

    Args:
        teacher_paths (list): List of file paths to the trained teacher models.
        debug (bool): If True, processes a subset of data.

    Returns:
        tuple: (test_ids, ensemble_probs)
    """
    set_seed(SEED)
    print("\n--- Generating Ensemble Pseudo-Labels ---")

    df_test = load_metadata("test")

    ensemble_probs = None
    test_ids = None

    for i, (width, model_path) in enumerate(zip(TEACHER_WIDTHS, teacher_paths)):
        print(f"Inference with Teacher {i} ({width}) from {model_path}...")

        # Load test data at specific resolution
        test_images, test_labels, ids = process_and_cache_data(
            df_test, "test", width, IMG_HEIGHT, load_cached_data=True
        )

        if debug:
            subset_size = 20
            test_images = test_images[:subset_size]
            test_labels = test_labels[:subset_size]
            ids = ids[:subset_size]

        test_loader = get_loader(
            test_images, test_labels, ids, BATCH_SIZE, shuffle=False, augment=False
        )

        # Load Model
        model = get_bird_model(pretrained=False).to(DEVICE)
        load_checkpoint(model, model_path, DEVICE)

        # Evaluate
        _, _, probs, _ = evaluate(model, test_loader, DEVICE)

        if ensemble_probs is None:
            ensemble_probs = probs
            test_ids = ids
        else:
            ensemble_probs += probs

    # Average probabilities
    if ensemble_probs is not None:
        ensemble_probs /= len(TEACHER_WIDTHS)

    # Sanitize
    if ensemble_probs is not None and np.isnan(ensemble_probs).any():
        print("Warning: NaNs detected in pseudo-labels. Replacing with zeros.")
        ensemble_probs = np.nan_to_num(ensemble_probs)

    return test_ids, ensemble_probs


def train_student_with_swa(test_ids, pseudo_labels, debug=False):
    """
    Stage 3: Train student model on combined data with SWA.

    Args:
        test_ids (np.ndarray): IDs of the test set samples.
        pseudo_labels (np.ndarray): Predicted probabilities for the test set.
        debug (bool): Debug flag.

    Returns:
        torch.nn.Module: The trained SWA student model.
    """
    set_seed(SEED)
    print(
        f"\n--- Training Student at Resolution {IMG_HEIGHT}x{STUDENT_WIDTH} with SWA ---"
    )

    # Load Metadata
    df_train = load_metadata("train")
    df_val = load_metadata("val")
    df_test = load_metadata("test")

    # 1. Load Labeled Train Data
    train_images, train_labels, train_ids = process_and_cache_data(
        df_train, "train", STUDENT_WIDTH, IMG_HEIGHT, load_cached_data=True
    )

    # 2. Load Unlabeled Test Data (Images only)
    # We load images at STUDENT_WIDTH.
    # Note: We assume df_test order matches test_ids/pseudo_labels.
    # Since both come from the same source file and process_and_cache_data is deterministic, this holds.
    test_images_raw, _, test_ids_raw = process_and_cache_data(
        df_test, "test", STUDENT_WIDTH, IMG_HEIGHT, load_cached_data=True
    )

    if debug:
        subset_size = 20
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
        train_ids = train_ids[:subset_size]

        test_images_raw = test_images_raw[:subset_size]
        pseudo_labels = pseudo_labels[:subset_size]
        test_ids_raw = test_ids_raw[:subset_size]

        epochs = 3
        swa_start = 1
    else:
        epochs = STUDENT_EPOCHS
        swa_start = SWA_START_EPOCH

    # Combine Data
    if isinstance(pseudo_labels, np.ndarray):
        pseudo_labels = torch.from_numpy(pseudo_labels).float()

    combined_images = torch.cat([train_images, test_images_raw], dim=0)
    combined_labels = torch.cat([train_labels, pseudo_labels], dim=0)
    combined_ids = np.concatenate([train_ids, test_ids_raw], axis=0)

    print(f"Combined Dataset Size: {len(combined_images)}")

    # Load Validation Data
    val_images, val_labels, val_ids = process_and_cache_data(
        df_val, "val", STUDENT_WIDTH, IMG_HEIGHT, load_cached_data=True
    )
    if debug:
        subset_size = 20
        val_images = val_images[:subset_size]
        val_labels = val_labels[:subset_size]
        val_ids = val_ids[:subset_size]

    # Loaders
    train_loader = get_loader(
        combined_images,
        combined_labels,
        combined_ids,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        augment=AUGMENT,
    )
    val_loader = get_loader(
        val_images, val_labels, val_ids, BATCH_SIZE, shuffle=False, augment=False
    )

    # Student Model
    model = get_bird_model(pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # SWA Setup
    swa_model = AveragedModel(model).to(DEVICE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    student_path = os.path.join(WORKING_DIR, "student_swa.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, epoch, mixup_alpha=MIXUP_ALPHA
        )

        # SWA Logic
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr_curr = swa_scheduler.get_last_lr()[0]
            mode_str = "SWA"
        else:
            scheduler.step()
            lr_curr = scheduler.get_last_lr()[0]
            mode_str = "Standard"

        # Evaluate current model to monitor progress
        val_loss, val_auc, _, _ = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch+1}/{epochs} [{mode_str}] LR: {lr_curr:.6f} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.16f}"
        )

    # End of training: Update BN for SWA model
    print("Updating SWA Batch Normalization statistics...")
    update_bn(train_loader, swa_model, device=DEVICE)

    # Final Evaluation of SWA Model
    print("Evaluating Final SWA Student...")
    val_loss, val_auc, _, _ = evaluate(swa_model, val_loader, DEVICE)
    print(f"Final SWA Student Val AUC: {val_auc:.16f}")

    save_checkpoint(swa_model, student_path)

    return swa_model
