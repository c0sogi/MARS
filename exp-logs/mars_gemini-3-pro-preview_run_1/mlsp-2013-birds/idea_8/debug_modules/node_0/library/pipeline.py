import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, check_tensor_sanitation
from library.dataset import get_data, BirdDataset, get_transforms
from library.model import get_model
from library.swa_utils import SWAHandler, update_bn_statistics
from library.engine import train_one_epoch, validate, predict


def train_teacher_ensemble(train_data, val_data, num_teachers=None, epochs=None):
    """
    Stage 1: Train an ensemble of Teacher models.

    Args:
        train_data (tuple): (images, labels, ids) for training.
        val_data (tuple): (images, labels, ids) for validation.
        num_teachers (int): Number of models to train.
        epochs (int): Number of epochs per model.

    Returns:
        list: List of trained PyTorch models.
    """
    if num_teachers is None:
        num_teachers = Config.NUM_TEACHERS
    if epochs is None:
        epochs = Config.TEACHER_EPOCHS

    train_images, train_labels, train_ids = train_data
    val_images, val_labels, val_ids = val_data

    # Create datasets
    train_dataset = BirdDataset(
        train_images, train_labels, train_ids, transform=get_transforms("train")
    )
    val_dataset = BirdDataset(
        val_images, val_labels, val_ids, transform=get_transforms("val")
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    teachers = []
    print(f"Starting Stage 1: Training {num_teachers} Teacher Models...")

    for i in range(num_teachers):
        print(f"\n--- Training Teacher {i+1}/{num_teachers} ---")
        # Set distinct seed for diversity
        current_seed = Config.SEED + i
        set_seed(current_seed)

        model = get_model(pretrained=Config.PRETRAINED)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Cosine Annealing Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, optimizer, train_loader, Config.DEVICE, epoch
            )
            val_loss, val_auc = validate(model, val_loader, Config.DEVICE)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Val AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())

        # Load best state for this teacher
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Teacher {i+1} Best Val AUC: {best_auc}")

        teachers.append(model)

    return teachers


def generate_sanitized_pseudo_labels(teachers, test_data, load_cached_data=False):
    """
    Stage 2: Generate sanitized pseudo-labels for the test set.

    Args:
        teachers (list): List of trained teacher models.
        test_data (tuple): (images, labels, ids) for testing.
        load_cached_data (bool): Whether to load pseudo-labels from disk.

    Returns:
        np.ndarray: Sanitized pseudo-labels (N_test, Num_Classes).
    """
    cache_path = os.path.join(Config.WORKING_DIR, "sanitized_pseudo_labels.npy")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pseudo-labels from {cache_path}")
        try:
            pseudo_labels = np.load(cache_path)
            check_tensor_sanitation(pseudo_labels, "Cached Pseudo Labels")
            return pseudo_labels
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    test_images, _, test_ids = test_data

    # Create loader (No augmentation for inference)
    test_dataset = BirdDataset(
        test_images,
        np.zeros((len(test_images), Config.NUM_CLASSES)),
        test_ids,
        transform=get_transforms("val"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Starting Stage 2: Generating Pseudo-Labels...")

    all_teacher_preds = []

    for i, model in enumerate(teachers):
        preds, _ = predict(model, test_loader, Config.DEVICE)
        all_teacher_preds.append(preds)

    # Stack and Average
    all_teacher_preds = np.stack(
        all_teacher_preds
    )  # (Num_Teachers, N_samples, Num_Classes)
    avg_preds = np.mean(all_teacher_preds, axis=0)

    # Sanitation Check
    print("Performing Sanitation Check on Pseudo-Labels...")
    check_tensor_sanitation(avg_preds, "Averaged Pseudo Labels")

    # Save to cache
    np.save(cache_path, avg_preds)
    print(f"Sanitized pseudo-labels saved to {cache_path}")

    return avg_preds


def train_student_swa(train_data, test_data, pseudo_labels, epochs=None):
    """
    Stage 3: Train Student Model with SWA on Combined Data.

    Args:
        train_data (tuple): (images, labels, ids)
        test_data (tuple): (images, labels, ids) - labels here are ignored
        pseudo_labels (np.ndarray): Soft labels for test data.
        epochs (int): Total training epochs.

    Returns:
        torch.nn.Module: The final SWA student model.
    """
    if epochs is None:
        epochs = Config.STUDENT_EPOCHS

    train_images, train_labels, train_ids = train_data
    test_images, _, test_ids = test_data

    # Combine Data
    print("Combining Labeled Train Data and Pseudo-Labeled Test Data...")
    combined_images = np.concatenate([train_images, test_images], axis=0)
    combined_labels = np.concatenate([train_labels, pseudo_labels], axis=0)
    combined_ids = np.concatenate([train_ids, test_ids], axis=0)

    # Sanitation check on combined labels
    check_tensor_sanitation(combined_labels, "Combined Labels")

    # Dataset & Loader
    student_dataset = BirdDataset(
        combined_images,
        combined_labels,
        combined_ids,
        transform=get_transforms("train"),
    )
    student_loader = DataLoader(
        student_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Student
    print("Starting Stage 3: Training Student with SWA...")
    set_seed(Config.SEED)  # Reset seed for reproducibility
    student_model = get_model(pretrained=Config.PRETRAINED)

    # Optimizer
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # SWA Handler
    swa_handler = SWAHandler(student_model)
    swa_start = Config.SWA_START_EPOCH

    for epoch in range(epochs):
        # Adjust Learning Rate for SWA
        if epoch >= swa_start:
            for param_group in optimizer.param_groups:
                param_group["lr"] = Config.SWA_LR

        loss = train_one_epoch(
            student_model,
            optimizer,
            student_loader,
            Config.DEVICE,
            epoch,
            swa_handler=swa_handler,
            swa_start_epoch=swa_start,
        )

        status = "SWA Active" if epoch >= swa_start else "Standard Training"
        print(f"Student Epoch {epoch+1}/{epochs} | Loss: {loss:.6f} | {status}")

    # Finalize SWA Model
    print("Finalizing SWA Model (Updating BN Statistics)...")
    final_model = swa_handler.get_averaged_model()
    update_bn_statistics(final_model, student_loader, Config.DEVICE)

    return final_model


def generate_submission(model, test_data):
    """
    Generates the submission CSV.

    Args:
        model (torch.nn.Module): Trained model.
        test_data (tuple): (images, labels, ids).
    """
    test_images, _, test_ids = test_data

    dataset = BirdDataset(
        test_images,
        np.zeros((len(test_images), Config.NUM_CLASSES)),
        test_ids,
        transform=get_transforms("val"),
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print("Generating Final Predictions...")
    preds, ids = predict(model, loader, Config.DEVICE)

    # Format Submission
    # Id = rec_id * 100 + species_id
    submission_rows = []

    for i in range(len(ids)):
        rec_id = int(ids[i])
        probs = preds[i]
        for species_idx in range(Config.NUM_CLASSES):
            row_id = rec_id * 100 + species_idx
            prob = probs[species_idx]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline(debug=False):
    """
    Main orchestration function.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    if debug:
        print("DEBUG MODE: Subsetting data...")
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Load Data
    # Disable caching in debug mode to avoid overwriting full data cache with subset
    use_cache = not debug
    print(f"Loading Data (Cache Enabled: {use_cache})...")

    train_data = get_data(train_df, load_cached_data=use_cache, cache_prefix="train")
    val_data = get_data(val_df, load_cached_data=use_cache, cache_prefix="val")
    test_data = get_data(test_df, load_cached_data=use_cache, cache_prefix="test")

    # Stage 1: Teachers
    teachers = train_teacher_ensemble(train_data, val_data)

    # Stage 2: Pseudo-Labels
    pseudo_labels = generate_sanitized_pseudo_labels(
        teachers, test_data, load_cached_data=use_cache
    )

    # Stage 3: Student
    student_model = train_student_swa(train_data, test_data, pseudo_labels)

    # Submission
    generate_submission(student_model, test_data)
