import os
import shutil
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    Logger,
    write_submission_csv,
    calculate_roc_auc,
)
from library.data import get_dataloaders, Mixup
from library.model import create_model
from library.engine import train_one_epoch, evaluate, SWAEngine


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path):
        score = epoch_score

        if self.mode == "min":
            improved = (
                score < (self.best_score - self.delta)
                if self.best_score is not None
                else True
            )
        else:
            improved = (
                score > (self.best_score + self.delta)
                if self.best_score is not None
                else True
            )

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
        elif improved:
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model, model_path):
        torch.save(model.state_dict(), model_path)


def sanitize_state_dict(state_dict):
    """
    Helper to sanitize state_dict keys by removing 'module.' prefix
    and ignoring 'n_averaged' buffer from SWA models.
    Cite debug_lesson_9.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def train_single_model(config, model_name, pseudo_labels=None, swa_start_epoch=None):
    """
    Trains a single model (Teacher or Student) with SWA and Mixup.
    """
    # Initialize environment
    set_seed(config.SEED)
    device = config.DEVICE

    # Get DataLoaders
    dataloaders = get_dataloaders(config, pseudo_labels=pseudo_labels)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Create Model
    model = create_model(config)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    # Determine SWA Start Epoch if not provided
    if swa_start_epoch is None:
        if pseudo_labels is None:
            swa_start_epoch = config.SWA_START_EPOCH_TEACHER
        else:
            swa_start_epoch = config.SWA_START_EPOCH_STUDENT

    # Initialize SWA Engine
    swa_engine = SWAEngine(model, optimizer, config, swa_start_epoch)

    # Initialize Mixup
    mixup_fn = Mixup(alpha=config.MIXUP_ALPHA) if config.MIXUP_PROB > 0 else None

    # Initialize Early Stopping
    best_model_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_base_best.pth")
    # We use a patience of 10 epochs
    early_stopping = EarlyStopping(patience=10, mode="max")

    print(f"[{model_name}] Starting training... SWA starts at epoch {swa_start_epoch}")

    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mixup_fn
        )

        # SWA Step
        is_swa = swa_engine.step(epoch)

        # Validation
        val_loss, val_auc = evaluate(model, val_loader, device)
        print(
            f"[{model_name}] Epoch {epoch}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, Val AUC {val_auc:.6f}"
        )

        # Scheduler Step (Only if SWA is not active, as SWA has its own scheduler)
        if not is_swa:
            scheduler.step()

        # Early Stopping Check
        # Explicitly disable early stopping during SWA phase to ensure SWA convergence
        if not is_swa:
            early_stopping(val_auc, model, best_model_path)
            if early_stopping.early_stop:
                print(f"[{model_name}] Early stopping triggered at epoch {epoch}.")
                # Fallback: If we stop before SWA, use the best base model as the result.
                # We copy the best base model to the SWA path to maintain pipeline consistency.
                swa_model_path = os.path.join(
                    config.OUTPUT_DIR, f"{model_name}_swa.pth"
                )
                shutil.copy(best_model_path, swa_model_path)
                print(f"[{model_name}] Saved best base model to {swa_model_path}")
                return swa_model_path

    # End of Training: Finalize SWA
    print(f"[{model_name}] Updating SWA BatchNorm statistics...")
    swa_engine.update_bn(train_loader)

    swa_model_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_swa.pth")
    torch.save(swa_engine.get_averaged_model().state_dict(), swa_model_path)
    print(f"[{model_name}] Saved SWA model to {swa_model_path}")

    return swa_model_path


def train_teachers(config):
    """
    Stage 1: Train multiple teacher models with different seeds.
    """
    teacher_paths = []
    original_seed = config.SEED

    for i in range(config.NUM_TEACHERS):
        name = f"teacher_{i}"
        path = os.path.join(config.OUTPUT_DIR, f"{name}_swa.pth")

        if os.path.exists(path):
            print(f"Teacher {name} already exists. Skipping.")
            teacher_paths.append(path)
            continue

        # Vary seed for ensemble diversity
        config.SEED = original_seed + i
        print(f"Training {name} with seed {config.SEED}")

        trained_path = train_single_model(config, name, pseudo_labels=None)
        teacher_paths.append(trained_path)

    # Restore original seed
    config.SEED = original_seed
    return teacher_paths


def generate_pseudo_labels(teacher_paths, config, load_cached_data=True):
    """
    Stage 2: Generate pseudo-labels for the test set using the teacher ensemble.
    Implements caching and Test-Time Augmentation (TTA).
    """
    cache_path = os.path.join(config.OUTPUT_DIR, "pseudo_labels.parquet")

    # Caching Mechanism
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pseudo-labels from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating pseudo-labels with Teacher Ensemble...")
    device = config.DEVICE

    # Get Test Loader
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]

    ensemble_probs = None

    for path in teacher_paths:
        print(f"Inference with model: {path}")
        model = create_model(config)
        state_dict = torch.load(path, map_location=device)
        state_dict = sanitize_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        teacher_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # TTA: Original
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # TTA: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])  # Flip width dimension
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)

                # Average TTA
                probs = (probs_orig + probs_flip) / 2.0
                teacher_preds.append(probs.cpu().numpy())

        teacher_preds = np.concatenate(teacher_preds, axis=0)

        if ensemble_probs is None:
            ensemble_probs = teacher_preds
        else:
            ensemble_probs += teacher_preds

    # Average across all teachers
    ensemble_probs /= len(teacher_paths)

    # Sanitize (NaN check)
    if np.isnan(ensemble_probs).any():
        print("Warning: NaNs found in pseudo-labels. Replacing with 0.")
        ensemble_probs = np.nan_to_num(ensemble_probs, nan=0.0)

    # Create DataFrame
    test_df = test_loader.dataset.df
    rec_ids = test_df["rec_id"].values

    data = {"rec_id": rec_ids}
    for i in range(19):
        data[f"species_{i}"] = ensemble_probs[:, i]

    df_pseudo = pd.DataFrame(data)

    # Save to cache
    df_pseudo.to_parquet(cache_path)
    print(f"Pseudo-labels saved to {cache_path}")

    return df_pseudo


def train_student(pseudo_labels_df, config):
    """
    Stage 3: Train the student model on Combined (Train + Pseudo-Test) dataset.
    """
    name = "student"
    path = os.path.join(config.OUTPUT_DIR, f"{name}_swa.pth")

    if os.path.exists(path):
        print(f"Student model already exists at {path}. Skipping.")
        return path

    # Ensure seed is reset
    set_seed(config.SEED)

    return train_single_model(config, name, pseudo_labels=pseudo_labels_df)


def run_pipeline(config):
    """
    Orchestrates the full distillation pipeline and generates submission.
    """
    # 1. Train Teachers
    print("--- Stage 1: Train Teachers ---")
    teacher_paths = train_teachers(config)

    # 2. Generate Pseudo Labels
    print("--- Stage 2: Generate Pseudo Labels ---")
    pseudo_df = generate_pseudo_labels(teacher_paths, config, load_cached_data=True)

    # 3. Train Student
    print("--- Stage 3: Train Student ---")
    student_path = train_student(pseudo_df, config)

    # 4. Final Inference & Submission
    print("--- Final Inference ---")
    device = config.DEVICE
    model = create_model(config)
    state_dict = torch.load(student_path, map_location=device)
    state_dict = sanitize_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]

    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            # Single forward pass for student (no TTA)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Get IDs
    test_df = test_loader.dataset.df
    rec_ids = test_df["rec_id"].values

    write_submission_csv(rec_ids, all_probs, config.SUBMISSION_PATH)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
