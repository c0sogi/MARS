import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.utils import set_seed, load_checkpoint
from library.model import get_seresnet_model
from library.dataset import create_dataloaders
from library.engine import run_training_session, predict_with_tta

# Configuration Constants
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"
NUM_CLASSES = 19
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_teachers(num_teachers=3, epochs=50, batch_size=32, lr=1e-3, seed=42):
    """
    Trains an ensemble of Teacher models (SE-ResNet-34).

    Args:
        num_teachers (int): Number of independent teachers to train.
        epochs (int): Number of training epochs per teacher.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        seed (int): Base random seed.

    Returns:
        list: Paths to the trained teacher model checkpoints.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    teacher_paths = []

    # Get DataLoaders (only need train/val for teachers)
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=batch_size, pseudo_labels_df=None, seed=seed
    )

    for i in range(num_teachers):
        current_seed = seed + i
        set_seed(current_seed)

        print(f"\n--- Training Teacher {i+1}/{num_teachers} ---")

        # Initialize Model
        model = get_seresnet_model(
            num_classes=NUM_CLASSES, pretrained=True, device=DEVICE
        )

        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Define save directory for this teacher
        save_dir = os.path.join(WORKING_DIR, f"teacher_{i}")

        # Run Training
        # run_training_session handles SWA and Early Stopping logic internally
        best_model_path = run_training_session(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=epochs,
            save_dir=save_dir,
            patience=15,  # Slightly higher patience for stability
        )

        teacher_paths.append(best_model_path)

    return teacher_paths


def generate_pseudo_labels(
    model_paths,
    output_filename="pseudo_labels.parquet",
    batch_size=32,
    seed=42,
    load_cached_data=True,
):
    """
    Generates pseudo-labels for the test set using an ensemble of models.
    Averages predictions from all provided models.

    Args:
        model_paths (list): List of paths to model checkpoints.
        output_filename (str): Filename for the cached pseudo-labels.
        batch_size (int): Batch size for inference.
        seed (int): Random seed.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing pseudo-labels.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, output_filename)

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pseudo-labels from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print(f"Generating pseudo-labels using {len(model_paths)} models...")
    set_seed(seed)

    # 2. Load Test Data
    # We only need the test loader here
    _, _, test_loader = create_dataloaders(batch_size=batch_size, seed=seed)

    ensemble_preds = []
    rec_ids = None

    # 3. Inference Loop
    for path in model_paths:
        print(f"Inference with model: {path}")
        model = get_seresnet_model(
            num_classes=NUM_CLASSES, pretrained=False, device=DEVICE
        )
        load_checkpoint(model, path, device=DEVICE)

        # Predict with TTA
        preds, ids = predict_with_tta(model, test_loader, device=DEVICE)

        ensemble_preds.append(preds)
        if rec_ids is None:
            rec_ids = ids
        else:
            # Verify ID alignment
            if not np.array_equal(rec_ids, ids):
                raise ValueError(
                    "Mismatch in recording IDs between models during ensemble."
                )

    # 4. Average Predictions
    avg_preds = np.mean(ensemble_preds, axis=0)

    # 5. Create DataFrame
    # Columns: rec_id, species_0, species_1, ...
    data = {"rec_id": rec_ids}
    for i in range(NUM_CLASSES):
        data[f"species_{i}"] = avg_preds[:, i]

    df_pseudo = pd.DataFrame(data)

    # 6. Save to Cache
    df_pseudo.to_parquet(cache_path, index=False)
    print(f"Pseudo-labels saved to {cache_path}")

    return df_pseudo


def train_student(
    pseudo_labels_df,
    student_name="student_1",
    epochs=50,
    batch_size=32,
    lr=1e-3,
    seed=42,
):
    """
    Trains a Student model on Combined (Labeled + Pseudo-labeled) data.

    Args:
        pseudo_labels_df (pd.DataFrame): DataFrame with soft targets for test set.
        student_name (str): Name identifier for saving checkpoints.
        epochs (int): Training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        seed (int): Random seed.

    Returns:
        str: Path to the trained student model.
    """
    set_seed(seed)
    print(f"\n--- Training Student: {student_name} ---")

    # 1. Create DataLoaders with Pseudo-labels
    # This triggers the merging logic in dataset.py
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=batch_size, pseudo_labels_df=pseudo_labels_df, seed=seed
    )

    # 2. Initialize Student Model
    model = get_seresnet_model(num_classes=NUM_CLASSES, pretrained=True, device=DEVICE)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    save_dir = os.path.join(WORKING_DIR, student_name)

    # 4. Train
    best_model_path = run_training_session(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        epochs=epochs,
        save_dir=save_dir,
        patience=15,
    )

    return best_model_path


def generate_submission(
    model_path, output_path="./submission/submission.csv", batch_size=32, seed=42
):
    """
    Generates the final submission CSV using the specified model.

    Args:
        model_path (str): Path to the model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size.
        seed (int): Random seed.
    """
    print(f"\n--- Generating Submission with {model_path} ---")
    set_seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load Data
    _, _, test_loader = create_dataloaders(batch_size=batch_size, seed=seed)

    # 2. Load Model
    model = get_seresnet_model(num_classes=NUM_CLASSES, pretrained=False, device=DEVICE)
    load_checkpoint(model, model_path, device=DEVICE)

    # 3. Predict
    preds, rec_ids = predict_with_tta(model, test_loader, device=DEVICE)

    # 4. Format Submission
    # Id = rec_id * 100 + species_number
    # Flatten the predictions
    submission_rows = []

    for i, rid in enumerate(rec_ids):
        probs = preds[i]
        for species_idx, prob in enumerate(probs):
            row_id = int(rid * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # 5. Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
