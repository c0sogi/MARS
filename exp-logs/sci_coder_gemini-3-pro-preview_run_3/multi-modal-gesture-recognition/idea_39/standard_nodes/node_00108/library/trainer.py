import os
import torch
import torch.optim as optim
import numpy as np
from library.config import (
    WORKING_DIR,
    VAL_METADATA_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    SEED,
    EPOCHS,
    BATCH_SIZE,
    BACKGROUND_CLASS_ID,
    MIN_DURATION,
)
from library.model import (
    ASH_KN,
    train_one_epoch,
    validate as validate_loss,
    predict_sequence,
)
from library.data_loader import get_dataloaders, load_and_process_data
from library.utils import compute_levenshtein_score, run_length_encoding


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_levenshtein(model, val_data, device):
    """
    Evaluates the model on the validation set using the Levenshtein distance metric.

    Args:
        model: The trained model.
        val_data: Dictionary containing raw validation samples.
        device: Torch device.

    Returns:
        float: The average Levenshtein error rate.
    """
    model.eval()
    hypotheses = []
    references = []

    # Iterate over all validation samples
    # We sort keys to ensure deterministic order, though not strictly necessary for score
    for sid in sorted(val_data.keys()):
        sample = val_data[sid]
        skeleton = sample["skeleton"]
        audio = sample["audio"]
        labels_frame_wise = sample["labels"]

        # 1. Get Prediction Sequence
        # predict_sequence handles sliding window inference and aggregation
        frame_preds = predict_sequence(model, skeleton, audio, device)

        # Decode using RLE and Min Duration filter
        hyp_seq = run_length_encoding(frame_preds, min_duration=MIN_DURATION)
        hypotheses.append(hyp_seq)

        # 2. Get Reference Sequence
        # Decode ground truth frame-wise labels
        # We use a small min_duration for GT to collapse segments,
        # assuming GT is clean.
        ref_seq = run_length_encoding(labels_frame_wise, min_duration=1)
        references.append(ref_seq)

    # Compute global score
    score = compute_levenshtein_score(hypotheses, references)
    return score


def run_training_session(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Main execution function for the training pipeline.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate.
        weight_decay (float): L2 regularization factor.
        patience (int): Early stopping patience epochs.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        str: Path to the saved best model.
    """
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training session on device: {device}")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    # 2. Data Loading
    # Get windowed loaders for training/validation loss calculation
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, load_cached=load_cached_data
    )

    # Get raw validation data dictionary for Levenshtein metric calculation
    # We explicitly load the validation set structure to perform sequence-level inference
    val_data_raw = load_and_process_data(
        VAL_METADATA_PATH, "dataset_val", load_cached_data
    )

    # 3. Model Initialization
    model = ASH_KN().to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler monitors validation loss for LR reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_metric = float("inf")  # Lower Levenshtein score is better
    patience_counter = 0

    for epoch in range(epochs):
        # A. Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # B. Validation Step (Loss)
        val_loss = validate_loss(model, val_loader, device)

        # C. Validation Step (Metric - Levenshtein)
        # This is the primary metric for model selection as per task description
        lev_score = evaluate_levenshtein(model, val_data_raw, device)

        # Update Scheduler based on Loss (standard practice for stability)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Levenshtein: {lev_score:.6f}"
        )

        # D. Checkpointing & Early Stopping
        # We save based on the Levenshtein score since that is the competition metric
        if lev_score < best_metric:
            best_metric = lev_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! Score: {best_metric:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Levenshtein Score: {best_metric}")
    return best_model_path
