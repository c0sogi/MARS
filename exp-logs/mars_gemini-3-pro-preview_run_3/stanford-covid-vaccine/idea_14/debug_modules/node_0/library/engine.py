import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import RNAModel, train_model as lib_train_model
from library.utils import set_seed


def train_engine(train_loader, val_loader, epochs=None):
    """
    Executes the training pipeline.
    Wraps the library's train_model function to allow dynamic configuration of epochs.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs (int, optional): Number of epochs to train. Overrides Config.EPOCHS.

    Returns:
        float: The best validation MCRMSE score.
    """
    # Allow overriding epochs for debugging or tuning
    if epochs is not None:
        Config.EPOCHS = epochs
        # Note: The scheduler in lib_train_model uses Config.EPOCHS as T_max

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Starting training for {Config.EPOCHS} epochs...")

    # Execute training using the provided library function
    # This handles loss, optimizer, scheduler, clipping, and early stopping
    best_score = lib_train_model(train_loader, val_loader)

    return best_score


def predict_submission(test_loader, output_file=None):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Unlike the library's predict function, this generates predictions for all
    107 sequence positions as required by the submission format.

    Args:
        test_loader: DataLoader for test data.
        output_file (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_FILE.
    """
    device = torch.device(Config.DEVICE)

    # Initialize model structure
    model = RNAModel().to(device)

    # Load the best trained model weights
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model weights from {Config.MODEL_PATH}")
    else:
        print(
            f"Warning: Model file {Config.MODEL_PATH} not found. Using random initialization."
        )

    model.eval()

    ids_seqpos = []
    preds_flat = []

    print("Generating predictions for submission...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            inputs = batch["sequence"].to(device)
            pair_indices = batch["pair_index"].to(device)
            batch_ids = batch["id"]  # List of sample IDs

            # Forward pass
            # Shape: (Batch, Seq_Len=107, Targets=5)
            logits = model(inputs, pair_indices)

            # Move to CPU for processing
            logits = logits.cpu().numpy()

            # Iterate through the batch to format output
            for i, sample_id in enumerate(batch_ids):
                sample_logits = logits[i]  # Shape (107, 5)

                # We must predict for all 107 positions
                for seq_pos in range(Config.SEQ_LENGTH):
                    # Construct ID: id_seqpos
                    ids_seqpos.append(f"{sample_id}_{seq_pos}")

                    # Append prediction vector
                    preds_flat.append(sample_logits[seq_pos])

    # Create DataFrame
    # Columns must match the target columns order
    df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)

    # Insert the identifier column at the beginning
    df.insert(0, "id_seqpos", ids_seqpos)

    # Determine save path
    save_path = output_file if output_file else Config.SUBMISSION_FILE

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {df.shape}")
