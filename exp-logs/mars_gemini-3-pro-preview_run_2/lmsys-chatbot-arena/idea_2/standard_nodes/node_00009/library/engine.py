import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import get_logger, seed_everything
from library.dataset import get_dataloaders
from library.model import SiameseDualEncoder, train_one_epoch, evaluate

logger = get_logger("engine")


def train(
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    train_batch_size=Config.TRAIN_BATCH_SIZE,
    valid_batch_size=Config.VALID_BATCH_SIZE,
    debug=Config.DEBUG,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Orchestrates the training process with flexibility for hyperparameters.

    Args:
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation.
        debug (bool): Whether to run in debug mode (subset of data).
        save_path (str): Path to save the best model checkpoint.

    Returns:
        model: The trained model instance.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 1. Load Data
    # get_dataloaders handles caching internally
    train_loader, val_loader, _ = get_dataloaders(
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        debug=debug,
        load_cached_data=True,
    )

    # 2. Initialize Model
    model = SiameseDualEncoder(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        dropout_prob=Config.DROPOUT,
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        # Print full precision metrics as requested
        print(
            f"Epoch {epoch+1}: Train Loss {train_loss}, Train Acc {train_acc}, Val Loss {val_loss}, Val Acc {val_acc}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    return model


def predict(
    test_batch_size=Config.VALID_BATCH_SIZE,
    debug=Config.DEBUG,
    model_path=Config.MODEL_SAVE_PATH,
    submission_path=Config.SUBMISSION_PATH,
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        test_batch_size (int): Batch size for inference.
        debug (bool): Whether to run in debug mode.
        model_path (str): Path to the saved model checkpoint.
        submission_path (str): Path to save the submission CSV.
    """
    device = Config.DEVICE

    # 1. Load Data
    _, _, test_loader = get_dataloaders(
        valid_batch_size=test_batch_size, debug=debug, load_cached_data=True
    )

    # 2. Load Model
    model = SiameseDualEncoder(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        dropout_prob=Config.DROPOUT,
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"No checkpoint found at {model_path}. Using random weights.")

    model.eval()

    # 3. Inference
    all_ids = []
    all_probs = []

    logger.info("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)

            # IDs are needed for the submission file
            ids = batch["id"]
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy()

            logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_ids.extend(ids)
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    # 4. Save Submission
    # Columns: id, winner_model_a, winner_model_b, winner_tie
    df_sub = pd.DataFrame(
        {
            "id": all_ids,
            "winner_model_a": all_probs[:, 0],
            "winner_model_b": all_probs[:, 1],
            "winner_tie": all_probs[:, 2],
        }
    )

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_sub.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
