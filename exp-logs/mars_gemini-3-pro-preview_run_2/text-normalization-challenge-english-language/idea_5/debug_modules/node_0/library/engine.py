import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import get_logger, AverageMeter
from library.model import TransformerCRF
from library.normalization_rules import normalize_token

logger = get_logger()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    steps = len(dataloader)
    log_interval = max(1, steps // 10)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass with labels returns NLL loss
        loss = model(input_ids, attention_mask, labels=labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), input_ids.size(0))

        if (step + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch+1} | Step {step+1}/{steps} | Loss: {loss_meter.avg:.6f}"
            )

    return loss_meter.avg


def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """
    Evaluates the model on the validation set.
    Returns average loss.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels=labels)
            loss_meter.update(loss.item(), input_ids.size(0))

    return loss_meter.avg


def fit(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.TRAIN_BATCH_SIZE,
    device: torch.device = Config.DEVICE,
):
    """
    Main training loop with Early Stopping.
    """
    logger.info(f"Starting training for {epochs} epochs on {device}...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Optimizer & Scheduler
    # Group parameters to handle weight decay correctly
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    patience = 0
    patience_limit = 1  # Strict patience due to time constraints

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch+1} Summary: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(
                f"Validation loss improved. Saving model to {Config.MODEL_CHECKPOINT_PATH}"
            )
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            patience = 0
        else:
            patience += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience}/{patience_limit}"
            )
            if patience >= patience_limit:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")


def predict(model: torch.nn.Module, test_dataset, device: torch.device = Config.DEVICE):
    """
    Runs inference on the test dataset and generates the submission file.
    """
    logger.info("Starting inference on test set...")

    # Load best model
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        logger.info(f"Loading weights from {Config.MODEL_CHECKPOINT_PATH}")
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )
    else:
        logger.warning(
            "No checkpoint found! Using current model weights (likely untrained or random)."
        )

    model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    # We need the tokenizer to re-align subwords to words
    tokenizer = test_dataset.tokenizer

    with torch.no_grad():
        # Iterate over batches
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get tag sequences (List[List[int]])
            # These correspond to the sub-word sequence
            batch_tag_seqs = model(input_ids, attention_mask)

            # Process each sentence in the batch
            for i, tag_seq in enumerate(batch_tag_seqs):
                # Calculate global index to retrieve raw tokens
                global_idx = batch_idx * test_loader.batch_size + i
                if global_idx >= len(test_dataset):
                    break

                # Retrieve raw data
                raw_tokens = test_dataset.tokens_list[global_idx]
                row_ids = test_dataset.ids_list[global_idx]

                # Re-tokenize to get word_ids mapping
                # We use the same parameters as in Dataset
                encoding = tokenizer(
                    raw_tokens,
                    is_split_into_words=True,
                    max_length=Config.MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                word_ids = encoding.word_ids()

                # Map subword tags to word-level labels
                # Strategy: Take the label of the first subword of each word

                pred_labels = {}  # word_index -> label_str
                current_word_idx = -1

                # tag_seq length is the valid length of the sequence (sum of mask)
                # word_ids length is MAX_LEN
                # We iterate until we run out of tags or word_ids

                limit = min(len(tag_seq), len(word_ids))

                for t in range(limit):
                    w_id = word_ids[t]
                    tag_id = tag_seq[t]

                    # Skip special tokens
                    if w_id is None:
                        continue

                    # If this is a new word we haven't labeled yet
                    if w_id != current_word_idx:
                        label_str = Config.ID2LABEL.get(tag_id, "PLAIN")
                        pred_labels[w_id] = label_str
                        current_word_idx = w_id

                # Generate normalized text
                for t_idx, token_text in enumerate(raw_tokens):
                    # Default to PLAIN if something went wrong in alignment
                    lbl = pred_labels.get(t_idx, "PLAIN")

                    # Apply normalization rule
                    norm_text = normalize_token(token_text, lbl)

                    results.append({"id": row_ids[t_idx], "after": norm_text})

            if (batch_idx + 1) % 200 == 0:
                logger.info(f"Processed {batch_idx + 1} batches...")

    # Create DataFrame
    logger.info("Constructing submission DataFrame...")
    df_sub = pd.DataFrame(results)

    # Save
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info("Submission saved successfully.")
