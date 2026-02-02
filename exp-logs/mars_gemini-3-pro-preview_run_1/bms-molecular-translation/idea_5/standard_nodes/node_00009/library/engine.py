import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import Tokenizer, compute_levenshtein
from library.dataset import prepare_datasets, collate_fn
from library.model import AttributeAugmentedAttnNet


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    import random

    random.seed(seed)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(dataloader, model, optimizer, device, config):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()
    seq_losses = AverageMeter()
    attr_losses = AverageMeter()

    # Define Loss Functions
    # Ignore padding index (0) for sequence loss
    criterion_seq = nn.CrossEntropyLoss(ignore_index=0)
    criterion_attr = nn.MSELoss()

    for i, data in enumerate(dataloader):
        images = data["image"].to(device)
        token_ids = data["token_ids"].to(device)
        attributes = data["attributes"].to(device)

        optimizer.zero_grad()

        # Forward Pass
        # seq_logits: (B, max_len, vocab_size)
        # attr_pred: (B, num_attributes)
        seq_logits, attr_pred = model(
            images,
            targets=token_ids,
            teacher_forcing_ratio=config.TEACHER_FORCING_RATIO,
        )

        # Calculate Sequence Loss
        # Targets for loss are token_ids[:, 1:] (exclude SOS)
        # Logits correspond to predictions for these steps (up to max_len-1)
        # We align dimensions:
        # seq_logits[:, :-1] predicts token_ids[:, 1:]

        # Determine valid length for loss calculation based on batch max length
        # token_ids shape: [B, L]
        # seq_logits shape: [B, L, V]

        # We flatten batch and sequence dims
        output_dim = seq_logits.shape[-1]

        # Slice to align: Output at t predicts Target at t+1
        # output[:, t] -> target[:, t+1]
        # We ignore the very last output of the decoder (prediction after EOS)
        # and the very first token of target (SOS)

        # Ensure shapes match
        seq_len = token_ids.size(1)

        # Flatten for CrossEntropy
        loss_seq = criterion_seq(
            seq_logits[:, :-1, :].reshape(-1, output_dim), token_ids[:, 1:].reshape(-1)
        )

        # Calculate Attribute Loss
        loss_attr = criterion_attr(attr_pred, attributes)

        # Combined Loss
        loss = loss_seq + (config.LOSS_LAMBDA * loss_attr)

        # Backward Pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)

        optimizer.step()

        # Update Metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        seq_losses.update(loss_seq.item(), batch_size)
        attr_losses.update(loss_attr.item(), batch_size)

    return losses.avg, seq_losses.avg, attr_losses.avg


def eval_fn(dataloader, model, device, tokenizer, config):
    """
    Evaluates the model on the validation set.
    Computes Levenshtein distance and Attribute MSE.
    """
    model.eval()
    attr_losses = AverageMeter()
    levenshtein_scores = AverageMeter()

    criterion_attr = nn.MSELoss()

    with torch.no_grad():
        for data in dataloader:
            images = data["image"].to(device)
            attributes = data["attributes"].to(device)
            original_texts = data["original_text"]

            # Forward Pass (Inference Mode)
            # targets=None triggers greedy decoding up to MAX_LEN
            seq_logits, attr_pred = model(
                images, targets=None, teacher_forcing_ratio=0.0
            )

            # Attribute Loss
            loss_attr = criterion_attr(attr_pred, attributes)
            attr_losses.update(loss_attr.item(), images.size(0))

            # Decode Sequences
            # seq_logits: (B, max_len, V) -> Indices: (B, max_len)
            pred_indices = torch.argmax(seq_logits, dim=2)

            predictions = []
            for i in range(pred_indices.size(0)):
                pred_text = tokenizer.sequence_to_text(pred_indices[i])
                predictions.append(pred_text)

            # Compute Levenshtein Distance
            batch_score = compute_levenshtein(predictions, original_texts)
            levenshtein_scores.update(batch_score, images.size(0))

    return levenshtein_scores.avg, attr_losses.avg


def run_training(config: Config):
    """
    Main training loop with Early Stopping.
    """
    set_seed(config.SEED)

    print(f"Initializing training on device: {config.DEVICE}")

    # 1. Prepare Data
    train_dataset, val_dataset, _, tokenizer = prepare_datasets(
        config, load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Model Setup
    model = AttributeAugmentedAttnNet(config, tokenizer.vocab_size)
    model.to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation metric stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 3. Training Loop
    best_levenshtein = float("inf")
    patience_counter = 0
    early_stopping_patience = 5

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_seq_loss, train_attr_loss = train_fn(
            train_loader, model, optimizer, config.DEVICE, config
        )

        # Validate
        val_levenshtein, val_attr_loss = eval_fn(
            val_loader, model, config.DEVICE, tokenizer, config
        )

        elapsed = time.time() - start_time

        # Logging
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Time: {elapsed:.0f}s")
        print(
            f"  Train Loss: {train_loss:.6f} (Seq: {train_seq_loss:.6f}, Attr: {train_attr_loss:.6f})"
        )
        print(f"  Val Attr Loss: {val_attr_loss:.6f}")
        print(f"  Val Levenshtein: {val_levenshtein}")  # Full precision as requested

        # Scheduler Step
        scheduler.step(val_levenshtein)

        # Checkpointing & Early Stopping
        if val_levenshtein < best_levenshtein:
            best_levenshtein = val_levenshtein
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  -> New best model saved! (Levenshtein: {best_levenshtein})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")


def generate_predictions(config: Config):
    """
    Generates submission file using the trained model.
    """
    set_seed(config.SEED)
    print("Generating predictions for test set...")

    # 1. Load Data
    # We need the tokenizer from training to decode correctly
    # We pass load_cached_data=True to ensure we get the same tokenizer
    _, _, test_dataset, tokenizer = prepare_datasets(config, load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Load Model
    model = AttributeAugmentedAttnNet(config, tokenizer.vocab_size)
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Run training first."
        )

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # 3. Inference Loop
    results = []

    with torch.no_grad():
        for data in test_loader:
            images = data["image"].to(config.DEVICE)
            image_ids = data["image_id"]

            # Forward Pass (Greedy Decoding)
            seq_logits, _ = model(images, targets=None, teacher_forcing_ratio=0.0)

            # Decode
            pred_indices = torch.argmax(seq_logits, dim=2)

            for i, img_id in enumerate(image_ids):
                pred_text = tokenizer.sequence_to_text(pred_indices[i])
                results.append({"image_id": img_id, "InChI": pred_text})

    # 4. Save Submission
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(df_sub)}")
