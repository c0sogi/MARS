import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import CHECKPOINT_DIR, DEVICE, ModelConfig
from library.transformer_model import CharToBPESeq2Seq

# =============================================================================
# TRAINING HELPER FUNCTIONS
# =============================================================================


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Extract data
        # src: [batch_size, enc_len]
        src = batch["encoder_input"].to(device)
        # tgt: [batch_size, dec_len] (includes BOS and EOS)
        tgt = batch["decoder_target"].to(device)

        # Prepare inputs for Seq2Seq
        # Decoder Input: Remove last token (EOS or PAD)
        tgt_input = tgt[:, :-1]
        # Target Output: Remove first token (BOS)
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        # Forward pass
        # logits: [batch_size, seq_len, vocab_size]
        logits = model(src, tgt_input)

        # Reshape for Loss
        # logits: [batch_size * seq_len, vocab_size]
        # tgt_output: [batch_size * seq_len]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for Transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Runs validation loop.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["encoder_input"].to(device)
            tgt = batch["decoder_target"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# MAIN TRAINING ROUTINE
# =============================================================================


def train_model(
    config: ModelConfig, train_loader, val_loader, char_vocab_size, bpe_vocab_size
):
    """
    Initializes and trains the Transformer model.
    Saves the best checkpoint based on validation loss.
    """
    print(f"Initializing Transformer Model on {DEVICE}...")
    print(f"  Char Vocab: {char_vocab_size}, BPE Vocab: {bpe_vocab_size}")

    # 1. Initialize Model
    # PAD ID is 0 for both CharTokenizer and SentencePiece BPE
    model = CharToBPESeq2Seq(
        config,
        char_vocab_size=char_vocab_size,
        bpe_vocab_size=bpe_vocab_size,
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(DEVICE)

    # 2. Setup Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # CrossEntropyLoss with Label Smoothing
    # ignore_index=0 handles the padding tokens in the target
    criterion = nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=config.label_smoothing
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "transformer_best.pth")

    print(f"Starting training for {config.num_epochs} epochs...")

    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE)

        epoch_duration = time.time() - epoch_start

        # Logging
        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{config.patience}"
            )

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation Loss: {best_val_loss}")

    return model
