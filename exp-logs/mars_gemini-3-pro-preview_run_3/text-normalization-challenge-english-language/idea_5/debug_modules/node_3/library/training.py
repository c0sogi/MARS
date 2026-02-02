import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.neural_data import prepare_neural_data, NormalizationDataset, collate_fn
from library.transformer import Seq2SeqTransformer, create_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Optimization
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.get_id(Config.PAD_TOKEN)
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler (Optional, but good practice)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()

        pad_id = self.tokenizer.get_id(Config.PAD_TOKEN)

        for batch_idx, batch in enumerate(dataloader):
            src = batch["src"].to(self.device)
            tgt_in = batch["tgt_in"].to(self.device)
            tgt_out = batch["tgt_out"].to(self.device)

            # Create masks
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_in, pad_id, self.device
            )

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                src,
                tgt_in,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,  # memory_key_padding_mask same as src_padding_mask
            )

            # Reshape for loss: (batch*seq_len, vocab_size) vs (batch*seq_len)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
            )

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()

            total_loss += loss.item()

            # Calculate Accuracy (Token-level)
            # Exclude padding from accuracy calculation
            preds = torch.argmax(logits, dim=-1)
            mask = tgt_out != pad_id
            correct = (preds == tgt_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch_idx} | Train Loss: {avg_loss:.6f} | Train Acc: {accuracy:.6f} | Time: {elapsed:.2f}s"
        )
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        pad_id = self.tokenizer.get_id(Config.PAD_TOKEN)

        with torch.no_grad():
            for batch in dataloader:
                src = batch["src"].to(self.device)
                tgt_in = batch["tgt_in"].to(self.device)
                tgt_out = batch["tgt_out"].to(self.device)

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt_in, pad_id, self.device
                )

                logits = self.model(
                    src,
                    tgt_in,
                    src_mask,
                    tgt_mask,
                    src_padding_mask,
                    tgt_padding_mask,
                    src_padding_mask,
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                mask = tgt_out != pad_id
                correct = (preds == tgt_out) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, epochs, patience):
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)

            print(
                f"Epoch {epoch} | Val Loss: {val_loss:.10f} | Val Acc: {val_acc:.10f}"
            )

            self.scheduler.step(val_loss)

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(
                    f"Validation loss improved. Saving model to {Config.MODEL_CHECKPOINT}"
                )
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break


def run_training(load_cached_data=True):
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # 2. Data Preparation
    print("Preparing data...")
    tokenizer = prepare_neural_data(load_cached_data=load_cached_data)

    # 3. Datasets & Loaders
    print("Loading datasets...")
    train_dataset = NormalizationDataset(
        Config.PROCESSED_TRAIN, tokenizer, split="train", max_len=Config.MAX_SEQ_LEN
    )
    val_dataset = NormalizationDataset(
        Config.PROCESSED_VAL, tokenizer, split="val", max_len=Config.MAX_SEQ_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 4. Model Initialization
    print("Initializing model...")
    vocab_size = len(tokenizer)
    model = Seq2SeqTransformer(
        num_tokens=vocab_size,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
    ).to(device)

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 5. Training
    trainer = Trainer(model, tokenizer, device)
    trainer.fit(
        train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE
    )

    print("Training complete.")
