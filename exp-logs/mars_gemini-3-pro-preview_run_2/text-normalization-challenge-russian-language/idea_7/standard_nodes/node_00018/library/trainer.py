import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.utils import setup_logger, set_seed
from library.transformer_model import CharToSubwordTransformer
from library.dataset import NormalizationDataset, collate_fn
from library.data_factory import DataFactory


class Trainer:
    """
    Manages the training lifecycle of the Transformer model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LambdaLR,
        criterion: nn.Module,
        device: str,
        grad_clip: float,
    ):
        self.logger = setup_logger("Trainer")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        self.scaler = GradScaler()

    def train_epoch(self, epoch: int) -> float:
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        # Log interval for large datasets
        log_interval = max(1, len(self.train_loader) // 10)

        for i, batch in enumerate(self.train_loader):
            # Unpack batch
            # src: (batch, src_len)
            # tgt: (batch, tgt_len) - includes BOS and EOS
            src, tgt = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Prepare inputs and targets for teacher forcing
            # Input to decoder: BOS ... Token_N
            tgt_input = tgt[:, :-1]
            # Target for loss: Token_1 ... EOS
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast():
                # logits: (batch, tgt_len-1, vocab_size)
                logits = self.model(src, tgt_input)

                # Reshape for loss calculation
                # logits: (batch * seq_len, vocab_size)
                # target: (batch * seq_len)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

            # Backward pass with Scaler
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            total_loss += loss.item()

            if (i + 1) % log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Epoch {epoch} | Step {i+1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.2e}"
                )

        avg_loss = total_loss / len(self.train_loader)
        duration = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch} Train Loss: {avg_loss:.6f} | Time: {duration:.2f}s"
        )
        return avg_loss

    def validate(self, epoch: int) -> float:
        """
        Runs validation loop. Returns average loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                src, tgt = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                with autocast():
                    logits = self.model(src, tgt_input)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                    )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch} Val Loss: {avg_loss:.10f}")
        return avg_loss

    def fit(self, epochs: int, patience: int, save_path: str):
        """
        Main training loop with Early Stopping.
        """
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            _ = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.logger.info(f"New best model found! Saving to {save_path}")
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss:.10f}")


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and pi, after a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_transformer(load_cached_data: bool = True):
    """
    Orchestrates the data loading, model initialization, and training process.
    """
    logger = setup_logger("TrainPipeline")
    set_seed(Config.SEED)

    # 1. Prepare Data
    factory = DataFactory()

    # Load metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)

    # Train BPE Tokenizer (Deterministic)
    factory.train_bpe_tokenizer(df_train_meta)

    # Generate/Load Datasets
    # Train set: Enriched with Residuals + Anchors
    df_train_enriched = factory.generate_curriculum_data(
        df_train_meta, load_cached_data=load_cached_data
    )

    # Val set: Filtered for relevant tokens
    df_val_filtered = factory.prepare_val_data(
        df_train_meta, df_val_meta, load_cached_data=load_cached_data
    )

    # 2. Create Datasets & Loaders
    char_vocab_path = os.path.join(Config.WORKING_DIR, "char_vocab.json")

    # Train Dataset
    train_dataset = NormalizationDataset(
        data=df_train_enriched,
        bpe_model_path=Config.BPE_MODEL_PREFIX,
        context_source_path=None,  # Enriched data is already self-contained rows usually, or we let it generate context
        char_vocab_path=char_vocab_path,
        mode="train",
    )

    # Val Dataset (Must use same char vocab)
    val_dataset = NormalizationDataset(
        data=df_val_filtered,
        bpe_model_path=Config.BPE_MODEL_PREFIX,
        context_source_path=Config.VAL_META,  # Use full val meta to recover context for validation samples
        char_vocab_path=char_vocab_path,
        mode="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    src_vocab_size = train_dataset.char_tokenizer.vocab_size
    tgt_vocab_size = (
        train_dataset.sp.get_piece_size() + 10
    )  # Safety buffer for special tokens

    # Get special token IDs from dataset
    pad_idx_src = train_dataset.pad_idx_src
    pad_idx_tgt = train_dataset.pad_idx_tgt
    bos_idx = train_dataset.bos_idx
    eos_idx = train_dataset.eos_idx

    logger.info(
        f"Initializing Transformer: Src Vocab={src_vocab_size}, Tgt Vocab={tgt_vocab_size}"
    )

    model = CharToSubwordTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx_src=pad_idx_src,
        pad_idx_tgt=pad_idx_tgt,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    ).to(Config.DEVICE)

    # 4. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx_tgt, label_smoothing=Config.LABEL_SMOOTHING
    )

    # Scheduler
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    # 5. Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=Config.DEVICE,
        grad_clip=Config.GRAD_CLIP,
    )

    trainer.fit(
        epochs=Config.EPOCHS, patience=Config.PATIENCE, save_path=Config.BEST_MODEL_PATH
    )


# Import pandas here to avoid top-level dependency if module is just imported
import pandas as pd
