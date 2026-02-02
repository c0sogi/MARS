import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.transformer_model import get_model, TransformerTrainer


class BatchScheduler:
    """
    Wrapper to adapt a batch-level scheduler (like transformers linear schedule)
    to the interface expected by TransformerTrainer.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step_batch(self):
        """Called after every batch."""
        self.scheduler.step()

    def step(self, val_loss=None):
        """Called after every epoch. No-op for batch schedulers."""
        pass


def train_model(load_cached_data=True):
    """
    Orchestrates the training of the Transformer model.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
                                 If False, triggers re-processing of datasets.

    Returns:
        model: The trained PyTorch model loaded with the best weights.
    """
    # 1. Setup Environment
    set_seed()
    device = get_device()
    print(f"Trainer: Using device {device}")

    # 2. Data Loading
    # get_dataloaders handles tokenizer building/loading and dataset construction/caching
    train_loader, val_loader, char_tokenizer, target_tokenizer = get_dataloaders(
        load_cached_data=load_cached_data,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    print("Trainer: Initializing model...")
    model = get_model(
        src_vocab_size=char_tokenizer.vocab_size,
        tgt_vocab_size=target_tokenizer.vocab_size,
        device=device,
    )

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total training steps for the scheduler
    # len(train_loader) is the number of batches per epoch
    total_steps = len(train_loader) * Config.EPOCHS

    # Use a linear schedule with warmup
    raw_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    # Wrap it for the TransformerTrainer which expects specific method names
    scheduler = BatchScheduler(raw_scheduler)

    # 5. Loss Function
    # CrossEntropyLoss with Label Smoothing
    # We use the pad_id from the target tokenizer (SentencePiece usually uses 0)
    criterion = nn.CrossEntropyLoss(
        ignore_index=target_tokenizer.pad_id, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 6. Trainer Initialization
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config=Config,
    )

    # 7. Execution
    print("Trainer: Starting training loop...")
    trainer.fit()

    # 8. Load Best Model
    # The trainer saves the best model state dict to Config.MODEL_BEST_PATH
    print(f"Trainer: Loading best model from {Config.MODEL_BEST_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_BEST_PATH, map_location=device))

    return model
