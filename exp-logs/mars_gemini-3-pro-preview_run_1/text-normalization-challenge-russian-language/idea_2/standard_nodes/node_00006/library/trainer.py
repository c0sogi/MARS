import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.tokenizer import CharTokenizer
from library.data_manager import NormalizationDataset
from library.neural_model import TransformerSeq2Seq, Trainer


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(load_cached_data=True):
    """
    Orchestrates the training lifecycle of the neural normalization model.

    Args:
        load_cached_data (bool): If True, attempts to load tokenizer/stats from cache.
                                 If False, rebuilds them from scratch.

    Returns:
        model: The trained PyTorch model.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    Config.print_summary()
    print(f"Initializing training on device: {Config.DEVICE}")

    # 2. Initialize Tokenizer
    # The tokenizer's build_vocab method handles the caching logic internally.
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(Config.TRAIN_FILE, load_cached=load_cached_data)

    # 3. Initialize Datasets
    # The NormalizationDataset filters for digit-containing tokens and handles context.
    print("Initializing Training Dataset...")
    train_dataset = NormalizationDataset(
        data_path=Config.TRAIN_FILE,
        tokenizer=tokenizer,
        max_len=Config.MAX_INPUT_LEN,
        context_window=Config.CONTEXT_WINDOW,
        mode="train",
        load_cached=load_cached_data,
    )

    print("Initializing Validation Dataset...")
    val_dataset = NormalizationDataset(
        data_path=Config.VAL_FILE,
        tokenizer=tokenizer,
        max_len=Config.MAX_INPUT_LEN,
        context_window=Config.CONTEXT_WINDOW,
        mode="val",
        load_cached=load_cached_data,
    )

    # 4. Create DataLoaders
    # Pin memory is beneficial for GPU training
    use_pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    # 5. Initialize Model
    # We use the same number of layers for encoder and decoder as per Config
    model = TransformerSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        d_model=Config.EMBED_DIM,
        nhead=Config.N_HEADS,
        num_encoder_layers=Config.N_LAYERS,
        num_decoder_layers=Config.N_LAYERS,
        dim_feedforward=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        sos_token_id=tokenizer.sos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_len=Config.MAX_INPUT_LEN,
    ).to(Config.DEVICE)

    # 6. Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 7. Initialize Trainer
    # The Trainer class handles the loop, validation, and checkpoint saving
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        save_path=Config.MODEL_CHECKPOINT,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 8. Start Training
    trainer.fit(epochs=Config.NUM_EPOCHS)

    return model
