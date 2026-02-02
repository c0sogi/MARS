import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure the library modules can be imported
sys.path.append(".")

# Import provided library components
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.dataset import prepare_data, PhraseDataset
from library.model import PhraseModel
from library.loss import HybridLoss
from library.optimization import get_optimizer_params, AWP, EMA
from library.engine import train_fn, valid_fn, inference_fn


def demo_pipeline():
    print("=== Starting Phrase Matching Demo Pipeline ===")

    # 1. Configuration & Setup
    # Override CFG for speed and demonstration purposes
    print("Configuring environment...")
    CFG.model_name = (
        "microsoft/deberta-v3-xsmall"  # Use a small model for fast execution
    )
    CFG.debug = True
    CFG.epochs = 1
    CFG.train_batch_size = 4
    CFG.valid_batch_size = 4
    CFG.max_len = 64  # Reduce sequence length
    CFG.print_freq = 5
    CFG.awp_start_epoch = 0  # Enable AWP immediately for testing
    CFG.warmup_epochs = 0  # Skip warmup freezing for testing

    # Set device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(CFG.seed)
    transformers.logging.set_verbosity_error()  # Suppress transformer warnings

    print(f"Device: {device}")
    print(f"Model: {CFG.model_name}")

    # 2. Data Preparation
    print("\n=== Data Preparation ===")
    # Create a small subset of data for the demo
    original_train_path = "./metadata/train.csv"
    demo_train_path = "./working/demo_train.csv"

    if not os.path.exists(original_train_path):
        raise FileNotFoundError(f"Metadata not found at {original_train_path}")

    df_full = pd.read_csv(original_train_path)
    df_subset = df_full.head(20).copy()  # Use only 20 samples
    df_subset.to_csv(demo_train_path, index=False)
    print(f"Created demo dataset with {len(df_subset)} samples.")

    # Process data (Load CPC context texts)
    # This utilizes CPCLoader internally
    df_processed = prepare_data(demo_train_path, load_cached_data=False)

    # Validate processing
    assert "context_text" in df_processed.columns, "Context text merging failed"
    assert len(df_processed) == 20, "Data subsetting failed"
    print("Data processed successfully.")

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    train_dataset = PhraseDataset(df_processed, tokenizer, max_len=CFG.max_len)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for demo
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Validate Batch
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "label" in sample_batch
    print("DataLoader operational.")

    # 3. Model Initialization
    print("\n=== Model Initialization ===")
    model = PhraseModel(model_name=CFG.model_name, pretrained=True)
    model.to(device)
    print("PhraseModel instantiated.")

    # 4. Optimizer & Scheduler
    print("\n=== Optimization Setup ===")
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        head_lr=CFG.head_lr,
        weight_decay=CFG.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    num_training_steps = len(train_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    criterion = HybridLoss()

    # Advanced Training Components
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-4)
    ema = EMA(model, decay=0.999)
    ema.register()
    print("Optimizer, Scheduler, AWP, and EMA initialized.")

    # 5. Training Loop
    print("\n=== Starting Training Loop (1 Epoch) ===")
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
        ema=ema,
    )
    print(f"Training Epoch Completed. Average Loss: {avg_loss:.6f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # 6. Validation Loop
    print("\n=== Starting Validation Loop ===")
    # Using train_loader as validation loader for demonstration
    val_loss, val_score, val_preds = valid_fn(
        valid_loader=train_loader,
        model=model,
        criterion=criterion,
        device=device,
        ema=ema,
    )
    print(f"Validation Completed. Loss: {val_loss:.6f}, Pearson Score: {val_score:.6f}")
    assert len(val_preds) == len(df_subset), "Prediction count mismatch"
    # Pearson score can be negative, just checking it's a valid float within reasonable bounds
    assert -1.0 <= val_score <= 1.0, "Pearson score out of valid range"

    # 7. Inference Loop
    print("\n=== Starting Inference Loop ===")
    test_preds = inference_fn(
        test_loader=train_loader, model=model, device=device, ema=ema
    )
    print(f"Inference Completed. Predictions shape: {test_preds.shape}")
    assert test_preds.shape == (len(df_subset),), "Inference output shape mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
