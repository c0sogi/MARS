import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.cpc_loader import get_cpc_texts
from library.dataset import load_processed_data, PhraseDataset
from library.model import PhraseModel, get_optimizer_grouped_parameters
from library.engine import train_fn, eval_fn, inference_fn

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


def main():
    print("Initializing Phrase Similarity Demo...")

    # 1. Setup & Configuration Overrides for Speed
    # We enable debug mode to use a small subset of data (100 rows)
    Config.debug = True
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.print_freq = 10
    # Enable AWP from epoch 0 to demonstrate it works immediately
    Config.awp_start_epoch = 0.0

    seed_everything(Config.seed)

    print(f"Device: {Config.device}")
    print(f"Debug Mode: {Config.debug}")

    # 2. Data Loading & Processing
    print("\n[1/6] Loading Data and CPC Contexts...")
    # Load CPC descriptions (cached or parsed)
    cpc_texts = get_cpc_texts(load_cached_data=True)

    # Load DataFrames (Train, Val, Test)
    # The library handles caching and mapping context codes to text
    df_train = load_processed_data(Config.train_path, cpc_texts, debug=Config.debug)
    df_val = load_processed_data(Config.val_path, cpc_texts, debug=Config.debug)
    df_test = load_processed_data(Config.test_path, cpc_texts, debug=Config.debug)

    # Validation: Check data loaded correctly
    assert len(df_train) > 0, "Training data is empty"
    assert "context_text" in df_train.columns, "Context text mapping failed"
    print(f"Loaded {len(df_train)} training samples (Debug subset)")

    # 3. Tokenizer & Dataset
    print("\n[2/6] Preparing Datasets and Dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    train_dataset = PhraseDataset(df_train, tokenizer, max_length=Config.max_length)
    val_dataset = PhraseDataset(df_val, tokenizer, max_length=Config.max_length)
    test_dataset = PhraseDataset(df_test, tokenizer, max_length=Config.max_length)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Validation: Check batch structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch
    assert (
        sample_batch["input_ids"].shape[1] == Config.max_length
    ), f"Expected seq length {Config.max_length}, got {sample_batch['input_ids'].shape[1]}"
    print("Dataset and DataLoader verification passed.")

    # 4. Model Initialization
    print("\n[3/6] Initializing Model and Optimizer...")
    model = PhraseModel(model_name=Config.model_name, pretrained=True)
    model.to(Config.device)

    # Setup Optimizer with LLRD (Layer-wise Learning Rate Decay)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        layer_decay=Config.layer_decay,
    )
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.eps
    )

    # Setup Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Setup Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    print("Model initialized successfully.")

    # 5. Training Loop
    print("\n[4/6] Starting Training (1 Epoch with AWP)...")
    # We run the training function provided in engine.py
    # This handles forward/backward passes, AWP, and logging
    avg_loss = train_fn(
        train_loader,
        model,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=Config.device,
        scaler=scaler,
    )

    # Validation: Check loss
    assert isinstance(avg_loss, float), "Training loss should be a float"
    assert avg_loss > 0, "Training loss should be positive"
    print(f"Training completed. Average Loss: {avg_loss:.4f}")

    # 6. Evaluation
    print("\n[5/6] Running Evaluation...")
    val_loss, val_pearson = eval_fn(val_loader, model, Config.device)

    # Validation: Check metrics
    assert (
        -1.0 <= val_pearson <= 1.0
    ), f"Pearson score {val_pearson} out of range [-1, 1]"
    print(f"Evaluation completed. Val Loss: {val_loss:.4f}, Pearson: {val_pearson:.4f}")

    # 7. Inference
    print("\n[6/6] Running Inference on Test Set...")
    predictions = inference_fn(test_loader, model, Config.device)

    # Validation: Check predictions
    assert len(predictions) == len(
        df_test
    ), f"Prediction count {len(predictions)} does not match test set size {len(df_test)}"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions should be sigmoid probabilities in [0, 1]"

    # Create submission dataframe (in memory)
    submission = pd.DataFrame({"id": df_test["id"], "score": predictions})

    print("Inference completed.")
    print("\nSample Predictions:")
    print(submission.head())

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
