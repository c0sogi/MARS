import os
import sys
import torch
import pandas as pd
import numpy as np
import transformers
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library
from library.configuration import Config
from library.utilities import set_seed, decode_text
from library.dataset import InsultDataset, load_data, get_dataloader
from library.architecture import TransformerModel
from library.engine import train_fn, eval_fn, predict


def demo_pipeline():
    # 1. Setup
    # Suppress verbose transformer logs
    transformers.logging.set_verbosity_error()

    # Set seed for reproducibility
    set_seed(42)
    device = Config.DEVICE
    print(f"Running demo on device: {device}")

    # Define a lightweight config for demonstration purposes
    # We use a tiny BERT model to ensure the code runs quickly (seconds)
    # instead of downloading and running the large models in Config.
    DEMO_MODEL_NAME = "prajjwal1/bert-tiny"
    BATCH_SIZE = 4
    MAX_LEN = 32  # Short sequence length for speed

    print(f"\n--- 1. Data Loading & Dataset Verification ---")
    # Load a small subset of data directly to avoid processing the whole file
    # We use the metadata paths defined in Config
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found at {Config.TRAIN_PATH}")

    df_full = pd.read_csv(Config.TRAIN_PATH)

    # Take a tiny subset (e.g., 20 samples)
    df_train_subset = df_full.head(20).copy()
    df_val_subset = df_full.iloc[20:30].copy()

    # Ensure text decoding is applied (demonstrating utility usage)
    df_train_subset["Comment"] = df_train_subset["Comment"].apply(decode_text)
    df_val_subset["Comment"] = df_val_subset["Comment"].apply(decode_text)

    print(f"Train subset shape: {df_train_subset.shape}")
    print(f"Val subset shape: {df_val_subset.shape}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEMO_MODEL_NAME)

    # Instantiate Dataset
    train_ds = InsultDataset(df_train_subset, tokenizer, max_len=MAX_LEN, is_test=False)

    # Verify Dataset Item
    sample_item = train_ds[0]
    print("Sample dataset item keys:", sample_item.keys())

    # Assertions for Dataset
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "target" in sample_item
    assert sample_item["input_ids"].dim() == 1
    assert len(sample_item["input_ids"]) == MAX_LEN
    assert isinstance(sample_item["target"], torch.Tensor)
    print("Dataset verification passed.")

    print(f"\n--- 2. Model Architecture Verification ---")
    # Instantiate Model
    model = TransformerModel(model_name=DEMO_MODEL_NAME, dropout=0.1, freeze_layers=0)
    model.to(device)

    # Create a dummy batch
    dummy_input_ids = sample_item["input_ids"].unsqueeze(0).to(device)  # (1, SeqLen)
    dummy_mask = sample_item["attention_mask"].unsqueeze(0).to(device)  # (1, SeqLen)

    # Forward pass check
    model.eval()
    with torch.no_grad():
        output = model(dummy_input_ids, dummy_mask)

    print(f"Model output shape: {output.shape}")

    # Assertions for Model
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    print("Model architecture verification passed.")

    print(f"\n--- 3. Training Loop (Engine) Verification ---")
    # Create DataLoaders
    train_loader = get_dataloader(
        df_train_subset, tokenizer, batch_size=BATCH_SIZE, is_test=False, shuffle=True
    )
    val_loader = get_dataloader(
        df_val_subset, tokenizer, batch_size=BATCH_SIZE, is_test=False, shuffle=False
    )

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
    )

    # Run one epoch of training
    print("Running train_fn...")
    train_loss = train_fn(
        train_loader, model, optimizer, scheduler, device, grad_acc_steps=1
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Assertions for Training
    assert isinstance(train_loss, float)
    assert train_loss >= 0
    print("Training function verification passed.")

    # Run evaluation
    print("Running eval_fn...")
    val_loss, val_auc = eval_fn(val_loader, model, device)
    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Assertions for Evaluation
    assert isinstance(val_loss, float)
    assert isinstance(val_auc, float)
    assert 0.0 <= val_auc <= 1.0
    print("Evaluation function verification passed.")

    print(f"\n--- 4. Inference (Predict) Verification ---")
    # Simulate Test Data (Reuse val subset but treat as test)
    test_loader = get_dataloader(
        df_val_subset, tokenizer, batch_size=BATCH_SIZE, is_test=True, shuffle=False
    )

    print("Running predict...")
    predictions = predict(test_loader, model, device)

    print(f"Number of predictions: {len(predictions)}")
    print(f"First 5 predictions: {predictions[:5]}")

    # Assertions for Prediction
    assert len(predictions) == len(df_val_subset)
    assert all(
        0.0 <= p <= 1.0 for p in predictions
    ), "Predictions must be probabilities in [0, 1]"
    print("Prediction function verification passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    demo_pipeline()
