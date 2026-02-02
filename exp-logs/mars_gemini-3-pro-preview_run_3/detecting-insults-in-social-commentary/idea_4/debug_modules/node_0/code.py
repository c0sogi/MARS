import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import provided library modules
from library.config import ModelConfig
from library.dataset import load_dataset_df, InsultDataset
from library.model import InsultModel
from library.engine import fit
from library.utils import set_seed, get_score


def run_demo():
    # 1. Setup and Configuration
    print("Initializing Configuration...")
    set_seed(42)
    config = ModelConfig()

    # Override config for speed and demonstration purposes
    # We use a tiny BERT model to ensure the code runs in seconds
    config.model_name = "prajjwal1/bert-tiny"
    config.epochs = 1
    config.train_batch_size = 2
    config.valid_batch_size = 2
    config.working_dir = "./working/demo_run"
    config.debug = True

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    config.display()

    # 2. Data Loading
    print("\nLoading Data...")
    # Load raw dataframe
    df_train_full = load_dataset_df(config, split="train")

    # Slice to a tiny subset for demonstration
    df_subset = df_train_full.head(10).reset_index(drop=True)

    # Split into train/val for the demo (8 train, 2 val)
    train_texts = df_subset["Comment"].values[:8]
    train_labels = df_subset["Insult"].values[:8]
    val_texts = df_subset["Comment"].values[8:]
    val_labels = df_subset["Insult"].values[8:]

    print(f"Demo Train Size: {len(train_texts)}")
    print(f"Demo Val Size: {len(val_texts)}")

    # 3. Dataset & Dataloader
    print("\nPreparing Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = InsultDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_len=config.max_len,
        labels=train_labels,
    )

    val_dataset = InsultDataset(
        texts=val_texts, tokenizer=tokenizer, max_len=config.max_len, labels=val_labels
    )

    # Verify Dataset Logic
    assert len(train_dataset) == 8
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "label" in sample_item
    assert sample_item["input_ids"].shape == (config.max_len,)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple demo to avoid multiprocessing overhead
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=0
    )

    # 4. Model Initialization
    print("\nInitializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = InsultModel(config)
    model.to(device)

    # Verify Model Output Shape
    print("Verifying Model Forward Pass...")
    dummy_batch = next(iter(train_loader))
    dummy_ids = dummy_batch["input_ids"].to(device)
    dummy_mask = dummy_batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(dummy_ids, dummy_mask)

    # Output should be [batch_size, 1]
    assert outputs.shape == (
        config.train_batch_size,
        1,
    ), f"Expected output shape {(config.train_batch_size, 1)}, got {outputs.shape}"
    print("Model forward pass successful.")

    # 5. Training Setup (Optimizer & Scheduler)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # 6. Run Training (Engine)
    print("\nStarting Training Loop (Demo)...")
    # We pass fold=0 just for logging purposes
    trained_model, best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        fold=0,
    )

    print(f"\nTraining finished. Best AUC: {best_auc}")

    # 7. Verify Metric Calculation
    print("\nVerifying Metric Function...")
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    # AUC calculation:
    # Pairs: (0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)
    # Negatives: 0.1, 0.4. Positives: 0.35, 0.8
    # Comparisons:
    # 0.35 > 0.1 (Win)
    # 0.35 < 0.4 (Loss)
    # 0.8 > 0.1 (Win)
    # 0.8 > 0.4 (Win)
    # Total 3 wins out of 4 pairs = 0.75
    score = get_score(y_true, y_pred)
    print(f"Calculated AUC: {score}")
    assert score == 0.75, f"Metric verification failed. Expected 0.75, got {score}"

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
