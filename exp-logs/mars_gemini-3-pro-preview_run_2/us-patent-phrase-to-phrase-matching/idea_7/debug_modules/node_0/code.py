import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_cpc_texts
from library.dataset import PearsonDataset
from library.model import HybridModel
from library.awp import AWP
from library.engine import train_fn, valid_fn


def run_demo():
    print(">>> Starting Phrase Similarity Model Demo")

    # 1. Configuration Overrides for Speed and Demo
    # We modify the Config class directly to run a fast, lightweight demonstration.
    print(">>> Configuring environment...")
    Config.debug = True  # Subsamples dataset to 100 rows
    Config.epochs = 2
    Config.batch_size = 8
    Config.gradient_accumulation_steps = 1
    Config.print_freq = 10
    Config.num_workers = 0  # Disable multiprocessing for simple script execution

    # Use a tiny model to ensure the demo runs in < 1 minute on CPU or GPU
    # This overrides the large DeBERTa model in the original config
    demo_model_name = "prajjwal1/bert-tiny"
    Config.models = [
        {
            "model_name": demo_model_name,
            "tokenizer_name": demo_model_name,
            "short_name": "bert_tiny_demo",
        }
    ]

    # Set device
    device = Config.device
    print(f"    Device: {device}")

    # Set seeds
    seed_everything(Config.seed)

    # 2. Verify Utility Functions
    print("\n>>> Verifying Utilities...")
    cpc_texts = get_cpc_texts()
    # Check direct mapping
    assert cpc_texts["A"] == "Human Necessities", "CPC Mapping for 'A' failed"
    # Check fallback mapping (A47 -> A)
    assert cpc_texts["A47"] == "Human Necessities", "CPC Fallback logic failed"
    print("    CPC Context mapping verified.")

    # 3. Dataset Preparation
    print("\n>>> Initializing Dataset and Tokenizer...")
    model_conf = Config.models[0]
    tokenizer = AutoTokenizer.from_pretrained(model_conf["tokenizer_name"])

    # Initialize Train Dataset
    # load_cached_data=False ensures we demonstrate the full processing pipeline
    train_ds = PearsonDataset(
        mode="train",
        tokenizer=tokenizer,
        max_length=64,
        short_name=model_conf["short_name"],
        load_cached_data=False,
    )

    # Initialize Validation Dataset
    val_ds = PearsonDataset(
        mode="val",
        tokenizer=tokenizer,
        max_length=64,
        short_name=model_conf["short_name"],
        load_cached_data=False,
    )

    print(f"    Train dataset size: {len(train_ds)}")
    print(f"    Val dataset size: {len(val_ds)}")

    # Verify Dataset Integrity
    sample = train_ds[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "structural_features",
        "labels",
        "scores",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample"

    # Verify shapes
    assert sample["input_ids"].dim() == 1
    assert sample["structural_features"].shape[0] == len(Config.structural_features)
    print("    Dataset structure verified.")

    # 4. DataLoader Setup
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # 5. Model Initialization
    print("\n>>> Initializing HybridModel...")
    model = HybridModel(
        model_name=model_conf["model_name"],
        num_classes=Config.num_classes,
        num_structural_features=len(Config.structural_features),
        pretrained=True,
    )
    model.to(device)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    dummy_batch = next(iter(train_loader))
    input_ids = dummy_batch["input_ids"].to(device)
    mask = dummy_batch["attention_mask"].to(device)
    feats = dummy_batch["structural_features"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, mask, feats)

    assert outputs.shape == (
        input_ids.size(0),
        Config.num_classes,
    ), f"Output shape mismatch. Expected {(input_ids.size(0), Config.num_classes)}, got {outputs.shape}"
    print("    Forward pass successful.")

    # 6. Training Components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs
    )

    # Initialize Adversarial Weight Perturbation (AWP)
    # We set start_epoch=1 to ensure it runs during our short demo loop
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-3, start_epoch=1)

    # 7. Execution Loop
    print("\n>>> Starting Training & Validation Loop...")
    for epoch in range(1, Config.epochs + 1):
        print(f"\n--- Epoch {epoch}/{Config.epochs} ---")

        # Train
        train_loss = train_fn(
            fold=0,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
            device=device,
            awp=awp,
        )
        print(f"    Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_score, _ = valid_fn(
            val_loader=val_loader, model=model, criterion=criterion, device=device
        )
        print(f"    Val Loss: {val_loss:.4f} | Pearson Score: {val_score:.4f}")

        # Sanity Checks
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert -1.0 <= val_score <= 1.0, f"Pearson score out of range: {val_score}"

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
