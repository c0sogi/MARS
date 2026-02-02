import os
import sys
import shutil
import warnings
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, get_logger, AWP, EMA
from library.loss import HybridLoss
from library.cpc_loader import CPCLoader
from library.dataset import CPCDataset, prepare_data
from library.model import DebertaV3Model
from library.engine import get_optimizer_params, train_fn, valid_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("======================================================")
    print("   Patent Phrase Matching: Library Usage Demonstration")
    print("======================================================")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override CFG settings for speed and resource efficiency
    CFG.debug = True
    CFG.debug_sample_size = 20  # Use only 20 samples
    CFG.model_name = "prajjwal1/bert-tiny"  # Use a tiny model for CPU/fast execution
    CFG.batch_size = 4
    CFG.epochs = 1
    CFG.working_dir = "./working/demo_execution"
    CFG.context_map_path = os.path.join(CFG.working_dir, "context_map.parquet")
    CFG.train_cache_path = os.path.join(CFG.working_dir, "train_cache.parquet")
    CFG.val_cache_path = os.path.join(CFG.working_dir, "val_cache.parquet")
    CFG.test_cache_path = os.path.join(CFG.working_dir, "test_cache.parquet")

    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    seed_everything(CFG.seed)
    print(f"    Model: {CFG.model_name}")
    print(f"    Debug Mode: {CFG.debug}")
    print(f"    Working Dir: {CFG.working_dir}")

    # ---------------------------------------------------------
    # 2. Data Processing (CPCLoader & prepare_data)
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Data Processing...")

    # Load and process training data (subset due to debug=True)
    df_train = prepare_data(config=CFG, split="train", load_cached_data=False)

    print(f"    Loaded Train DataFrame Shape: {df_train.shape}")
    print(f"    Columns: {list(df_train.columns)}")

    # Assertions
    assert (
        "context_text" in df_train.columns
    ), "context_text column missing after processing"
    assert (
        len(df_train) == CFG.debug_sample_size
    ), f"Expected {CFG.debug_sample_size} samples, got {len(df_train)}"

    sample_context = df_train.iloc[0]["context_text"]
    print(
        f"    Sample Context Expansion: '{df_train.iloc[0]['context']}': '{sample_context}'"
    )

    # ---------------------------------------------------------
    # 3. Dataset & Tokenizer
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Dataset & Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    train_dataset = CPCDataset(df_train, tokenizer, config=CFG, mode="train")

    # Fetch one sample
    sample_data, sample_label = train_dataset[0]

    print("    Fetched one sample from CPCDataset:")
    print(f"      Input IDs Shape: {sample_data['input_ids'].shape}")
    print(f"      Attention Mask Shape: {sample_data['attention_mask'].shape}")
    print(f"      Label Value: {sample_label}")

    # Assertions
    assert torch.is_tensor(sample_data["input_ids"])
    assert sample_data["input_ids"].dim() == 1
    assert isinstance(sample_label, torch.Tensor)

    # ---------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Demonstrating Model Initialization & Forward Pass...")

    device = CFG.device
    print(f"    Device: {device}")

    model = DebertaV3Model(config=CFG, pretrained=True)
    model.to(device)

    # Create a batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    batch_inputs, batch_labels = next(iter(train_loader))

    # Move to device
    for k, v in batch_inputs.items():
        batch_inputs[k] = v.to(device)
    batch_labels = batch_labels.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch_inputs["input_ids"],
            batch_inputs["attention_mask"],
            batch_inputs.get("token_type_ids"),
        )

    print("    Forward Pass Successful.")
    print(f"      Score Output Shape: {outputs['score'].shape}")
    print(f"      Logits Output Shape: {outputs['logits'].shape}")

    # Assertions
    assert outputs["score"].shape == (CFG.batch_size, 1)
    assert outputs["logits"].shape == (CFG.batch_size, CFG.loss_config["ce_bins"])

    # ---------------------------------------------------------
    # 5. Loss Function
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Hybrid Loss...")

    criterion = HybridLoss(config=CFG)
    loss = criterion(outputs, batch_labels)

    print(f"    Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss)
    assert loss.item() >= 0

    # ---------------------------------------------------------
    # 6. Optimizer, AWP, EMA
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Optimizer, AWP, and EMA...")

    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        decoder_lr=CFG.head_lr,
        weight_decay=CFG.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
    )

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
    )
    print("    AWP Initialized.")

    # Initialize EMA
    ema = EMA(model, decay=CFG.ema_decay)
    print("    EMA Initialized.")

    # ---------------------------------------------------------
    # 7. Engine (Train & Validation Loop)
    # ---------------------------------------------------------
    print("\n[7] Demonstrating Training Loop (Engine)...")

    # Scheduler
    num_train_steps = int(len(df_train) / CFG.batch_size * CFG.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_train_steps, eta_min=1e-6
    )

    print(
        f"    Starting training for {CFG.epochs} epoch(s) on {len(df_train)} samples..."
    )

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
        config=CFG,
    )

    print(f"    Training Epoch 1 Complete. Avg Loss: {avg_loss:.4f}")

    # Validation
    print("    Running Validation...")
    # Using train set as validation set just for demo purposes
    val_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size * 2, shuffle=False, num_workers=0
    )

    val_loss, val_score = valid_fn(
        valid_loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        config=CFG,
    )

    print(
        f"    Validation Complete. Loss: {val_loss:.4f}, Pearson Score: {val_score:.4f}"
    )

    # Assertions
    assert not np.isnan(val_score), "Validation score is NaN"

    print("\n======================================================")
    print("   Demonstration Completed Successfully")
    print("======================================================")


if __name__ == "__main__":
    run_demo()
