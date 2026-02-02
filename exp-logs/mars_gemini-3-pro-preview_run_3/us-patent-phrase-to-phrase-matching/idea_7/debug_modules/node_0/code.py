import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as transformers_logging

# Import from the provided library
from library.config import Config
from library.cpc_utils import CPCHelper
from library.dataset import PearsonDataset, MLMDataset
from library.model import CustomDeberta
from library.loss import CompositeLoss
from library.awp import AWP
from library.ema import ModelEMA
from library.engine import train_fn, eval_fn, predict_fn, run_dapt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()


def run_demo():
    print("======================================================")
    print("      Phrase Matching Library Demo Execution          ")
    print("======================================================")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for speed...")

    # Override Config for speed
    Config.seed = 42
    Config.debug = True
    Config.debug_sample_size = 50  # Use only 50 samples
    Config.model_backbone = (
        "prajjwal1/bert-tiny"  # Tiny model for fast CPU/GPU execution
    )
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.dapt_epochs = 1
    Config.dapt_batch_size = 4
    Config.print_freq = 5
    Config.working_dir = "./working/demo_run"
    Config.dapt_model_path = os.path.join(Config.working_dir, "dapt_model")

    # Ensure working directory exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.device = device
    print(f"    Device: {device}")
    print(f"    Model Backbone: {Config.model_backbone}")

    # ---------------------------------------------------------
    # 2. Data Processing (CPCHelper)
    # ---------------------------------------------------------
    print("\n[2] Testing CPCHelper (Context Expansion)...")
    cpc_helper = CPCHelper()

    # Test single context expansion
    ctx_code = "A47"
    ctx_text = cpc_helper.get_context_text(ctx_code)
    print(f"    Context '{ctx_code}' -> '{ctx_text}'")
    assert (
        isinstance(ctx_text, str) and len(ctx_text) > 0
    ), "CPCHelper failed to expand context"

    # Load raw data subset
    train_df_full = pd.read_csv(Config.train_path)
    val_df_full = pd.read_csv(Config.val_path)

    # Subset for demo
    train_df = train_df_full.head(Config.debug_sample_size).copy()
    val_df = val_df_full.head(Config.debug_sample_size).copy()

    # Process dataset (add context_text)
    train_df = cpc_helper.process_dataset(
        train_df, cache_path=None, load_cached_data=False
    )
    val_df = cpc_helper.process_dataset(val_df, cache_path=None, load_cached_data=False)

    assert (
        "context_text" in train_df.columns
    ), "context_text column missing after processing"
    print(f"    Processed {len(train_df)} training samples.")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader
    # ---------------------------------------------------------
    print("\n[3] Testing PearsonDataset & DataLoader...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_backbone)

    train_dataset = PearsonDataset(train_df, tokenizer, max_length=64, mode="train")
    val_dataset = PearsonDataset(val_df, tokenizer, max_length=64, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )

    # Verify batch structure
    batch = next(iter(train_loader))
    print(f"    Batch keys: {batch.keys()}")
    assert "input_ids" in batch
    assert "target" in batch
    assert batch["input_ids"].shape[0] == Config.train_batch_size
    print("    Dataset and DataLoader initialized successfully.")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing CustomDeberta Model...")
    # Initialize model (pretrained=True downloads the tiny bert weights)
    model = CustomDeberta(model_path=Config.model_backbone, pretrained=True)
    model.to(device)

    # Forward pass check
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    reg_logits, cls_logits = model(input_ids, attention_mask, token_type_ids)

    print(f"    Regression Logits Shape: {reg_logits.shape}")
    print(f"    Classification Logits Shape: {cls_logits.shape}")

    assert reg_logits.shape == (
        Config.train_batch_size,
        1,
    ), "Incorrect regression output shape"
    assert cls_logits.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), "Incorrect classification output shape"

    # ---------------------------------------------------------
    # 5. Loss Function
    # ---------------------------------------------------------
    print("\n[5] Testing CompositeLoss...")
    loss_fn = CompositeLoss(Config)
    targets = batch["target"].to(device)

    loss, metrics = loss_fn(reg_logits, cls_logits, targets)

    print(f"    Total Loss: {loss.item():.4f}")
    print(f"    Metrics: {metrics}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert "loss_pearson" in metrics, "Pearson loss missing from metrics"

    # ---------------------------------------------------------
    # 6. Advanced Training Utilities (AWP & EMA)
    # ---------------------------------------------------------
    print("\n[6] Testing AWP and EMA...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize AWP
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-2, start_epoch=0)

    # Initialize EMA
    ema = ModelEMA(model, decay=0.99)

    # Simulate one training step with AWP and EMA
    model.train()
    optimizer.zero_grad()

    # 1. Forward
    r_logits, c_logits = model(input_ids, attention_mask, token_type_ids)
    loss_val, _ = loss_fn(r_logits, c_logits, targets)

    # 2. Backward
    loss_val.backward()

    # 3. AWP Attack
    # Check weight before attack
    param_name = list(model.named_parameters())[0][0]  # Get first param name
    weight_before = dict(model.named_parameters())[param_name].clone()

    awp.attack_step()

    weight_attacked = dict(model.named_parameters())[param_name]
    # Weights should be different (unless gradient was exactly zero, which is unlikely)
    diff = torch.norm(weight_before - weight_attacked).item()
    print(f"    AWP Perturbation Norm: {diff:.6f}")

    # 4. AWP Restore
    awp.restore()
    weight_restored = dict(model.named_parameters())[param_name]
    assert torch.allclose(weight_before, weight_restored), "AWP restore failed"

    # 5. Optimizer Step
    optimizer.step()

    # 6. EMA Update
    ema.update(model)
    assert len(ema.shadow) > 0, "EMA shadow weights not registered"
    print("    AWP and EMA steps verified.")

    # ---------------------------------------------------------
    # 7. Engine Functions (Train/Eval/Predict)
    # ---------------------------------------------------------
    print("\n[7] Testing Engine Functions...")

    # Train Loop
    print("    Running train_fn (1 epoch)...")
    epoch_loss = train_fn(
        model,
        train_loader,
        optimizer,
        scheduler=None,
        device=device,
        epoch=0,
        config=Config,
        awp=awp,
        ema=ema,
        loss_fn=loss_fn,
    )
    assert epoch_loss > 0, "Training loss should be positive"

    # Eval Loop
    print("    Running eval_fn...")
    pearson_score = eval_fn(model, val_loader, device, Config, ema=ema)
    print(f"    Validation Pearson Score: {pearson_score:.4f}")
    assert -1.0 <= pearson_score <= 1.0, "Pearson score out of range"

    # Predict Loop
    print("    Running predict_fn...")
    preds = predict_fn(model, val_loader, device, Config)
    print(f"    Predictions shape: {preds.shape}")
    assert preds.shape[0] == len(val_dataset), "Prediction count mismatch"
    assert (preds >= 0).all() and (preds <= 1).all(), "Predictions out of range [0, 1]"

    # ---------------------------------------------------------
    # 8. Domain-Adaptive Pre-training (DAPT)
    # ---------------------------------------------------------
    print("\n[8] Testing DAPT (MLM)...")

    # Create MLM Dataset (using the small subset logic inside dataset.py requires overriding paths or mocking)
    # Since dataset.py loads from Config.train_path, we create a temporary csv there for the demo
    # to avoid processing the full 30k rows which takes time.

    temp_train_path = os.path.join(Config.working_dir, "temp_train.csv")
    train_df.to_csv(temp_train_path, index=False)

    # Temporarily point Config to small file
    original_train_path = Config.train_path
    Config.train_path = temp_train_path

    try:
        # Run DAPT
        # Note: run_dapt creates its own model/tokenizer internally based on Config
        run_dapt(Config)

        assert os.path.exists(
            Config.dapt_model_path
        ), "DAPT model directory not created"
        assert os.path.exists(
            os.path.join(Config.dapt_model_path, "config.json")
        ), "DAPT model config not saved"
        print("    DAPT execution successful.")

    finally:
        # Restore Config
        Config.train_path = original_train_path

    print("\n======================================================")
    print("      Demo Completed Successfully                     ")
    print("======================================================")


if __name__ == "__main__":
    run_demo()
