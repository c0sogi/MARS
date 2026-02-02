import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_logger
from library.data import (
    prepare_supervised_data,
    get_tokenizer,
    InsultDataset,
    MLMDataset,
    prepare_pseudo_data,
    prepare_tapt_data,
)
from library.model import InsultModel
from library.awp import AWP
from library.engine import (
    train_fn,
    train_fn_awp,
    evaluate_fn,
    inference_fn,
    train_mlm,
    run_training,
)


def run_demo():
    # 1. Setup and Configuration Override
    print(">>> Setting up configuration for demo...")
    set_seed(42)

    # Override Config for speed
    Config.model_name = "prajjwal1/bert-tiny"  # Tiny model for fast execution
    Config.working_dir = "./working/demo_run"
    Config.tapt_epochs = 1
    Config.teacher_epochs = 1
    Config.student_epochs = 1
    Config.teacher_batch_size = 4
    Config.student_batch_size = 4
    Config.max_len = 32  # Short sequence length for speed
    Config.setup()  # Create directories

    logger = get_logger(os.path.join(Config.working_dir, "demo_log.txt"))
    device = Config.device
    logger.info(f"Device: {device}")

    # 2. Data Preparation
    print("\n>>> Testing Data Preparation...")
    # Load data (forcing no cache to test processing logic)
    df_train, df_val, df_test = prepare_supervised_data(load_cached_data=False)

    # Subsample for speed
    df_train = df_train.head(50).reset_index(drop=True)
    df_val = df_val.head(20).reset_index(drop=True)
    df_test = df_test.head(20).reset_index(drop=True)

    logger.info(f"Subsampled Train shape: {df_train.shape}")
    logger.info(f"Subsampled Val shape: {df_val.shape}")
    logger.info(f"Subsampled Test shape: {df_test.shape}")

    # 3. Tokenizer and Dataset
    print("\n>>> Testing Tokenizer and Dataset...")
    tokenizer = get_tokenizer()

    # Test InsultDataset (Supervised)
    train_ds = InsultDataset(df_train, tokenizer, Config.max_len, is_test=False)
    sample_item = train_ds[0]

    # Assertions for Dataset
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "target" in sample_item
    assert sample_item["input_ids"].shape[0] == Config.max_len
    assert isinstance(sample_item["target"], torch.Tensor)
    logger.info("InsultDataset verification passed.")

    # Test MLMDataset (Unsupervised/TAPT)
    tapt_texts = prepare_tapt_data(df_train, df_val, df_test)
    mlm_ds = MLMDataset(tapt_texts[:10], tokenizer, Config.max_len)
    mlm_item = mlm_ds[0]
    assert "labels" in mlm_item
    logger.info("MLMDataset verification passed.")

    # 4. Model Initialization
    print("\n>>> Testing Model Initialization...")
    model = InsultModel(pretrained=True)
    model.to(device)

    # Test Forward Pass
    dummy_input = sample_item["input_ids"].unsqueeze(0).to(device)
    dummy_mask = sample_item["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(dummy_input, dummy_mask)

    # Assert output shape [batch_size, num_classes]
    assert output.shape == (1, 1)
    logger.info(f"Model forward pass successful. Output shape: {output.shape}")

    # 5. Training Loop Components
    print("\n>>> Testing Training Components...")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.teacher_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        InsultDataset(df_val, tokenizer, Config.max_len),
        batch_size=Config.teacher_batch_size * 2,
        shuffle=False,
    )

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.teacher_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
    )

    # Test Standard Training Step (train_fn)
    logger.info("Testing standard training function...")
    train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch=0)
    assert not np.isnan(train_loss)
    logger.info(f"Train fn passed. Loss: {train_loss:.4f}")

    # Test Evaluation (evaluate_fn)
    logger.info("Testing evaluation function...")
    val_loss, val_auc = evaluate_fn(model, val_loader, device)
    assert not np.isnan(val_loss)
    assert 0.0 <= val_auc <= 1.0
    logger.info(f"Evaluate fn passed. Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 6. Adversarial Weight Perturbation (AWP)
    print("\n>>> Testing AWP Integration...")
    # Re-initialize model/optimizer for clean state
    model_awp = InsultModel(pretrained=True).to(device)
    optimizer_awp = AdamW(model_awp.parameters(), lr=Config.teacher_lr)

    awp = AWP(
        model_awp,
        optimizer_awp,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=0,  # Force start immediately for demo
    )

    logger.info("Testing training with AWP...")
    awp_loss = train_fn_awp(
        model_awp, train_loader, optimizer_awp, None, device, epoch=0, awp=awp
    )
    assert not np.isnan(awp_loss)
    logger.info(f"AWP training passed. Loss: {awp_loss:.4f}")

    # 7. Inference
    print("\n>>> Testing Inference...")
    test_ds = InsultDataset(df_test, tokenizer, Config.max_len, is_test=True)
    test_loader = DataLoader(
        test_ds, batch_size=Config.teacher_batch_size * 2, shuffle=False
    )

    predictions = inference_fn(model, test_loader, device)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(df_test)
    assert np.all((predictions >= 0) & (predictions <= 1))
    logger.info("Inference passed.")

    # 8. Pseudo-Labeling Logic
    print("\n>>> Testing Pseudo-Labeling...")
    # Create synthetic probabilities to ensure we trigger the thresholds
    # First 5 -> High confidence Insult (0.95)
    # Next 5 -> High confidence Neutral (0.05)
    # Rest -> Middle (0.5)
    synthetic_preds = np.array([0.95] * 5 + [0.05] * 5 + [0.5] * (len(df_test) - 10))

    # Ensure Config thresholds match our synthetic data intent
    Config.conf_thresh_high = 0.9
    Config.conf_thresh_low = 0.1

    df_augmented = prepare_pseudo_data(df_train, df_test, synthetic_preds)

    # We expect original train (50) + 5 insults + 5 neutrals = 60
    expected_len = len(df_train) + 10
    assert len(df_augmented) == expected_len
    assert Config.target_col in df_augmented.columns
    logger.info(
        f"Pseudo-labeling passed. Augmented size: {len(df_augmented)} (Expected: {expected_len})"
    )

    # 9. Full Training Run Integration
    print("\n>>> Testing Full Training Run Wrapper...")
    save_path = os.path.join(Config.working_dir, "model_demo.pth")
    # Reset model
    model = InsultModel(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=Config.teacher_lr)

    trained_model, best_auc = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=1,
        patience=1,
        save_path=save_path,
        use_awp=False,
    )

    assert os.path.exists(save_path)
    logger.info("Full training run passed.")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
