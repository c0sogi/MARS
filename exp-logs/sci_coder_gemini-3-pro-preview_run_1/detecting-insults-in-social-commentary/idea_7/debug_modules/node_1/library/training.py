import os
import time
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import save_checkpoint, load_checkpoint, format_time
from library.data_processing import create_dataloader
from library.model import HybridDeberta, AWP
from library.model import train_one_epoch as train_fn
from library.model import validate as valid_fn


def run_fold(
    fold: int,
    train_df,
    val_df,
    train_svd,
    val_svd,
    tokenizer,
    device,
    stage_name: str = "Teacher",
):
    """
    Manages the full training lifecycle for a single fold.

    Args:
        fold (int): The current fold number.
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        train_svd (np.ndarray): SVD features for training data.
        val_svd (np.ndarray): SVD features for validation data.
        tokenizer: The tokenizer instance.
        device: The torch device.
        stage_name (str): 'Teacher' or 'Student' for naming checkpoints.

    Returns:
        tuple: (best_model, best_auc)
    """
    print(f"\n[{stage_name}] Starting Fold {fold}")

    # Create DataLoaders
    train_loader = create_dataloader(
        train_df, train_svd, tokenizer, is_train=True, shuffle=True
    )
    val_loader = create_dataloader(
        val_df, val_svd, tokenizer, is_train=False, shuffle=False
    )

    # Initialize Model
    model = HybridDeberta(config=Config).to(device)

    # Optimizer with Differential Learning Rates
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_parameters)

    # Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Adversarial Weight Perturbation (AWP)
    awp = None
    if Config.USE_AWP:
        awp = AWP(model, optimizer, adv_lr=Config.AWP_LR, adv_eps=Config.AWP_EPS)

    best_auc = 0.0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"model_{stage_name.lower()}_fold_{fold}.bin"
    )

    # Training Loop
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Enable AWP after start epoch
        use_awp_epoch = Config.USE_AWP and (epoch >= Config.AWP_START_EPOCH)

        # Train
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            use_awp=use_awp_epoch,
            awp=awp,
        )

        # Validate
        val_loss, val_auc, _ = valid_fn(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision for AUC
        print(
            f"Epoch {epoch} | Time: {format_time(elapsed)} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"  >>> New Best AUC! Model saved to {best_model_path}")

    # Load best model state before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(load_checkpoint(best_model_path, device))

    return model, best_auc
