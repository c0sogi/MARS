import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from library.config import Config
from library.utils import get_logger, seed_everything, compute_qwk
from library.data import EssayDataset
from library.model import EssayModel
from library.awp import AWP

# Initialize logger
logger = get_logger(os.path.join(Config.output_dir, "trainer.log"))


def get_optimizer_params(model, config):
    """
    Configures layer-wise learning rate decay (LLRD) for the model.
    """
    if not config.use_llrd:
        return model.parameters()

    # DeBERTa-v3-large specific parameter grouping
    named_parameters = list(model.named_parameters())

    # Define groups
    # 1. Embeddings
    # 2. Encoder Layers (0-23)
    # 3. Head (Pooling + FC)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # Base LR for the head
    lr = config.lr
    decay = config.llrd_decay

    # Group 1: Head (Pooling + Linear) - Highest LR
    head_params = []
    for name, params in named_parameters:
        if "backbone" not in name:
            head_params.append((name, params))

    optimizer_parameters.append(
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        }
    )

    # Group 2: Backbone Layers (Decaying LR)
    # DeBERTa-v3-large has 24 layers. We iterate backwards.
    # layer 23 gets lr, layer 22 gets lr * decay, etc.
    n_layers = model.backbone.config.num_hidden_layers

    for layer_idx in range(n_layers - 1, -1, -1):
        layer_lr = lr * (decay ** (n_layers - layer_idx))

        layer_params = []
        for name, params in named_parameters:
            if f"encoder.layer.{layer_idx}." in name:
                layer_params.append((name, params))

        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in layer_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
                "lr": layer_lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in layer_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }
        )

    # Group 3: Embeddings - Lowest LR
    embed_lr = lr * (decay ** (n_layers + 1))
    embed_params = []
    for name, params in named_parameters:
        if "embeddings" in name and "backbone" in name:
            embed_params.append((name, params))

    optimizer_parameters.append(
        {
            "params": [
                p for n, p in embed_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": embed_lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": [p for n, p in embed_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": embed_lr,
        }
    )

    return optimizer_parameters


def train_one_epoch(
    epoch, model, train_loader, optimizer, scheduler, device, awp, scaler, config
):
    """
    Trains the model for one epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.MSELoss()

    for step, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)
        meta_features = data["meta_features"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Context
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if config.use_mixed_precision else torch.float32,
        ):
            outputs = model(input_ids, attention_mask, meta_features)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss = loss / config.gradient_accumulation_steps

        # Backward Pass (Clean)
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation (AWP)
        if config.use_awp and epoch >= config.awp_start_epoch:
            # Perturb weights based on gradients accumulated so far
            awp.attack()

            # Forward pass with perturbed weights
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if config.use_mixed_precision else torch.float32,
            ):
                outputs_adv = model(input_ids, attention_mask, meta_features)
                loss_adv = criterion(outputs_adv["logits"], labels)
                loss_adv = loss_adv / config.gradient_accumulation_steps

            # Backward pass with perturbed weights
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * config.gradient_accumulation_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, val_loader, device, config):
    """
    Validates the model. Returns loss, score, predictions, and embeddings.
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    preds = []
    labels_list = []
    embeddings_list = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in val_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            meta_features = data["meta_features"].to(device)

            batch_size = input_ids.size(0)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if config.use_mixed_precision else torch.float32,
            ):
                outputs = model(input_ids, attention_mask, meta_features)
                logits = outputs["logits"]
                loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.extend(logits.detach().float().cpu().numpy())
            labels_list.extend(labels.detach().cpu().numpy())
            embeddings_list.append(outputs["embeddings"].detach().float().cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)

    # Compute Metric
    val_score = compute_qwk(labels_list, preds)

    return epoch_loss, val_score, np.array(preds), embeddings


def run_fold(fold, train_df, val_df, config):
    """
    Runs training for a single fold.
    """
    logger.info(f"=== Starting Fold {fold} ===")

    seed_everything(config.seed + fold)
    device = config.device

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Datasets
    train_dataset = EssayDataset(train_df, tokenizer, config, is_train=True)
    val_dataset = EssayDataset(val_df, tokenizer, config, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = EssayModel(pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer_grouped_parameters = get_optimizer_params(model, config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-6)

    num_train_steps = int(
        len(train_dataset)
        / config.batch_size
        / config.gradient_accumulation_steps
        * config.epochs
    )

    if config.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * config.warmup_ratio),
            num_training_steps=num_train_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

    # AWP & Scaler
    # Scaler is not needed for bfloat16 or float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    awp = AWP(
        model,
        optimizer,
        adv_lr=config.awp_lr,
        adv_eps=config.awp_eps,
        start_epoch=config.awp_start_epoch,
        scaler=scaler,
    )

    # Training Loop
    best_loss = float("inf")
    best_score = -1.0

    save_path = os.path.join(config.model_dir, f"backbone_fold_{fold}.pth")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            awp,
            scaler,
            config,
        )
        val_loss, val_score, _, _ = valid_one_epoch(model, val_loader, device, config)

        logger.info(
            f"Fold {fold} | Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val QWK: {val_score:.6f}"
        )

        # Save Best Model (Monitoring RMSE Loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            logger.info(
                f"New best model saved for Fold {fold} with Loss: {best_loss:.6f}"
            )

    # Load Best Model to generate OOF Embeddings
    logger.info(f"Loading best model for Fold {fold} to generate OOF embeddings...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    _, final_score, final_preds, final_embeddings = valid_one_epoch(
        model, val_loader, device, config
    )
    logger.info(f"Final Best Score for Fold {fold}: {final_score:.6f}")

    # Save OOF Data for Stacking
    oof_emb_path = os.path.join(config.cache_dir, f"oof_embeddings_fold_{fold}.npy")
    oof_target_path = os.path.join(config.cache_dir, f"oof_targets_fold_{fold}.npy")
    oof_ids_path = os.path.join(config.cache_dir, f"oof_ids_fold_{fold}.npy")

    np.save(oof_emb_path, final_embeddings)
    np.save(oof_target_path, val_df["score"].values)
    np.save(oof_ids_path, val_df["essay_id"].values)

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader, awp, scaler
    torch.cuda.empty_cache()
    gc.collect()


def train_backbone(train_df, config):
    """
    Main entry point to train the backbone across all folds.
    """
    logger.info("Starting Backbone Training (5-Fold CV)...")

    for fold in range(config.n_folds):
        # Split data
        fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        run_fold(fold, fold_train_df, fold_val_df, config)

    logger.info("Backbone training complete.")
