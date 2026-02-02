import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler

from library.config import Config
from library.utils import get_logger, compute_qwk
from library.data import get_dataloaders
from library.model import EssayModel
from library.awp import AWP

# Initialize logger
logger = get_logger(os.path.join(Config.WORKING_DIR, "output", "trainer.log"))


def get_optimizer_params(model, encoder_lr, weight_decay, llrd_decay):
    """
    Configures Layer-wise Learning Rate Decay (LLRD) for DeBERTa-v3-large.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # DeBERTa-v3-large has 24 layers.
    # We assign layer indices:
    # Embeddings: 0
    # Layers 0-23: 1-24
    # Head (Pooling+Linear): 25

    optimizer_grouped_parameters = []

    # 1. Identify layers and assign LRs
    for name, params in param_optimizer:
        if not params.requires_grad:
            continue

        # Determine layer ID
        if "embeddings" in name or "shared" in name:
            layer_id = 0
        elif "encoder.layer" in name:
            # Extract layer number
            try:
                # name format: backbone.encoder.layer.15.output...
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])
                layer_id = layer_idx + 1
            except ValueError:
                layer_id = 0  # Fallback
        else:
            # Head parameters (pooling, fc, etc.)
            layer_id = 25

        # Calculate LR for this layer
        # Head gets base_lr, lower layers get decayed lr
        # lr = base_lr * (decay ^ (max_layer - current_layer))
        lr = encoder_lr * (llrd_decay ** (25 - layer_id))

        # Weight decay logic
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters.append(
                {"params": [params], "weight_decay": 0.0, "lr": lr}
            )
        else:
            optimizer_grouped_parameters.append(
                {"params": [params], "weight_decay": weight_decay, "lr": lr}
            )

    return optimizer_grouped_parameters


def validate_fold(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns metrics, predictions, embeddings, and targets.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_embeds = []
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_ids = batch["batch_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask, batch_ids)
                logits = outputs["logits"].squeeze(-1)
                embeddings = outputs["embeddings"]
                loss = criterion(logits, labels)

            total_loss += loss.item()

            all_preds.append(logits.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
            all_embeds.append(embeddings.float().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_embeds = np.concatenate(all_embeds, axis=0)

    # Calculate Metrics
    avg_loss = total_loss / len(dataloader)
    qwk = compute_qwk(all_labels, all_preds)

    metrics = {"loss": avg_loss, "qwk": qwk}

    return metrics, all_preds, all_embeds, all_labels


def train_fold(fold_idx):
    """
    Trains the model for a single fold.
    """
    logger.info(f"=== Starting Training for Fold {fold_idx} ===")

    # 1. Load Data
    # Note: In this setup, get_dataloaders returns fixed splits based on metadata.
    # We use the provided loaders.
    train_loader, val_loader, _ = get_dataloaders(
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
    )

    # 2. Initialize Model
    device = Config.DEVICE
    model = EssayModel()
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = int(len(train_loader) * Config.EPOCHS / Config.GRAD_ACCUM_STEPS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 4. AWP & Scaler
    awp = AWP(model, optimizer, adv_lr=Config.AWP_LR, adv_eps=Config.AWP_EPS)
    scaler = GradScaler()
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_qwk = -1.0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        # Enable AWP after specified epoch
        use_awp = Config.USE_AWP and (epoch >= Config.AWP_START_EPOCH)
        if use_awp:
            logger.info(f"Epoch {epoch+1}: AWP Enabled.")

        for step, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_ids = batch["batch_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward Pass
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask, batch_ids)
                loss = criterion(outputs["logits"].squeeze(-1), labels)

            # Normalize loss for accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()

            # AWP Attack Step
            if use_awp:
                awp.attack_step()
                with torch.amp.autocast("cuda"):
                    outputs_adv = model(input_ids, attention_mask, batch_ids)
                    loss_adv = criterion(outputs_adv["logits"].squeeze(-1), labels)
                    loss_adv = loss_adv / Config.GRAD_ACCUM_STEPS

                scaler.scale(loss_adv).backward()
                awp.restore()

            # Optimizer Step
            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * Config.GRAD_ACCUM_STEPS

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_metrics, val_preds, val_embeds, val_targets = validate_fold(
            model, val_loader, device
        )

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val QWK: {val_metrics['qwk']:.6f}"
        )

        # Save Best Model & OOF
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            logger.info(f"New Best QWK: {best_qwk:.6f}. Saving checkpoint and OOF...")

            # Save Checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, f"backbone_fold_{fold_idx}.pth"),
            )

            # Save OOF Embeddings for Stacking
            np.save(
                os.path.join(Config.CACHE_DIR, f"oof_embeddings_fold_{fold_idx}.npy"),
                val_embeds,
            )
            np.save(
                os.path.join(Config.CACHE_DIR, f"oof_targets_fold_{fold_idx}.npy"),
                val_targets,
            )
            # We don't strictly need IDs if order is preserved, but saving targets ensures alignment

    return best_qwk


def predict_test_set():
    """
    Generates predictions for the test set using the best trained model(s).
    Averages predictions from available folds.
    """
    logger.info("Generating Test Predictions...")

    _, _, test_loader = get_dataloaders(load_cached_data=True)
    device = Config.DEVICE

    # Placeholder for ensemble predictions
    fold_preds = []

    # Iterate over folds
    for fold in range(Config.N_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"backbone_fold_{fold}.pth")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint for fold {fold} not found. Skipping.")
            continue

        logger.info(f"Loading model for fold {fold}...")
        model = EssayModel()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_ids = batch["batch_ids"].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, attention_mask, batch_ids)
                    logits = outputs["logits"].squeeze(-1)

                preds.append(logits.float().cpu().numpy())

        fold_preds.append(np.concatenate(preds))

        # Clear memory
        del model
        torch.cuda.empty_cache()

    if not fold_preds:
        logger.error("No models found for inference!")
        return

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Clip and Round
    final_scores = np.clip(avg_preds, 1, 6).round().astype(int)

    # Save Submission
    submission = pd.read_csv(Config.TEST_METADATA_PATH)
    # Ensure alignment: test_loader preserves order of test_metadata
    submission["score"] = final_scores

    # Keep only required columns
    submission = submission[["essay_id", "score"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
