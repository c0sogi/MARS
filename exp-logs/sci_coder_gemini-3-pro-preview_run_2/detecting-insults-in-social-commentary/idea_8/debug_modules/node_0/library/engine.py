import os
import time
import math
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    logging,
)
from datasets import load_dataset
from library.utils import get_score, Logger
from library.data import prepare_tapt_data

# Suppress HF warnings
logging.set_verbosity_error()


def get_optimizer_params(model, config):
    """
    Configures layer-wise learning rate decay (LLRD) for the model.
    """
    if not config.use_llrd:
        return model.parameters()

    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Base learning rate
    lr = config.learning_rate
    weight_decay = config.weight_decay
    decay_factor = config.llrd_decay

    # Identify backbone layers
    # DeBERTa-v3-Large has 24 layers.
    n_layers = 24
    if "base" in config.model_name:
        n_layers = 12
    elif "xsmall" in config.model_name or "tiny" in config.model_name:
        n_layers = 3

    optimizer_grouped_parameters = []

    for name, params in named_parameters:
        if not params.requires_grad:
            continue

        layer_lr = lr

        # Apply LLRD
        if "backbone.embeddings" in name:
            layer_lr = lr * (decay_factor ** (n_layers + 1))
        elif "backbone.encoder.layer" in name:
            try:
                # name format: backbone.encoder.layer.15.output...
                parts = name.split(".")
                # Find the index after 'layer'
                if "layer" in parts:
                    layer_idx = int(parts[parts.index("layer") + 1])
                    # Higher layer index = closer to output = higher LR
                    layer_dist_from_top = n_layers - 1 - layer_idx
                    layer_lr = lr * (decay_factor ** (layer_dist_from_top + 1))
            except (ValueError, IndexError):
                layer_lr = lr

        # Apply weight decay filter
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters.append(
                {"params": [params], "weight_decay": 0.0, "lr": layer_lr}
            )
        else:
            optimizer_grouped_parameters.append(
                {"params": [params], "weight_decay": weight_decay, "lr": layer_lr}
            )

    return optimizer_grouped_parameters


def run_tapt(config, logger):
    """
    Executes Task-Adaptive Pre-Training (TAPT) using MLM.
    """
    if not config.use_tapt:
        logger.log("TAPT is disabled in config.")
        return

    logger.log(f"Starting TAPT on {config.model_name}...")

    # Prepare data
    corpus_path = prepare_tapt_data(config)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load Dataset
    datasets = load_dataset("text", data_files={"train": corpus_path})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=config.num_workers,
    )

    # Data Collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
    )

    # Model
    model = AutoModelForMaskedLM.from_pretrained(config.model_name)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config.tapt_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.tapt_epochs,
        per_device_train_batch_size=config.tapt_batch_size,
        learning_rate=config.tapt_lr,
        weight_decay=1e-2,
        save_strategy="no",  # Save only at the end
        logging_steps=50,
        seed=config.seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=config.num_workers,
        disable_tqdm=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    # Train
    logger.log("Running MLM training...")
    trainer.train()

    # Save
    logger.log(f"Saving TAPT model to {config.tapt_output_dir}")
    trainer.save_model(config.tapt_output_dir)
    tokenizer.save_pretrained(config.tapt_output_dir)


def train_fn(
    train_loader, model, optimizer, device, scheduler, epoch, config, awp=None
):
    """
    Training function for one epoch with optional AWP.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = []

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), targets.view(-1))

        # Scale loss and backward
        scaler.scale(loss).backward()

        # AWP Attack
        if config.use_awp and awp is not None and epoch >= config.awp_start_epoch:
            # Unscale gradients to allow AWP to access true gradients
            scaler.unscale_(optimizer)

            # Attack step (perturb weights)
            awp.attack_step()

            # Forward pass with perturbed weights
            with torch.cuda.amp.autocast(enabled=True):
                logits_adv = model(input_ids, attention_mask)
                loss_adv = nn.BCEWithLogitsLoss()(logits_adv.view(-1), targets.view(-1))

            # Backward pass with adversarial loss
            # Accumulate gradients (no zero_grad here)
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp._restore()

        # Optimizer Step
        # Check if gradients are already unscaled (if AWP ran) or need unscaling
        if not (config.use_awp and awp is not None and epoch >= config.awp_start_epoch):
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def valid_fn(val_loader, model, device):
    """
    Validation function.
    """
    model.eval()
    preds = []
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            # Forward
            logits = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), targets.view(-1))

            losses.append(loss.item())
            preds.append(logits.sigmoid().view(-1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    preds = np.concatenate(preds)

    return avg_loss, preds


def inference_fn(test_loader, model, device):
    """
    Inference function for test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            preds.append(logits.sigmoid().view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    return preds
