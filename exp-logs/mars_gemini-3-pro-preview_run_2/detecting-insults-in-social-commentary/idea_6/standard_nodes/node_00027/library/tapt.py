import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from library.config import Config
from library.utils import seed_everything, AverageMeter, Logger
from library.data import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    """
    Trains the model for one epoch using Masked Language Modeling objective.
    """
    model.train()
    loss_meter = AverageMeter()

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_meter.update(loss.item(), input_ids.size(0))

    elapsed = time.time() - start_time
    return loss_meter.avg, elapsed


def run_tapt():
    """
    Executes the Task-Adaptive Pre-Training (TAPT) pipeline.
    """
    # Setup
    seed_everything(Config.seed)
    os.makedirs(Config.tapt_output_dir, exist_ok=True)
    log_path = os.path.join(Config.tapt_output_dir, "tapt_log.txt")
    logger = Logger(log_path)

    device = Config.device
    logger.log(f"Device: {device}")
    logger.log("Starting Task-Adaptive Pre-Training (TAPT)...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Data Loaders
    # We only need the tapt_loader here
    tapt_loader, _, _, _ = get_dataloaders(tokenizer, load_cached_data=True)

    # Model
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.to(device)

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.tapt_weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.tapt_lr)

    # Scheduler
    num_train_steps = len(tapt_loader) * Config.tapt_epochs
    num_warmup_steps = int(num_train_steps * 0.1)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logger.log(f"Total Training Steps: {num_train_steps}")
    logger.log(f"Warmup Steps: {num_warmup_steps}")

    # Training Loop
    for epoch in range(1, Config.tapt_epochs + 1):
        avg_loss, elapsed = train_one_epoch(
            model, tapt_loader, optimizer, scheduler, device, scaler, epoch
        )

        logger.log(
            f"Epoch {epoch}/{Config.tapt_epochs} - "
            f"Avg Loss: {avg_loss:.8f} - "
            f"Time: {elapsed:.2f}s"
        )

    # Save Model
    save_path = os.path.join(Config.tapt_output_dir, "tapt_model.pth")
    logger.log(f"Saving TAPT model weights to {save_path}")

    # We save the state_dict. The InsultModel loader handles prefix stripping.
    torch.save(model.state_dict(), save_path)
    logger.log("TAPT Complete.")
