import os
import time
import random
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    logging as transformers_logging,
)

from library.config import Config
from library.dataset import MLMDataset
from library.loss import CompositeLoss
from library.awp import AWP
from library.ema import ModelEMA

# Suppress transformers progress bars and warnings
transformers_logging.set_verbosity_error()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_dapt(config):
    """
    Runs Domain-Adaptive Pre-training (Masked Language Modeling) on the corpus.
    Saves the adapted model to config.dapt_model_path.
    """
    print("Starting Domain-Adaptive Pre-training (DAPT)...")
    set_seed(config.seed)

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(config.model_backbone)
    model = AutoModelForMaskedLM.from_pretrained(config.model_backbone)

    # Load Dataset
    dataset = MLMDataset(tokenizer, load_cached_data=True)

    # Data Collator (handles masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.dapt_mlm_probability
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.working_dir, "dapt_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=config.dapt_epochs,
        per_device_train_batch_size=config.dapt_batch_size,
        learning_rate=config.dapt_lr,
        weight_decay=config.weight_decay,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,  # Disable progress bars
        seed=config.seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=config.num_workers,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save the adapted model
    print(f"Saving DAPT model to {config.dapt_model_path}...")
    os.makedirs(config.dapt_model_path, exist_ok=True)
    trainer.save_model(config.dapt_model_path)
    tokenizer.save_pretrained(config.dapt_model_path)
    print("DAPT complete.")


def train_fn(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    config,
    awp=None,
    ema=None,
    loss_fn=None,
):
    """
    Training loop for one epoch.
    """
    model.train()

    running_loss = 0.0
    running_mse = 0.0
    running_ce = 0.0
    running_pearson = 0.0
    dataset_size = 0

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        targets = batch["target"].to(device)

        batch_size = input_ids.size(0)

        # Forward Pass
        reg_logits, cls_logits = model(input_ids, attention_mask, token_type_ids)

        # Calculate Loss
        loss, metrics = loss_fn(reg_logits, cls_logits, targets)

        # Backward Pass
        loss.backward()

        # Adversarial Weight Perturbation (AWP)
        if config.use_awp and awp is not None and epoch >= config.awp_start_epoch:
            # Perturb weights
            awp.attack_step()

            # Forward pass with perturbed weights
            reg_logits_adv, cls_logits_adv = model(
                input_ids, attention_mask, token_type_ids
            )
            loss_adv, _ = loss_fn(reg_logits_adv, cls_logits_adv, targets)

            # Backward pass for adversarial loss
            loss_adv.backward()

            # Restore original weights
            awp.restore()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # EMA Update
        if config.use_ema and ema is not None:
            ema.update(model)

        # Zero Gradients
        optimizer.zero_grad()

        # Logging
        running_loss += metrics["loss_total"].item() * batch_size
        running_mse += metrics["loss_mse"].item() * batch_size
        running_ce += metrics["loss_ce"].item() * batch_size
        running_pearson += metrics["loss_pearson"].item() * batch_size
        dataset_size += batch_size

        if (step + 1) % config.print_freq == 0:
            print(
                f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | "
                f"Loss: {metrics['loss_total'].item():.4f} | "
                f"MSE: {metrics['loss_mse'].item():.4f} | "
                f"PearsonLoss: {metrics['loss_pearson'].item():.4f}"
            )

    epoch_loss = running_loss / dataset_size
    epoch_mse = running_mse / dataset_size
    epoch_ce = running_ce / dataset_size
    epoch_pearson_loss = running_pearson / dataset_size

    elapsed = time.time() - start_time

    print(f"Epoch {epoch+1} Training Complete. Time: {elapsed:.0f}s")
    print(
        f"Train Loss: {epoch_loss:.6f} | MSE: {epoch_mse:.6f} | PearsonLoss: {epoch_pearson_loss:.6f}"
    )

    return epoch_loss


def eval_fn(model, dataloader, device, config, ema=None):
    """
    Evaluation loop. Uses EMA weights if available.
    """
    # Apply EMA weights for evaluation
    if config.use_ema and ema is not None:
        ema.apply_shadow(model)

    model.eval()

    all_preds = []
    all_targets = []

    running_loss = 0.0
    dataset_size = 0

    loss_fn = CompositeLoss(config)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            targets = batch["target"].to(device)
            batch_size = input_ids.size(0)

            # Forward Pass
            reg_logits, cls_logits = model(input_ids, attention_mask, token_type_ids)

            # Compute Loss (for monitoring)
            loss, _ = loss_fn(reg_logits, cls_logits, targets)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets
            # Flatten to 1D array
            preds = reg_logits.view(-1).cpu().numpy()
            targets_np = targets.view(-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets_np)

    # Restore original weights for continued training
    if config.use_ema and ema is not None:
        ema.restore(model)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Clip predictions to [0, 1] for metric calculation (optional, but good practice)
    all_preds_clipped = np.clip(all_preds, 0, 1)

    # Calculate Pearson Correlation
    pearson_score, _ = pearsonr(all_preds_clipped, all_targets)

    avg_loss = running_loss / dataset_size

    print(f"Validation Result - Loss: {avg_loss:.6f} | Pearson: {pearson_score:.6f}")

    return pearson_score


def predict_fn(model, dataloader, device, config):
    """
    Inference loop. Returns raw predictions.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Forward Pass
            reg_logits, _ = model(input_ids, attention_mask, token_type_ids)

            preds = reg_logits.view(-1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    # Post-processing: Clip to [0, 1]
    all_preds = np.clip(all_preds, 0, 1)

    return all_preds
