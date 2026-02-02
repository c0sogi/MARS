import os
import gc
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from library.config import Config
from library.utils import get_logger, seed_everything, compute_qwk, AverageMeter
from library.data import EssayDataset, MLMDataset, get_tokenizer
from library.model import CustomModel

logger = get_logger("Trainer")


class AWP:
    """
    Adversarial Weight Perturbation.
    Perturbs the weights of the model to maximize the loss, improving robustness.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, inputs, labels, criterion, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with autocast(enabled=True):
                # Forward pass with perturbed weights
                # Unpack inputs based on dictionary structure
                input_ids = inputs["input_ids"].to(Config.device)
                attention_mask = inputs["attention_mask"].to(Config.device)
                outputs = self.model(input_ids, attention_mask)
                adv_loss = criterion(outputs.view(-1), labels.view(-1))

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Layer-wise Learning Rate Decay (LLRD) parameter grouping.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # DeBERTa-v3-large specific layer naming
    model_type = "backbone"

    # Initialize layers
    # DeBERTa V3 Large has 24 layers usually
    num_layers = 24
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers

    # Group parameters
    # 1. Embeddings
    # 2. Encoder Layers (0 to N-1)
    # 3. Task specific head (decoder)

    # Calculate LRs for each layer
    layer_lrs = []
    current_lr = encoder_lr
    for i in range(num_layers + 1):  # +1 for embeddings
        layer_lrs.append(current_lr)
        current_lr *= Config.llrd_decay
    layer_lrs.reverse()  # Lower layers get lower LR

    # Embeddings are at index 0 in reversed list (lowest LR)
    # Top layer is at index -1 (highest encoder LR)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lr = encoder_lr

        if "embeddings" in name:
            lr = layer_lrs[0]
        elif "encoder.layer" in name:
            # Find layer index
            try:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                lr = layer_lrs[layer_idx + 1]
            except:
                lr = encoder_lr
        elif "backbone" in name and "encoder" not in name and "embeddings" not in name:
            # Other backbone params (like pooler if exists, or final layernorm)
            lr = encoder_lr
        else:
            # Regression Head / Attention Pooling
            lr = decoder_lr

        if any(nd in name for nd in no_decay):
            optimizer_parameters.append({"params": [p], "weight_decay": 0.0, "lr": lr})
        else:
            optimizer_parameters.append(
                {"params": [p], "weight_decay": weight_decay, "lr": lr}
            )

    return optimizer_parameters


def train_mlm(train_df, test_df):
    """
    Performs Domain-Adaptive Pre-training (MLM) on combined train and test data.
    """
    logger.info("Starting MLM Pre-training...")

    # Check if already trained
    if os.path.exists(os.path.join(Config.mlm_model_dir, "config.json")):
        logger.info("MLM model already exists. Skipping training.")
        return Config.mlm_model_dir

    # Combine text
    all_texts = (
        pd.concat([train_df["full_text"], test_df["full_text"]])
        .reset_index(drop=True)
        .values.tolist()
    )

    tokenizer = get_tokenizer()
    dataset = MLMDataset(
        all_texts,
        tokenizer,
        max_length=Config.max_length,
        mlm_probability=Config.mlm_probability,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=Config.mlm_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.to(Config.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.mlm_learning_rate)
    scaler = GradScaler()

    for epoch in range(Config.mlm_epochs):
        losses = AverageMeter()
        progress_bar = tqdm(
            dataloader, desc=f"MLM Epoch {epoch+1}/{Config.mlm_epochs}", leave=False
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), input_ids.size(0))
            progress_bar.set_postfix(loss=losses.avg)

        logger.info(f"MLM Epoch {epoch+1} Loss: {losses.avg}")

    # Save the backbone
    logger.info(f"Saving MLM model to {Config.mlm_model_dir}")
    model.save_pretrained(Config.mlm_model_dir)
    tokenizer.save_pretrained(Config.mlm_model_dir)

    del model, optimizer, dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return Config.mlm_model_dir


def train_fn(
    model, train_loader, optimizer, scheduler, criterion, epoch, awp=None, scaler=None
):
    model.train()
    losses = AverageMeter()

    # progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(Config.device)
        attention_mask = batch["attention_mask"].to(Config.device)
        labels = batch["labels"].to(Config.device)

        batch_size = input_ids.size(0)

        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), labels.view(-1))

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            if Config.use_awp and awp is not None:
                # AWP attack
                awp.attack_backward(batch, labels, criterion, epoch)

            # Cite debug_lesson_4: Prevent GradScaler Double Unscaling
            try:
                scaler.unscale_(optimizer)
            except RuntimeError:
                pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)
        # progress_bar.set_postfix(loss=losses.avg)

    return losses.avg


def valid_fn(model, valid_loader, criterion):
    model.eval()
    losses = AverageMeter()
    preds = []
    labels_list = []

    # progress_bar = tqdm(valid_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1), labels.view(-1))

            losses.update(loss.item(), input_ids.size(0))

            preds.append(outputs.view(-1).cpu().numpy())
            labels_list.append(labels.view(-1).cpu().numpy())
            # progress_bar.set_postfix(loss=losses.avg)

    predictions = np.concatenate(preds)
    true_labels = np.concatenate(labels_list)

    return losses.avg, predictions, true_labels


def inference_fn(model, loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)

            with autocast():
                outputs = model(input_ids, attention_mask)

            preds.append(outputs.view(-1).cpu().numpy())

    return np.concatenate(preds)


def train_fold(fold, train_df, test_df, mlm_model_path=None):
    logger.info(f"=== Training Fold {fold} ===")

    # Prepare Data
    train_fold_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val_fold_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Debug mode
    if Config.debug:
        train_fold_df = train_fold_df.sample(
            n=100, random_state=Config.seed
        ).reset_index(drop=True)
        val_fold_df = val_fold_df.sample(n=50, random_state=Config.seed).reset_index(
            drop=True
        )
        test_df = test_df.sample(n=50, random_state=Config.seed).reset_index(drop=True)

    tokenizer = get_tokenizer()

    train_dataset = EssayDataset(train_fold_df, tokenizer, Config.max_length)
    val_dataset = EssayDataset(val_fold_df, tokenizer, Config.max_length)
    test_dataset = EssayDataset(test_df, tokenizer, Config.max_length, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    # If MLM was run, mlm_model_path contains the pretrained weights.
    # CustomModel initializes backbone from model_name.
    model_source = mlm_model_path if mlm_model_path else Config.model_name
    model = CustomModel(model_name=model_source, pretrained=True)
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer_grouped_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate * 5,  # Higher LR for head
        weight_decay=Config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_train_steps = int(
        len(train_fold_df)
        / Config.train_batch_size
        / Config.gradient_accumulation_steps
        * Config.epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    # AWP
    awp = None
    if Config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
            scaler=scaler,
        )

    best_score = -np.inf
    best_loss = np.inf

    # Training Loop
    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, epoch, awp, scaler
        )
        val_loss, val_preds, val_labels = valid_fn(model, val_loader, criterion)

        # Calculate QWK
        val_score = compute_qwk(val_labels, val_preds)

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val QWK: {val_score} - Time: {elapsed:.0f}s"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            logger.info(f"Epoch {epoch+1} - Best Score Updated: {best_score}")
            torch.save(
                model.state_dict(),
                os.path.join(Config.model_dir, f"model_fold_{fold}.pth"),
            )

    # Load Best Model for Inference
    model.load_state_dict(
        torch.load(os.path.join(Config.model_dir, f"model_fold_{fold}.pth"))
    )

    # Generate OOF and Test Predictions
    oof_preds = inference_fn(model, val_loader)
    test_preds = inference_fn(model, test_loader)

    # Cleanup
    del (
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return val_fold_df["essay_id"].values, oof_preds, test_preds, best_score


def run_training(train_df, test_df):
    """
    Orchestrates the full training pipeline.
    """
    seed_everything(Config.seed)

    # 1. MLM Pre-training
    mlm_path = train_mlm(train_df, test_df)

    # 2. Supervised Training (Cross-Validation)
    oof_df = pd.DataFrame()
    test_preds_accum = np.zeros(len(test_df))
    scores = []

    for fold in range(Config.num_folds):
        ids, oof_preds, test_preds, score = train_fold(
            fold, train_df, test_df, mlm_model_path=mlm_path
        )

        # Store OOF
        fold_oof_df = pd.DataFrame(
            {"essay_id": ids, "pred_score": oof_preds, "fold": fold}
        )
        oof_df = pd.concat([oof_df, fold_oof_df], axis=0)

        # Accumulate Test Preds
        test_preds_accum += test_preds
        scores.append(score)

    # Average Test Predictions
    avg_test_preds = test_preds_accum / Config.num_folds

    logger.info(f"CV Scores: {scores}")
    logger.info(f"Mean CV Score: {np.mean(scores)}")

    # Save OOF for Level 2 Stacking
    oof_df = oof_df.sort_values(by="essay_id").reset_index(drop=True)
    oof_df.to_csv(os.path.join(Config.working_dir, "oof_predictions.csv"), index=False)

    # Create Submission DataFrame (Level 1 Output - though Level 2 is preferred if implemented later)
    # The task asks to submit predictions. We will submit the averaged Level 1 predictions here.
    # Note: If a Level 2 model is implemented in a separate script, it would consume oof_predictions.csv
    # For this file's scope, we generate the submission based on the ensemble of Level 1 models.

    submission = pd.DataFrame(
        {"essay_id": test_df["essay_id"], "score": avg_test_preds}
    )

    # Apply rounding and clipping for final submission format
    submission["score"] = np.round(submission["score"]).clip(1, 6).astype(int)

    submission.to_csv(Config.submission_path, index=False)
    logger.info(f"Submission saved to {Config.submission_path}")

    return oof_df, submission
