import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
import pandas as pd
import os
import time
import gc

from library.config import Config
from library.utils import Logger, AverageMeter, time_since, get_device
from library.metrics import compute_bias_metrics
from library.model import ToxicityModel


class Trainer:
    """
    Manages the training, validation, and optimization of the ToxicityModel.
    Implements Device-Side Trimming, Mixed Precision, and SWA.
    """

    def __init__(self, train_loader, val_loader):
        self.device = get_device()
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Model
        self.model = ToxicityModel()
        self.model.to(self.device)

        # Optimizer Configuration
        # Separate weight decay for bias/LayerNorm to improve training stability
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = optim.AdamW(
            optimizer_parameters,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (OneCycleLR)
        # Calculate total training steps
        self.num_train_steps = int(
            len(self.train_loader) / Config.ACCUMULATE_GRAD_STEPS * Config.EPOCHS
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            total_steps=self.num_train_steps,
            pct_start=Config.WARMUP_RATIO,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        # Loss Functions
        # Primary: BCE with sample weights (reduction='none' to apply weights manually)
        self.criterion_tox = nn.BCEWithLogitsLoss(reduction="none")
        # Auxiliary: BCE for multi-label identities
        self.criterion_aux = nn.BCEWithLogitsLoss()

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # SWA Setup
        self.use_swa = Config.USE_SWA
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=Config.SWA_LR)
            self.swa_start_epoch = Config.SWA_START_EPOCH

        # Logging and Tracking
        self.logger = Logger()
        self.best_score = -float("inf")

    def _trim_tensors(self, input_ids, attention_mask):
        """
        Device-side trimming: Slices the batch to the maximum sequence length
        present in the current batch to save compute.
        """
        # Find the maximum length in this batch (where mask is 1)
        # attention_mask shape: (Batch, MaxLen)
        max_len = attention_mask.sum(dim=1).max().item()

        # Slice inputs to this length
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        return input_ids, attention_mask

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        start_time = time.time()

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            aux_targets = batch["aux_target"].to(self.device, non_blocking=True)
            sample_weights = batch["sample_weight"].to(self.device, non_blocking=True)

            # Apply Device-side trimming
            input_ids, attention_mask = self._trim_tensors(input_ids, attention_mask)

            batch_size = input_ids.size(0)

            # Forward Pass with Mixed Precision
            with autocast():
                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                aux_logits = outputs["aux_logits"]

                # Calculate Losses
                # 1. Toxicity Loss (Weighted by sample_weights)
                loss_tox_per_sample = self.criterion_tox(logits, targets)
                loss_tox = (loss_tox_per_sample * sample_weights).mean()

                # 2. Auxiliary Identity Loss
                loss_aux = self.criterion_aux(aux_logits, aux_targets)

                # Combined Loss
                loss = loss_tox + (Config.AUX_LOSS_WEIGHT * loss_aux)

                # Scale loss for gradient accumulation
                if Config.ACCUMULATE_GRAD_STEPS > 1:
                    loss = loss / Config.ACCUMULATE_GRAD_STEPS

            # Backward Pass
            self.scaler.scale(loss).backward()

            if (step + 1) % Config.ACCUMULATE_GRAD_STEPS == 0:
                # Clip Gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

                # Optimizer Step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Scheduler Step
                # If SWA is active and we are in the SWA phase, we pause the main scheduler
                # and let the SWA scheduler handle LR at the end of the epoch.
                if not self.use_swa or epoch < self.swa_start_epoch:
                    self.scheduler.step()

            losses.update(loss.item() * Config.ACCUMULATE_GRAD_STEPS, batch_size)

            if step % 100 == 0 or step == len(self.train_loader) - 1:
                self.logger.log(
                    f"Epoch {epoch+1}/{Config.EPOCHS} "
                    f"Step {step}/{len(self.train_loader)} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Time: {time_since(start_time, (step+1)/len(self.train_loader))}"
                )

    def validate(self, epoch, model_to_eval=None):
        if model_to_eval is None:
            model_to_eval = self.model

        model_to_eval.eval()
        all_preds = []
        all_targets = []
        all_ids = []
        all_identities = []

        start_time = time.time()

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                targets = batch["target"].to(self.device, non_blocking=True)
                aux_targets = batch["aux_target"].cpu().numpy()
                ids = batch["id"].cpu().numpy()

                # Device-side trimming
                input_ids, attention_mask = self._trim_tensors(
                    input_ids, attention_mask
                )

                with autocast():
                    outputs = model_to_eval(input_ids, attention_mask)
                    logits = outputs["logits"]

                preds = torch.sigmoid(logits).float().cpu().numpy()
                targets = targets.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)
                all_ids.append(ids)
                all_identities.append(aux_targets)

        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_ids = np.concatenate(all_ids)
        all_identities = np.concatenate(all_identities)

        # Construct DataFrame for metric calculation
        val_df = pd.DataFrame(
            {
                Config.ID_COL: all_ids,
                Config.TARGET_COL: all_targets,
                "prediction": all_preds,
            }
        )

        # Add identity columns back for bias metrics
        for i, col in enumerate(Config.IDENTITY_COLUMNS):
            val_df[col] = all_identities[:, i]

        # Calculate Metrics
        metrics = compute_bias_metrics(
            val_df, Config.TARGET_COL, "prediction", Config.IDENTITY_COLUMNS
        )

        elapsed = time.time() - start_time
        self.logger.log(f"Epoch {epoch+1} Validation Complete in {elapsed:.0f}s")
        self.logger.log_metrics(metrics, prefix=f"Epoch {epoch+1}")

        # Cleanup
        del all_preds, all_targets, all_ids, all_identities, val_df
        gc.collect()

        return metrics["score"]

    def train(self):
        self.logger.log(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            # 1. Train Loop
            self.train_one_epoch(epoch)

            # 2. SWA Update
            if self.use_swa and (epoch + 1) >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                self.logger.log(
                    f"SWA: Updated parameters and stepped scheduler at epoch {epoch+1}"
                )

            # 3. Validate (Standard Model)
            score = self.validate(epoch, self.model)

            # 4. Save Best Standard Model
            if score > self.best_score:
                self.best_score = score
                self.logger.log(
                    f"New Best Score: {self.best_score:.6f}. Saving model..."
                )
                torch.save(self.model.state_dict(), Config.OUTPUT_MODEL_PATH)

        # End of Training
        self.logger.log("Training complete.")

        # 5. Handle SWA Finalization
        if self.use_swa:
            self.logger.log("Finalizing SWA Model...")
            # Note: RoBERTa uses LayerNorm, so update_bn is not strictly necessary
            # and can be skipped to avoid complexity with dict-based loaders.

            # Save SWA Model
            torch.save(self.swa_model.module.state_dict(), Config.OUTPUT_SWA_MODEL_PATH)

            # Validate SWA Model
            self.logger.log("Validating SWA Model...")
            swa_score = self.validate(Config.EPOCHS, self.swa_model.module)
            self.logger.log(f"SWA Model Score: {swa_score:.6f}")
