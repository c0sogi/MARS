import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, format_time, print_metric
from library.metrics import compute_final_metric


class Trainer:
    """
    Trainer class to manage the training, validation, and inference of the ToxicityClassifier.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.best_score = -float("inf")

    def compute_loss(self, outputs, targets, weights=None):
        """
        Computes Weighted Binary Cross Entropy Loss.
        """
        # Flatten outputs and targets
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        criterion = nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(outputs, targets)

        if weights is not None:
            weights = weights.view(-1)
            loss = loss * weights

        return loss.mean()

    def configure_optimizers(self, num_training_steps):
        """
        Sets up AdamW optimizer and Linear Scheduler.
        """
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

        optimizer = AdamW(optimizer_parameters, lr=Config.LEARNING_RATE)

        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def train_epoch(self, dataloader, optimizer, scheduler):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)
            weights = batch["weight"].to(self.device) if "weight" in batch else None

            optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask)
            loss = self.compute_loss(outputs, targets, weights)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            if scheduler:
                scheduler.step()

            losses.update(loss.item(), input_ids.size(0))

        return losses.avg

    def evaluate(self, dataloader, val_df):
        """
        Runs validation and computes the competition metric.
        """
        self.model.eval()
        losses = AverageMeter()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)
                weights = batch["weight"].to(self.device) if "weight" in batch else None

                outputs = self.model(input_ids, attention_mask)
                loss = self.compute_loss(outputs, targets, weights)

                losses.update(loss.item(), input_ids.size(0))

                # Apply sigmoid to get probabilities
                batch_preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend(batch_preds)

        # Assign predictions to the validation dataframe to calculate metrics
        # We create a copy to avoid modifying the original dataframe passed in
        val_df_eval = val_df.copy()

        # Ensure lengths match (validation loader should not be shuffled for this to work perfectly)
        if len(preds) != len(val_df_eval):
            val_df_eval = val_df_eval.iloc[: len(preds)]

        val_df_eval["prediction"] = preds

        score = compute_final_metric(
            val_df_eval, "prediction", Config.TARGET_COL, verbose=True
        )

        return losses.avg, score

    def fit(self, train_loader, val_loader, val_df):
        """
        Main training loop with freezing strategy and early stopping.
        """
        num_training_steps = int(len(train_loader) * Config.EPOCHS)
        optimizer, scheduler = self.configure_optimizers(num_training_steps)

        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            # -----------------------------------------------------------
            # Freezing Strategy:
            # Epoch 0: Freeze transformer backbone, train only head.
            # Epoch 1+: Unfreeze backbone, train all.
            # -----------------------------------------------------------
            if epoch == 0:
                print("Epoch 1: Freezing transformer backbone.")
                for param in self.model.base_model.parameters():
                    param.requires_grad = False
            else:
                print(f"Epoch {epoch+1}: Unfreezing transformer backbone.")
                for param in self.model.base_model.parameters():
                    param.requires_grad = True

            # Timing
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_loss, val_score = self.evaluate(val_loader, val_df)

            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 1000

            print(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {format_time(elapsed)}")
            print_metric("Train Loss", train_loss)
            print_metric("Val Loss", val_loss)

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                print(
                    f"Score Improved ({self.best_score:.5f} -> {val_score:.5f}). Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Score did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    def predict(self, dataloader):
        """
        Generates predictions for a dataloader.
        """
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                batch_preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend(batch_preds)

        return preds

    def generate_submission(self, test_loader):
        """
        Loads the best model, predicts on test set, and saves submission.
        """
        print("\nLoading best model for inference...")
        self.model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

        print("Generating predictions on Test set...")
        preds = self.predict(test_loader)

        # Prepare Submission DataFrame
        if Config.DEBUG:
            # In debug mode, we used a subset of the test file
            test_df = pd.read_csv(Config.TEST_PATH)
            test_df = test_df.sample(
                n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)
            submission = pd.DataFrame({"id": test_df["id"], "prediction": preds})
        else:
            submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
            submission["prediction"] = preds

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
