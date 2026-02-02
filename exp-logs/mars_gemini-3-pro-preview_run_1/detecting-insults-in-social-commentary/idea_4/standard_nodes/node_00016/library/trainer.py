import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, calculate_auc
from library.model import HybridDeberta
from library.awp import AWP


class Trainer:
    """
    Trainer class for the Robust Hybrid DeBERTa model.
    Handles training loop, validation, AWP, and model saving.
    """

    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else Config.device
        self.criterion = nn.BCEWithLogitsLoss()

    def get_optimizer(self, model):
        """
        Sets up the optimizer with differential learning rates.
        Backbone gets a lower LR, Head/Fusion layers get a higher LR.
        """
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer_parameters = [
            {
                "params": backbone_params,
                "lr": Config.lr_backbone,
                "weight_decay": Config.weight_decay,
            },
            {
                "params": head_params,
                "lr": Config.lr_head,
                "weight_decay": Config.weight_decay,
            },
        ]

        return torch.optim.AdamW(optimizer_parameters)

    def train_fn(self, model, optimizer, scheduler, awp, epoch):
        """
        Executes one training epoch with Adversarial Weight Perturbation (AWP).
        """
        model.train()
        losses = AverageMeter()

        for step, batch in enumerate(self.train_loader):
            # Move inputs to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            svd_feat = batch["svd_feat"].to(self.device)
            labels = batch["label"].to(self.device)

            batch_size = labels.size(0)

            # 1. Standard Forward Pass
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, svd_feat)
            loss = self.criterion(preds, labels)

            # 2. Standard Backward Pass
            loss.backward()

            # 3. Adversarial Weight Perturbation (AWP)
            if awp.should_attack(epoch):
                # Save original weights and perturb based on gradients
                awp.attack_step()

                # Clear gradients to compute gradients on the perturbed surface
                optimizer.zero_grad()

                # Forward pass with perturbed weights
                preds_adv = model(input_ids, attention_mask, svd_feat)
                loss_adv = self.criterion(preds_adv, labels)

                # Backward pass to accumulate gradients for update
                loss_adv.backward()

                # Restore original weights (gradients remain)
                awp.restore()

            # 4. Optimizer and Scheduler Step
            optimizer.step()
            scheduler.step()

            losses.update(loss.item(), batch_size)

        return losses.avg

    def valid_fn(self, model):
        """
        Executes validation loop and calculates AUC.
        """
        model.eval()
        losses = AverageMeter()
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                svd_feat = batch["svd_feat"].to(self.device)
                labels = batch["label"].to(self.device)

                batch_size = labels.size(0)

                preds = model(input_ids, attention_mask, svd_feat)
                loss = self.criterion(preds, labels)

                losses.update(loss.item(), batch_size)

                # Apply sigmoid for probabilities
                preds_list.append(torch.sigmoid(preds).cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        predictions = np.concatenate(preds_list)
        ground_truth = np.concatenate(labels_list)
        auc = calculate_auc(ground_truth, predictions)

        return losses.avg, auc

    def train(self):
        """
        Main training loop.
        """
        # Initialize Model
        print(f"Initializing model: {Config.model_name}")
        model = HybridDeberta(pretrained=True)
        model.to(self.device)

        # Optimizer
        optimizer = self.get_optimizer(model)

        # Scheduler
        num_train_steps = len(self.train_loader) * Config.epochs
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # AWP
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

        best_auc = 0.0
        best_model_path = os.path.join(Config.working_dir, "best_model.bin")

        print("Starting training...")
        for epoch in range(Config.epochs):
            start_time = time.time()

            train_loss = self.train_fn(model, optimizer, scheduler, awp, epoch)
            val_loss, val_auc = self.valid_fn(model)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch+1}/{Config.epochs} | Time: {elapsed:.0f}s")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val AUC: {val_auc}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"New Best AUC! Model saved to {best_model_path}")

            print("-" * 30)

        # Clear memory
        del model, optimizer, scheduler, awp
        torch.cuda.empty_cache()
        gc.collect()

        return best_model_path, best_auc

    def predict(self, test_loader, model_path):
        """
        Generates predictions for the test set using the saved model.
        """
        print(f"Loading best model from {model_path} for inference...")
        model = HybridDeberta(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                svd_feat = batch["svd_feat"].to(self.device)

                preds = model(input_ids, attention_mask, svd_feat)
                preds_list.append(torch.sigmoid(preds).cpu().numpy())

        predictions = np.concatenate(preds_list)
        return predictions


def run_training(train_loader, val_loader, test_loader=None):
    """
    Orchestrates the training and submission generation process.
    """
    trainer = Trainer(train_loader, val_loader)

    # Train and get best model
    best_model_path, best_auc = trainer.train()
    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Generate Submission if test_loader is provided
    if test_loader is not None:
        print("Generating predictions for test set...")
        predictions = trainer.predict(test_loader, best_model_path)

        # Load test metadata to preserve structure
        df_test = pd.read_csv(Config.test_path)

        # Ensure lengths match
        if len(predictions) != len(df_test):
            print(
                f"Warning: Prediction length ({len(predictions)}) does not match Test set length ({len(df_test)})"
            )

        # Create submission DataFrame
        # We assume the submission format requires 'Insult', 'Date', 'Comment'
        submission_df = df_test.copy()
        submission_df["Insult"] = predictions

        # Reorder columns to match sample: Insult, Date, Comment
        cols = ["Insult", "Date", "Comment"]
        submission_df = submission_df[cols]

        # Save
        os.makedirs(Config.submission_dir, exist_ok=True)
        submission_df.to_csv(Config.submission_file, index=False)
        print(f"Submission saved to {Config.submission_file}")
