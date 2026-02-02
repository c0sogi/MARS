import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_log_loss


class DistillationDataset(Dataset):
    """
    Unified dataset for Distillation that handles both:
    1. Labeled data (Train): Has hard labels, no soft labels.
    2. Unlabeled data (Test): Has soft labels (probabilities), no hard labels.
    """

    def __init__(
        self,
        texts,
        tokenizer,
        hard_labels=None,
        soft_labels=None,
        max_length=Config.MAX_LENGTH,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
        self.max_length = max_length

        # Validation
        if self.hard_labels is not None:
            assert len(self.texts) == len(self.hard_labels)
        if self.soft_labels is not None:
            assert len(self.texts) == len(self.soft_labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # Determine type of sample
        if self.hard_labels is not None and self.hard_labels[idx] != -1:
            item["labels"] = torch.tensor(self.hard_labels[idx], dtype=torch.long)
            item["is_labeled"] = torch.tensor(1, dtype=torch.long)
            # Placeholder for soft labels to keep batch structure consistent
            item["soft_labels"] = torch.zeros(Config.NUM_CLASSES, dtype=torch.float)
        else:
            item["labels"] = torch.tensor(-1, dtype=torch.long)  # Ignore index
            item["is_labeled"] = torch.tensor(0, dtype=torch.long)
            if self.soft_labels is not None:
                item["soft_labels"] = torch.tensor(
                    self.soft_labels[idx], dtype=torch.float
                )
            else:
                item["soft_labels"] = torch.zeros(Config.NUM_CLASSES, dtype=torch.float)

        return item


class DistillationEngine:
    """
    Handles training (Supervised & Distillation) and evaluation of neural models.
    """

    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.best_val_loss = float("inf")

    def _get_optimizer_scheduler(self, num_train_steps):
        optimizer = AdamW(
            self.model.parameters(),
            lr=Config.FT_LR,
            weight_decay=Config.FT_WEIGHT_DECAY,
        )

        num_warmup_steps = int(num_train_steps * Config.FT_WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        return optimizer, scheduler

    def train_supervised(
        self, train_loader, val_loader, epochs=Config.FT_EPOCHS, fold_idx=0
    ):
        """
        Phase 1: Standard Supervised Training on Labeled Data.
        """
        print(f"\n[Fold {fold_idx}] Starting Supervised Training...")

        optimizer, scheduler = self._get_optimizer_scheduler(len(train_loader) * epochs)
        loss_fct = nn.CrossEntropyLoss()

        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss_meter = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]

                loss = loss_fct(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )
                optimizer.step()
                scheduler.step()

                train_loss_meter += loss.item()

            avg_train_loss = train_loss_meter / len(train_loader)

            # Validation
            val_loss, _ = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val Loss: {val_loss}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.best_val_loss

    def train_distilled(
        self,
        train_texts,
        train_labels,
        test_texts,
        test_soft_targets,
        val_loader,
        epochs=Config.FT_EPOCHS,
    ):
        """
        Phase 3: Distillation Training on Combined (Train + Test) Data.
        Uses Soft Targets for Test data and Hard Labels for Train data.
        """
        print(f"\nStarting Distillation Training...")

        # Reset best val loss for this phase
        self.best_val_loss = float("inf")

        # Prepare Combined Dataset
        # 1. Train Data (Hard Labels)
        train_len = len(train_texts)
        train_hard = train_labels
        train_soft = np.zeros((train_len, Config.NUM_CLASSES))  # Placeholder

        # 2. Test Data (Soft Targets)
        test_len = len(test_texts)
        test_hard = np.full(test_len, -1)  # Placeholder
        test_soft = test_soft_targets

        # Combine
        combined_texts = train_texts + test_texts
        combined_hard = np.concatenate([train_hard, test_hard])
        combined_soft = np.concatenate([train_soft, test_soft])

        dataset = DistillationDataset(
            combined_texts,
            self.tokenizer,
            hard_labels=combined_hard,
            soft_labels=combined_soft,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        optimizer, scheduler = self._get_optimizer_scheduler(len(dataloader) * epochs)

        # Loss functions
        ce_loss_fct = nn.CrossEntropyLoss()
        kl_loss_fct = nn.KLDivLoss(reduction="batchmean")

        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss_meter = 0.0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)  # Hard labels
                soft_labels = batch["soft_labels"].to(self.device)  # Soft targets
                is_labeled = batch["is_labeled"].to(
                    self.device
                )  # Mask: 1 for Train, 0 for Test

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]

                loss = 0.0

                # 1. Supervised Loss (Cross Entropy) for Labeled Data
                labeled_mask = is_labeled == 1
                if labeled_mask.sum() > 0:
                    loss_ce = ce_loss_fct(logits[labeled_mask], labels[labeled_mask])
                    loss += (1 - Config.DISTILLATION_ALPHA) * loss_ce

                # 2. Distillation Loss (KL Divergence) for Unlabeled Data
                unlabeled_mask = is_labeled == 0
                if unlabeled_mask.sum() > 0:
                    # T = Temperature
                    T = Config.DISTILLATION_TEMP

                    # Student Logits -> LogSoftmax with Temp
                    student_log_probs = F.log_softmax(logits[unlabeled_mask] / T, dim=1)

                    # Teacher Targets -> Probabilities (Already Softmaxed usually)
                    # We assume soft_labels are probabilities.
                    teacher_probs = soft_labels[unlabeled_mask]

                    loss_kl = kl_loss_fct(student_log_probs, teacher_probs)

                    # Scale by T^2 as per Hinton et al.
                    loss += Config.DISTILLATION_ALPHA * (T**2) * loss_kl

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )
                optimizer.step()
                scheduler.step()

                train_loss_meter += loss.item()

            avg_train_loss = train_loss_meter / len(dataloader)

            # Validation (Always on Ground Truth Val Set)
            val_loss, _ = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Distill Loss: {avg_train_loss} | Val Loss: {val_loss}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.best_val_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on a dataloader.
        Returns average loss and probabilities.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)

                all_preds.append(probs.cpu().numpy())

                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        loss = 0.0
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)
            # Only compute loss if we have valid labels (not placeholder -1)
            # This handles cases where we might predict on test set
            valid_mask = all_labels != -1
            if valid_mask.any():
                # compute_log_loss expects raw probabilities, it handles clipping/norm
                loss = compute_log_loss(all_labels[valid_mask], all_preds[valid_mask])

        return loss, all_preds

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
