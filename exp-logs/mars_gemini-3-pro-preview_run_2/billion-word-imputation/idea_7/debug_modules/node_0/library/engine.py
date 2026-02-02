import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import save_model


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance, specifically for the Locator model.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean", ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, seq_len, num_classes] or [N, num_classes]
            targets: [batch_size, seq_len] or [N]
        """
        # Flatten logits and targets
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Filter out ignore_index
        active_mask = targets != self.ignore_index
        logits = logits[active_mask]
        targets = targets[active_mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute Cross Entropy
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        # Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """
    Encapsulates training logic for the three stages of the pipeline.
    """

    @staticmethod
    def get_optimizer_and_scheduler(
        model, learning_rate, num_training_steps, warmup_ratio=0.1
    ):
        """
        Sets up AdamW optimizer and a linear scheduler with warmup.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

        num_warmup_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, scheduler

    @staticmethod
    def train_locator(
        model,
        train_loader,
        val_loader,
        epochs=Config.LOCATOR_EPOCHS,
        device=Config.DEVICE,
    ):
        """
        Training loop for Stage 1: Locator (Token Classification).
        Uses Focal Loss to handle the imbalance between 'Gap' (rare) and 'No Gap' tokens.
        """
        print(f"\n[Locator] Starting training for {epochs} epochs on {device}...")

        model.to(device)
        criterion = FocalLoss(alpha=1.0, gamma=2.0, ignore_index=-100)

        num_training_steps = len(train_loader) * epochs
        optimizer, scheduler = Trainer.get_optimizer_and_scheduler(
            model, Config.LOCATOR_LR, num_training_steps
        )

        best_val_f1 = -1.0
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    # Predictions
                    preds = torch.argmax(logits, dim=-1)

                    # Flatten and filter for metrics
                    active_mask = labels != -100
                    active_preds = preds[active_mask]
                    active_labels = labels[active_mask]

                    all_preds.extend(active_preds.cpu().numpy())
                    all_labels.extend(active_labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
            val_precision = precision_score(
                all_labels, all_preds, average="binary", pos_label=1, zero_division=0
            )
            val_recall = recall_score(
                all_labels, all_preds, average="binary", pos_label=1, zero_division=0
            )

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Val F1 (Gap): {val_f1:.6f} | "
                f"Val Precision: {val_precision:.6f} | "
                f"Val Recall: {val_recall:.6f}"
            )

            # Checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_model(model, Config.LOCATOR_CKPT_PATH)
                print(f"  -> New best model saved (F1: {best_val_f1:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

        return model

    @staticmethod
    def train_infiller(
        model,
        train_loader,
        val_loader,
        epochs=Config.INFILLER_EPOCHS,
        device=Config.DEVICE,
    ):
        """
        Training loop for Stage 2: In-Filler (Masked Language Modeling).
        """
        print(f"\n[Infiller] Starting training for {epochs} epochs on {device}...")

        model.to(device)
        # Standard Cross Entropy for MLM (ignore_index=-100 is default)
        criterion = nn.CrossEntropyLoss()

        num_training_steps = len(train_loader) * epochs
        optimizer, scheduler = Trainer.get_optimizer_and_scheduler(
            model, Config.INFILLER_LR, num_training_steps
        )

        best_val_acc = -1.0
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

                # Reshape for loss: [batch*seq_len, vocab_size] vs [batch*seq_len]
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits

                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    val_loss += loss.item()

                    # Calculate Accuracy on masked tokens only
                    preds = torch.argmax(logits, dim=-1)
                    mask = labels != -100

                    if mask.sum() > 0:
                        correct += (preds[mask] == labels[mask]).sum().item()
                        total += mask.sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total if total > 0 else 0.0

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Val Accuracy: {val_acc:.6f}"
            )

            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, Config.INFILLER_CKPT_PATH)
                print(f"  -> New best model saved (Acc: {best_val_acc:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

        return model

    @staticmethod
    def train_verifier(
        model,
        train_loader,
        val_loader,
        epochs=Config.VERIFIER_EPOCHS,
        device=Config.DEVICE,
    ):
        """
        Training loop for Stage 3: Verifier (Sequence Classification).
        Uses Binary Cross Entropy with Label Smoothing.
        """
        print(f"\n[Verifier] Starting training for {epochs} epochs on {device}...")

        model.to(device)
        # Label smoothing helps prevent overconfidence in the discriminator
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.VERIFIER_LABEL_SMOOTHING)

        num_training_steps = len(train_loader) * epochs
        optimizer, scheduler = Trainer.get_optimizer_and_scheduler(
            model, Config.VERIFIER_LR, num_training_steps
        )

        best_val_acc = -1.0
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits

                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, zero_division=0)
            val_recall = recall_score(all_labels, all_preds, zero_division=0)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Val Acc: {val_acc:.6f} | "
                f"Val Precision: {val_precision:.6f} | "
                f"Val Recall: {val_recall:.6f}"
            )

            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, Config.VERIFIER_CKPT_PATH)
                print(f"  -> New best model saved (Acc: {best_val_acc:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

        return model
