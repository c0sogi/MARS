import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from library.utils import set_seed, compute_spearman_metric
from library.loss import RDropLoss
from library.dataset import get_dataloaders
from library.model import DualDistilRoBERTa


class Trainer:
    def __init__(
        self,
        model_name="distilroberta-base",
        batch_size=16,
        max_length=512,
        seed=42,
        device=None,
        working_dir="./working/idea_16",
        submission_dir="./submission",
    ):

        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.working_dir = working_dir
        self.submission_dir = submission_dir

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        set_seed(self.seed)

        # Create directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Load Data
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.target_cols = (
            get_dataloaders(
                batch_size=self.batch_size,
                max_length=self.max_length,
                load_cached_data=True,
                seed=self.seed,
            )
        )
        self.num_labels = len(self.target_cols)

        # Initialize Model
        print(f"Initializing model: {model_name}")
        self.model = DualDistilRoBERTa(
            num_labels=self.num_labels, model_name=model_name
        )
        self.model.to(self.device)

        # Loss Function
        self.criterion = RDropLoss(alpha=1.0)

    def _get_parameter_groups(self, parameters, lr, weight_decay=0.01):
        """
        Separates parameters into groups with and without weight decay.
        """
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in parameters if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        return optimizer_grouped_parameters

    def train(self, epochs=8):
        print("Starting training...")
        best_score = -float("inf")
        best_model_path = os.path.join(self.working_dir, "best_model.pth")

        # ==========================================
        # Phase 1: Epoch 1 (Head Warmup)
        # ==========================================
        print("\nPhase 1: Head Warmup (Epoch 1)")

        # Freeze Backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Identify Head and Fusion parameters
        # We include fusion_norm and head
        head_params = list(self.model.head.named_parameters()) + list(
            self.model.fusion_norm.named_parameters()
        )

        # Initialize Optimizer for Head only
        optimizer = AdamW(
            self._get_parameter_groups(head_params, lr=1e-3, weight_decay=0.01)
        )

        # Train Epoch 1
        self._train_epoch(optimizer, epoch_idx=1, scheduler=None)

        # Validate
        val_score = self.validate()
        print(f"Epoch 1 Validation Score: {val_score:.6f}")

        if val_score > best_score:
            best_score = val_score
            torch.save(self.model.state_dict(), best_model_path)

        # ==========================================
        # Phase 2: Epochs 2-8 (Fine-tuning)
        # ==========================================
        print("\nPhase 2: Fine-tuning (Epochs 2-8)")

        # Unfreeze Backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        # Add Backbone parameters to optimizer
        backbone_params = list(self.model.backbone.named_parameters())
        backbone_groups = self._get_parameter_groups(
            backbone_params, lr=2e-5, weight_decay=0.01
        )

        for group in backbone_groups:
            optimizer.add_param_group(group)

        # Setup Scheduler
        # Total steps for remaining epochs
        remaining_epochs = epochs - 1
        total_steps = len(self.train_loader) * remaining_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Train Loop
        for epoch in range(2, epochs + 1):
            self._train_epoch(optimizer, epoch_idx=epoch, scheduler=scheduler)

            val_score = self.validate()
            print(f"Epoch {epoch} Validation Score: {val_score:.6f}")

            if val_score > best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with score: {best_score:.6f}")

        print(f"\nTraining finished. Best Validation Score: {best_score:.6f}")

        # Generate Submission
        self.generate_submission(best_model_path)

    def _train_epoch(self, optimizer, epoch_idx, scheduler=None):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            # Move to device
            q_input_ids = batch["q_input_ids"].to(self.device)
            q_attention_mask = batch["q_attention_mask"].to(self.device)
            a_input_ids = batch["a_input_ids"].to(self.device)
            a_attention_mask = batch["a_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            # R-Drop: Two forward passes with different dropout masks
            logits1 = self.model(
                q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
            )
            logits2 = self.model(
                q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
            )

            # Compute Loss
            loss = self.criterion(logits1, logits2, labels)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch_idx} Training Loss: {avg_loss:.6f}")

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                q_attention_mask = batch["q_attention_mask"].to(self.device)
                a_input_ids = batch["a_input_ids"].to(self.device)
                a_attention_mask = batch["a_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Single forward pass for validation
                logits = self.model(
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                )
                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        score = compute_spearman_metric(all_labels, all_preds)
        return score

    def generate_submission(self, model_path):
        print("\nGenerating submission...")

        # Load best model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        all_preds = []
        all_qa_ids = []

        with torch.no_grad():
            for batch in self.test_loader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                q_attention_mask = batch["q_attention_mask"].to(self.device)
                a_input_ids = batch["a_input_ids"].to(self.device)
                a_attention_mask = batch["a_attention_mask"].to(self.device)
                qa_ids = batch["qa_id"]

                logits = self.model(
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                )
                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu().numpy())
                all_qa_ids.extend(qa_ids.numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        # Create DataFrame
        sub_df = pd.DataFrame(all_preds, columns=self.target_cols)
        sub_df.insert(0, "qa_id", all_qa_ids)

        # Save
        save_path = os.path.join(self.submission_dir, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")


def main():
    # Entry point for training
    trainer = Trainer()
    trainer.train()


# Note: The main execution block is omitted as per instructions,
# but the class is fully functional.
