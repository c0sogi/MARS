import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_loaders, ALL_TARGETS
from library.model import DebertaDualHead


class Engine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Loss function for multi-label binary classification
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, data_loader, epoch_index):
        self.model.train()

        # Freeze backbone for the first epoch (epoch 0)
        # Unfreeze for subsequent epochs
        if epoch_index == 0:
            self.model.backbone.requires_grad_(False)
        else:
            self.model.backbone.requires_grad_(True)

        total_loss = 0
        counter = 0

        for batch in data_loader:
            counter += 1

            # Move inputs to device
            q_ids = batch["q_input_ids"].to(self.device)
            q_mask = batch["q_attention_mask"].to(self.device)
            a_ids = batch["a_input_ids"].to(self.device)
            a_mask = batch["a_attention_mask"].to(self.device)
            q_labels = batch["q_labels"].to(self.device)
            a_labels = batch["a_labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            q_logits, a_logits = self.model(q_ids, q_mask, a_ids, a_mask)

            # Compute loss for both heads separately and sum
            loss_q = self.criterion(q_logits, q_labels)
            loss_a = self.criterion(a_logits, a_labels)
            loss = loss_q + loss_a

            # Backward pass
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / counter

    def validate(self, data_loader):
        self.model.eval()
        final_targets = []
        final_preds = []

        with torch.no_grad():
            for batch in data_loader:
                q_ids = batch["q_input_ids"].to(self.device)
                q_mask = batch["q_attention_mask"].to(self.device)
                a_ids = batch["a_input_ids"].to(self.device)
                a_mask = batch["a_attention_mask"].to(self.device)

                # Get ground truth (on CPU for metric calculation)
                q_labels = batch["q_labels"].numpy()
                a_labels = batch["a_labels"].numpy()

                # Forward pass
                q_logits, a_logits = self.model(q_ids, q_mask, a_ids, a_mask)

                # Apply sigmoid to get probabilities
                q_preds = torch.sigmoid(q_logits).cpu().numpy()
                a_preds = torch.sigmoid(a_logits).cpu().numpy()

                # Concatenate predictions and targets to match ALL_TARGETS order
                batch_preds = np.concatenate([q_preds, a_preds], axis=1)
                batch_targets = np.concatenate([q_labels, a_labels], axis=1)

                final_preds.append(batch_preds)
                final_targets.append(batch_targets)

        final_preds = np.vstack(final_preds)
        final_targets = np.vstack(final_targets)

        # Compute metric
        score = compute_spearman_metric(final_preds, final_targets)
        return score

    def predict(self, data_loader):
        self.model.eval()
        final_qa_ids = []
        final_preds = []

        with torch.no_grad():
            for batch in data_loader:
                q_ids = batch["q_input_ids"].to(self.device)
                q_mask = batch["q_attention_mask"].to(self.device)
                a_ids = batch["a_input_ids"].to(self.device)
                a_mask = batch["a_attention_mask"].to(self.device)
                qa_ids = batch["qa_id"]

                # Forward pass
                q_logits, a_logits = self.model(q_ids, q_mask, a_ids, a_mask)

                # Apply sigmoid
                q_preds = torch.sigmoid(q_logits).cpu().numpy()
                a_preds = torch.sigmoid(a_logits).cpu().numpy()

                # Concatenate
                batch_preds = np.concatenate([q_preds, a_preds], axis=1)

                final_preds.append(batch_preds)
                final_qa_ids.extend(qa_ids.numpy())

        final_preds = np.vstack(final_preds)
        return final_qa_ids, final_preds


def run_task(epochs=5, batch_size=8, lr=2e-5, load_cached_data=True):
    """
    Main function to run the training and inference pipeline.
    """
    seed_everything(42)

    # Directories
    os.makedirs("./working/idea_3", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)
    checkpoint_path = "./working/idea_3/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Model Initialization
    model = DebertaDualHead()
    model.to(device)

    # Optimizer Setup (Weight Decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=lr)

    # Scheduler Setup
    num_train_steps = int(len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    engine = Engine(model, device, optimizer, scheduler)
    best_score = -1.0

    # Training Loop
    for epoch in range(epochs):
        train_loss = engine.train_one_epoch(train_loader, epoch)
        val_score = engine.validate(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Spearman: {val_score}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), checkpoint_path)

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Re-initialize engine for prediction (optimizer/scheduler not needed)
    inference_engine = Engine(model, device)
    qa_ids, preds = inference_engine.predict(test_loader)

    # Create Submission File
    sub_df = pd.DataFrame(preds, columns=ALL_TARGETS)
    sub_df.insert(0, "qa_id", qa_ids)

    submission_path = "./submission/submission.csv"
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
