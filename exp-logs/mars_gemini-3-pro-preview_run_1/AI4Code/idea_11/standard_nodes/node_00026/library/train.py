import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import get_device, compute_kendall_tau, get_ranks
from library.dataset import CachedNotebookDataset
from library.model import DualContextAnchorNetwork


class Trainer:
    """
    Manages the training, validation, and checkpointing process for the DC-AN model.
    """

    def __init__(self):
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = DualContextAnchorNetwork().to(self.device)

        # Optimizer: AdamW with constant learning rate (no warmup) as per design
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function: CrossEntropyLoss
        # We use ignore_index=-100 to handle padded labels in the batch
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Training State
        self.best_score = -float("inf")
        self.patience_counter = 0

    def train_epoch(self, dataloader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # Move data to device
            code_emb = batch["code_embeddings"].to(self.device)
            code_mask = batch["code_mask"].to(self.device)
            code_lens = batch["code_lens"].to(self.device)

            md_emb = batch["md_embeddings"].to(self.device)
            md_mask = batch["md_mask"].to(self.device)
            md_lens = batch["md_lens"].to(self.device)

            labels = batch["labels"].to(self.device)

            # Forward Pass
            # Logits shape: (Batch, MD_Len, Code_Len + 1)
            logits = self.model(
                code_emb, code_mask, code_lens, md_emb, md_mask, md_lens
            )

            # Flatten inputs for CrossEntropyLoss
            # Logits: (Batch * MD_Len, Code_Len + 1)
            # Labels: (Batch * MD_Len)
            # Note: The model masks invalid code positions with -inf, so they don't contribute to loss via softmax
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track Metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    def validate(self, dataloader, gt_df):
        """
        Runs validation and computes Kendall Tau score.
        """
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                ids = batch["ids"]
                md_cell_ids_batch = batch["md_cell_ids"]
                code_cell_ids_batch = batch["code_cell_ids"]

                code_emb = batch["code_embeddings"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                code_lens = batch["code_lens"].to(self.device)

                md_emb = batch["md_embeddings"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)
                md_lens = batch["md_lens"].to(self.device)

                # Forward Pass
                logits = self.model(
                    code_emb, code_mask, code_lens, md_emb, md_mask, md_lens
                )

                # Compute Probabilities
                probs = torch.softmax(logits, dim=-1)  # (B, L_md, L_code + 1)

                # Compute Expected Index (Soft Ranking)
                # We create a range tensor [0, 1, ..., Max_Code_Len]
                # Since the logits for padded code positions are -inf, their prob is 0.
                max_len = probs.size(-1)
                indices = torch.arange(max_len, device=self.device).float()

                # Expected index: sum(p_i * i)
                expected_indices = torch.sum(probs * indices, dim=-1)  # (B, L_md)

                # Reconstruct Notebook Orders
                expected_indices_cpu = expected_indices.cpu().numpy()

                for i, nb_id in enumerate(ids):
                    md_ids = md_cell_ids_batch[i]
                    code_ids = code_cell_ids_batch[i]
                    scores = expected_indices_cpu[i]

                    # Map scores to markdown IDs
                    # Note: scores contains padded values, we only take the first len(md_ids)
                    valid_scores = scores[: len(md_ids)]
                    pred_scores = {
                        mid: score for mid, score in zip(md_ids, valid_scores)
                    }

                    # Generate sorted order string
                    cell_order = get_ranks(pred_scores, code_ids)
                    preds.append({"id": nb_id, "cell_order": cell_order})

        df_preds = pd.DataFrame(preds)
        score = compute_kendall_tau(df_preds, gt_df)
        return score, df_preds

    def fit(self):
        """
        Main training loop.
        """
        set_seed(Config.SEED)

        # 1. Load Datasets
        print("Initializing datasets...")
        # Ensure features exist (assuming preprocessing has been run externally as per instructions)
        if not os.path.exists(Config.TRAIN_FEATURES_PATH) or not os.path.exists(
            Config.VAL_FEATURES_PATH
        ):
            raise FileNotFoundError(
                "Cached features not found. Please run preprocessing first."
            )

        train_dataset = CachedNotebookDataset(Config.TRAIN_FEATURES_PATH, split="train")
        val_dataset = CachedNotebookDataset(Config.VAL_FEATURES_PATH, split="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=CachedNotebookDataset.collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=CachedNotebookDataset.collate_fn,
            pin_memory=True,
        )

        # Load Ground Truth for Validation
        df_val_gt = pd.read_csv(Config.VAL_METADATA_PATH)

        print(
            f"Starting training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples."
        )

        # 2. Training Loop
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_score, _ = self.validate(val_loader, df_val_gt)

            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.8f} | Val Kendall Tau: {val_score:.8f}"
            )

            # 3. Checkpointing & Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Validation Score: {self.best_score:.8f}")


def run_training():
    """
    Helper function to instantiate the trainer and run the fit process.
    """
    trainer = Trainer()
    trainer.fit()
