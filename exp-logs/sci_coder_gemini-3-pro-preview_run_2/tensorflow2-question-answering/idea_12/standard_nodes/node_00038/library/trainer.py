import torch
import torch.nn as nn
import torch.optim as optim
import os
from library.config import Config


class Trainer:
    """
    Handles the training, evaluation, and optimization of the FiLMNetwork model.
    """

    def __init__(self, model, device):
        """
        Args:
            model (nn.Module): The FiLMNetwork model.
            device (torch.device): Device to run training on (CPU/GPU).
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Loss Functions
        # Ranking is binary (Correct Candidate vs Incorrect Candidate)
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Spans and Yes/No are multiclass classification problems
        self.ce_loss = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, epoch_idx):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): Training dataloader.
            epoch_idx (int): Current epoch index (for display).

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            q_input = batch["q_input"].to(self.device)
            pos_cand = batch["pos_cand_input"].to(self.device)
            neg_cand = batch["neg_cand_input"].to(self.device)

            # Move targets to device
            pos_label = batch["pos_label"].to(self.device)
            neg_label = batch["neg_label"].to(self.device)
            short_start = batch["short_start"].to(self.device)
            short_end = batch["short_end"].to(self.device)
            yn_label = batch["yn_label"].to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()

            # --- Forward Pass (Positive Samples) ---
            # We compute all heads for positive samples
            pos_out = self.model(q_input, pos_cand)

            # 1. Ranking Loss (Positive)
            # Squeeze to match label shape (Batch,)
            loss_rank_pos = self.bce_loss(pos_out["rank_logits"].squeeze(1), pos_label)

            # 2. Span Loss (Positive only)
            loss_start = self.ce_loss(pos_out["start_logits"], short_start)
            loss_end = self.ce_loss(pos_out["end_logits"], short_end)

            # 3. Yes/No Loss (Positive only)
            loss_yn = self.ce_loss(pos_out["yesno_logits"], yn_label)

            # --- Forward Pass (Negative Samples) ---
            # We only care about ranking loss for negatives (pushing their score down)
            neg_out = self.model(q_input, neg_cand)
            loss_rank_neg = self.bce_loss(neg_out["rank_logits"].squeeze(1), neg_label)

            # --- Combine Losses ---
            # Average ranking loss between positive and negative pair
            loss_ranking = (loss_rank_pos + loss_rank_neg) / 2.0

            # Average start and end span losses
            loss_span = (loss_start + loss_end) / 2.0

            # Weighted sum
            total_batch_loss = (
                Config.LOSS_WEIGHT_RANKING * loss_ranking
                + Config.LOSS_WEIGHT_SPAN * loss_span
                + Config.LOSS_WEIGHT_YESNO * loss_yn
            )

            # Backward pass and optimization
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch_idx + 1} Training Loss: {avg_loss}")
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Calculates ranking loss and ranking accuracy.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            float: Average validation loss (ranking).
        """
        self.model.eval()
        total_val_loss = 0.0
        correct_rankings = 0
        total_examples_with_answer = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                # Validation batch size is 1, but contains multiple candidates
                # Shapes: q_input (1, Q_Len), candidates (1, N, Ctx_Len), label_idx (1,)

                q_input = batch["q_input"].to(self.device)
                candidates = batch["candidates"].to(self.device)
                label_idx = batch["label_idx"].item()

                # Remove batch dimension to get (N, Ctx_Len)
                candidates = candidates.squeeze(0)
                num_cands = candidates.size(0)

                if num_cands == 0:
                    continue

                # Repeat question vector to match number of candidates: (N, Q_Len)
                q_input_expanded = q_input.repeat(num_cands, 1)

                # Forward pass
                outputs = self.model(q_input_expanded, candidates)
                rank_logits = outputs["rank_logits"].squeeze(1)  # (N,)

                # --- Validation Loss (Ranking) ---
                # Construct target vector: all 0s, with 1 at label_idx if valid
                rank_targets = torch.zeros(num_cands, device=self.device)
                if label_idx != -1 and label_idx < num_cands:
                    rank_targets[label_idx] = 1.0

                loss_rank = self.bce_loss(rank_logits, rank_targets)
                total_val_loss += loss_rank.item()

                # --- Metrics (Ranking Accuracy) ---
                # Only compute accuracy for examples that HAVE a correct answer in the candidate list
                if label_idx != -1 and label_idx < num_cands:
                    total_examples_with_answer += 1
                    # Prediction is the candidate with the highest logit
                    best_cand_idx = torch.argmax(rank_logits).item()
                    if best_cand_idx == label_idx:
                        correct_rankings += 1

        avg_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        acc = (
            correct_rankings / total_examples_with_answer
            if total_examples_with_answer > 0
            else 0.0
        )

        print(f"Validation Ranking Loss: {avg_loss}")
        print(f"Validation Ranking Accuracy: {acc}")

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            num_epochs (int): Maximum number of epochs.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader)

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(
                    f"No improvement in validation loss. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")
