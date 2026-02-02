import os
import time
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("engine")


class Trainer:
    """
    Handles the training and evaluation loop for the Interleaved Gap-Token Transformer.
    """

    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Loss Functions
        # Localization: Binary Cross Entropy (Logits -> Probabilities handled internally)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        # Identification: Cross Entropy (ignore padding/invalid targets)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        total_loc_loss = 0
        total_id_loss = 0

        # Metrics
        correct_loc = 0
        total_loc_samples = 0
        correct_id = 0
        total_id_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_word_id = batch["target_word_id"].to(self.device)
            target_gap_idx = batch["target_gap_idx"].to(self.device)

            batch_size = input_ids.size(0)

            # Forward Pass
            # loc_logits: (B, L, 1), id_logits: (B, L, V)
            loc_logits, id_logits = self.model(
                input_ids, token_type_ids, attention_mask
            )

            # --- Localization Loss ---
            loc_logits = loc_logits.squeeze(-1)  # (B, L)

            # Identify valid samples (where a gap target exists)
            valid_mask = target_gap_idx != -1

            # Construct Targets for BCE
            loc_targets = torch.zeros_like(loc_logits)
            if valid_mask.sum() > 0:
                # The gap index in the interleaved sequence is 2 * target_gap_idx
                valid_gap_indices = 2 * target_gap_idx[valid_mask]
                batch_indices = torch.arange(batch_size, device=self.device)[valid_mask]
                loc_targets[batch_indices, valid_gap_indices] = 1.0

            # Mask: Only calculate loss for Gap tokens (type 1) that are not padding
            loss_mask = (token_type_ids == 1) & (attention_mask == 1)

            bce_loss = self.bce_loss_fn(loc_logits, loc_targets)
            masked_bce_loss = (bce_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            # --- Identification Loss ---
            if valid_mask.sum() > 0:
                valid_gap_indices = 2 * target_gap_idx[valid_mask]
                batch_indices = torch.arange(batch_size, device=self.device)[valid_mask]

                # Gather logits at the specific gap
                selected_id_logits = id_logits[batch_indices, valid_gap_indices, :]
                selected_targets = target_word_id[valid_mask]

                ce_loss = self.ce_loss_fn(selected_id_logits, selected_targets)
            else:
                ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Total Multi-Task Loss
            loss = Config.LAMBDA_LOC * masked_bce_loss + Config.LAMBDA_ID * ce_loss

            # Optimization Step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Metrics Accumulation
            total_loss += loss.item()
            total_loc_loss += masked_bce_loss.item()
            total_id_loss += ce_loss.item()

            with torch.no_grad():
                # Localization Accuracy
                # Mask non-gap tokens for argmax by setting to -inf
                gap_logits = loc_logits.clone()
                gap_logits[~loss_mask] = -float("inf")
                pred_gap_indices = torch.argmax(gap_logits, dim=1)

                if valid_mask.sum() > 0:
                    true_indices = 2 * target_gap_idx[valid_mask]
                    preds = pred_gap_indices[valid_mask]
                    correct_loc += (preds == true_indices).sum().item()
                    total_loc_samples += valid_mask.sum().item()

                    # Identification Accuracy
                    pred_words = torch.argmax(selected_id_logits, dim=1)
                    correct_id += (pred_words == selected_targets).sum().item()
                    total_id_samples += valid_mask.sum().item()

        # Average Metrics
        avg_loss = total_loss / len(dataloader)
        avg_loc_loss = total_loc_loss / len(dataloader)
        avg_id_loss = total_id_loss / len(dataloader)
        acc_loc = correct_loc / total_loc_samples if total_loc_samples > 0 else 0.0
        acc_id = correct_id / total_id_samples if total_id_samples > 0 else 0.0

        logger.info(
            f"Epoch {epoch_idx} Train | Loss: {avg_loss:.6f} (Loc: {avg_loc_loss:.6f}, ID: {avg_id_loss:.6f}) | Acc Loc: {acc_loc:.6f} | Acc ID: {acc_id:.6f}"
        )
        return avg_loss

    def evaluate(self, dataloader, epoch_idx=None):
        self.model.eval()
        total_loss = 0
        total_loc_loss = 0
        total_id_loss = 0
        correct_loc = 0
        total_loc_samples = 0
        correct_id = 0
        total_id_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_word_id = batch["target_word_id"].to(self.device)
                target_gap_idx = batch["target_gap_idx"].to(self.device)

                batch_size = input_ids.size(0)

                loc_logits, id_logits = self.model(
                    input_ids, token_type_ids, attention_mask
                )
                loc_logits = loc_logits.squeeze(-1)

                valid_mask = target_gap_idx != -1

                # Loc Loss
                loc_targets = torch.zeros_like(loc_logits)
                if valid_mask.sum() > 0:
                    valid_gap_indices = 2 * target_gap_idx[valid_mask]
                    batch_indices = torch.arange(batch_size, device=self.device)[
                        valid_mask
                    ]
                    loc_targets[batch_indices, valid_gap_indices] = 1.0

                loss_mask = (token_type_ids == 1) & (attention_mask == 1)
                bce_loss = self.bce_loss_fn(loc_logits, loc_targets)
                masked_bce_loss = (bce_loss * loss_mask).sum() / (
                    loss_mask.sum() + 1e-8
                )

                # ID Loss
                if valid_mask.sum() > 0:
                    valid_gap_indices = 2 * target_gap_idx[valid_mask]
                    batch_indices = torch.arange(batch_size, device=self.device)[
                        valid_mask
                    ]
                    selected_id_logits = id_logits[batch_indices, valid_gap_indices, :]
                    selected_targets = target_word_id[valid_mask]
                    ce_loss = self.ce_loss_fn(selected_id_logits, selected_targets)
                else:
                    ce_loss = torch.tensor(0.0, device=self.device)

                loss = Config.LAMBDA_LOC * masked_bce_loss + Config.LAMBDA_ID * ce_loss

                total_loss += loss.item()
                total_loc_loss += masked_bce_loss.item()
                total_id_loss += ce_loss.item()

                # Metrics
                gap_logits = loc_logits.clone()
                gap_logits[~loss_mask] = -float("inf")
                pred_gap_indices = torch.argmax(gap_logits, dim=1)

                if valid_mask.sum() > 0:
                    true_indices = 2 * target_gap_idx[valid_mask]
                    preds = pred_gap_indices[valid_mask]
                    correct_loc += (preds == true_indices).sum().item()
                    total_loc_samples += valid_mask.sum().item()

                    pred_words = torch.argmax(selected_id_logits, dim=1)
                    correct_id += (pred_words == selected_targets).sum().item()
                    total_id_samples += valid_mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        avg_loc_loss = total_loc_loss / len(dataloader)
        avg_id_loss = total_id_loss / len(dataloader)
        acc_loc = correct_loc / total_loc_samples if total_loc_samples > 0 else 0.0
        acc_id = correct_id / total_id_samples if total_id_samples > 0 else 0.0

        prefix = f"Epoch {epoch_idx} " if epoch_idx is not None else ""
        logger.info(
            f"{prefix}Val   | Loss: {avg_loss:.6f} (Loc: {avg_loc_loss:.6f}, ID: {avg_id_loss:.6f}) | Acc Loc: {acc_loc:.6f} | Acc ID: {acc_id:.6f}"
        )

        return avg_loss


def fit_model(model, train_loader, val_loader):
    """
    Main training routine with Early Stopping.
    """
    device = Config.DEVICE
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
    )

    trainer = Trainer(model, optimizer, scheduler, device)

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader, epoch)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info(f"New best model saved with loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )
            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")


def generate_submission(model, test_loader, vocab, output_file=Config.SUBMISSION_FILE):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Uses score fusion of Localization and Identification probabilities.
    """
    logger.info("Generating submission...")
    model.eval()
    device = Config.DEVICE
    model.to(device)

    results = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            original_ids = batch["original_ids"]

            # Forward
            loc_logits, id_logits = model(input_ids, token_type_ids, attention_mask)

            # Score Fusion
            loc_probs = torch.sigmoid(loc_logits).squeeze(-1)  # (B, L)
            id_probs = torch.softmax(id_logits, dim=-1)  # (B, L, V)

            # Get max word prob for each position
            max_id_probs, best_word_indices = torch.max(id_probs, dim=-1)  # (B, L)

            # Combined Score
            combined_scores = loc_probs * max_id_probs

            # Mask non-gap tokens (type != 1) and padding (attention_mask == 0)
            mask = (token_type_ids == 1) & (attention_mask == 1)
            combined_scores[~mask] = -1.0

            # Find best gap
            best_gap_indices = torch.argmax(combined_scores, dim=1)  # (B,)

            # Get corresponding word
            final_word_ids = best_word_indices.gather(
                1, best_gap_indices.unsqueeze(1)
            ).squeeze(1)

            # Reconstruction
            input_ids_cpu = input_ids.cpu().numpy()
            best_gap_indices_cpu = best_gap_indices.cpu().numpy()
            final_word_ids_cpu = final_word_ids.cpu().numpy()

            for i in range(len(original_ids)):
                oid = original_ids[i]
                seq = input_ids_cpu[i]
                gap_idx = best_gap_indices_cpu[i]
                pred_word_id = final_word_ids_cpu[i]

                # Extract original words (odd indices in interleaved seq)
                # seq: [G, W, G, W, G, PAD...]
                current_words_ids = []
                for idx, token_id in enumerate(seq):
                    if token_id == Config.PAD_IDX:
                        break
                    if idx % 2 == 1:
                        current_words_ids.append(token_id)

                # Insertion Position
                # gap_idx is index in interleaved sequence.
                # gap 0 -> before w0 -> insert at 0
                # gap 2 -> between w0, w1 -> insert at 1
                insert_pos = gap_idx // 2

                # Safety clip
                if insert_pos > len(current_words_ids):
                    insert_pos = len(current_words_ids)

                current_words_ids.insert(insert_pos, pred_word_id)

                # Decode
                sentence = vocab.decode(current_words_ids)
                results.append({"id": oid, "sentence": sentence})

    # Save to CSV
    df = pd.DataFrame(results)
    # Using quote_all to ensure format matches requirements
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Submission saved to {output_file}")
