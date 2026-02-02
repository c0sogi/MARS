import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_

from library.config import Config
from library.model import DecoupledTransformer
from library.dataset import get_dataloaders
from library.utils import get_or_build_vocab, SOS_TOKEN, EOS_TOKEN


class Trainer:
    """
    Trainer class for the Decoupled Localization-Classification Transformer.
    Handles training, validation, and submission generation.
    """

    def __init__(self):
        # Ensure reproducibility
        Config.set_seed()

        self.device = torch.device(Config.DEVICE)

        # Load Vocabulary
        # We need the vocab to determine model output size and for decoding in submission
        self.vocab = get_or_build_vocab(load_cached_data=True)
        self.vocab_size = len(self.vocab)

        # Initialize Model
        self.model = DecoupledTransformer(self.vocab_size).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        # Localization: Binary Cross Entropy
        # We use a positive weight to account for the imbalance (1 gap vs L-1 non-gaps)
        # A weight of 10.0 is a reasonable heuristic for average sentence lengths.
        self.loc_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([10.0]).to(self.device)
        )

        # Identification: Cross Entropy
        self.id_criterion = nn.CrossEntropyLoss()

        # Scheduler (initialized in fit)
        self.scheduler = None

    def train_epoch(self, loader, epoch_idx):
        """
        Runs one training epoch.
        """
        self.model.train()
        total_loss = 0
        total_loc_loss = 0
        total_id_loss = 0

        correct_loc = 0
        correct_id = 0
        total_samples = 0

        for batch in loader:
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            gap_idxs = batch["gap_idx"].to(self.device)
            target_ids = batch["target_id"].to(self.device)

            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)

            # Forward Pass
            self.optimizer.zero_grad()
            loc_logits, id_logits = self.model(input_ids, attention_mask)

            # --- Localization Loss ---
            # Target: (B, L) with 1.0 at gap_idx
            loc_targets = torch.zeros((batch_size, seq_len), device=self.device)
            loc_targets.scatter_(1, gap_idxs.unsqueeze(1), 1.0)

            loc_loss = self.loc_criterion(loc_logits, loc_targets)

            # --- Identification Loss ---
            # We only care about the prediction at the gap index.
            # Select logits corresponding to the gap position: (B, V)
            batch_indices = torch.arange(batch_size, device=self.device)
            pred_logits_at_gap = id_logits[batch_indices, gap_idxs, :]

            id_loss = self.id_criterion(pred_logits_at_gap, target_ids)

            # --- Total Loss ---
            loss = (Config.LAMBDA_LOC * loc_loss) + (Config.LAMBDA_ID * id_loss)

            # Backward Pass
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # --- Metrics Tracking ---
            total_loss += loss.item() * batch_size
            total_loc_loss += loc_loss.item() * batch_size
            total_id_loss += id_loss.item() * batch_size

            # Loc Accuracy
            # Mask padding for argmax to ensure we don't predict padding as gap
            masked_loc_logits = loc_logits.clone()
            masked_loc_logits[attention_mask == 0] = float("-inf")
            pred_gaps = torch.argmax(masked_loc_logits, dim=1)
            correct_loc += (pred_gaps == gap_idxs).sum().item()

            # Id Accuracy
            pred_ids = torch.argmax(pred_logits_at_gap, dim=1)
            correct_id += (pred_ids == target_ids).sum().item()

            total_samples += batch_size

        # Calculate Averages
        avg_loss = total_loss / total_samples
        avg_loc_loss = total_loc_loss / total_samples
        avg_id_loss = total_id_loss / total_samples
        acc_loc = correct_loc / total_samples
        acc_id = correct_id / total_samples

        print(
            f"Epoch {epoch_idx+1} Train | Loss: {avg_loss} | Loc Loss: {avg_loc_loss} | Id Loss: {avg_id_loss} | Loc Acc: {acc_loc} | Id Acc: {acc_id}"
        )

    def validate(self, loader, epoch_idx):
        """
        Runs validation loop.
        """
        self.model.eval()
        total_loss = 0
        correct_loc = 0
        correct_id = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gap_idxs = batch["gap_idx"].to(self.device)
                target_ids = batch["target_id"].to(self.device)

                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)

                loc_logits, id_logits = self.model(input_ids, attention_mask)

                # Loc Loss
                loc_targets = torch.zeros((batch_size, seq_len), device=self.device)
                loc_targets.scatter_(1, gap_idxs.unsqueeze(1), 1.0)
                loc_loss = self.loc_criterion(loc_logits, loc_targets)

                # Id Loss
                batch_indices = torch.arange(batch_size, device=self.device)
                pred_logits_at_gap = id_logits[batch_indices, gap_idxs, :]
                id_loss = self.id_criterion(pred_logits_at_gap, target_ids)

                loss = (Config.LAMBDA_LOC * loc_loss) + (Config.LAMBDA_ID * id_loss)

                total_loss += loss.item() * batch_size

                # Metrics
                masked_loc_logits = loc_logits.clone()
                masked_loc_logits[attention_mask == 0] = float("-inf")
                pred_gaps = torch.argmax(masked_loc_logits, dim=1)
                correct_loc += (pred_gaps == gap_idxs).sum().item()

                pred_ids = torch.argmax(pred_logits_at_gap, dim=1)
                correct_id += (pred_ids == target_ids).sum().item()

                total_samples += batch_size

        avg_loss = total_loss / total_samples
        acc_loc = correct_loc / total_samples
        acc_id = correct_id / total_samples

        print(
            f"Epoch {epoch_idx+1} Val | Loss: {avg_loss} | Loc Acc: {acc_loc} | Id Acc: {acc_id}"
        )
        return avg_loss

    def fit(self):
        """
        Main training orchestration with Early Stopping.
        """
        print("Initializing Dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(
            self.vocab, load_cached_data=True
        )

        # Setup Scheduler
        total_steps = len(train_loader) * Config.MAX_EPOCHS
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps
        )

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {Config.MAX_EPOCHS} epochs...")

        for epoch in range(Config.MAX_EPOCHS):
            self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved to {Config.MODEL_PATH}")
            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model for submission
        print("Loading best model for submission generation...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=self.device)
        )

        # Generate Submission
        self.generate_submission(test_loader)

    def generate_submission(self, loader):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")
        self.model.eval()
        results = []

        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=-1)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                row_ids = batch["row_id"].cpu().numpy()

                batch_size = input_ids.size(0)

                loc_logits, id_logits = self.model(input_ids, attention_mask)

                # Calculate Joint Score Matrix S = P(Loc) * P(Word)
                # P(Loc): (B, L, 1)
                p_loc = sigmoid(loc_logits).unsqueeze(-1)

                # P(Word): (B, L, V)
                p_word = softmax(id_logits)

                # Score: (B, L, V)
                scores = p_loc * p_word

                # Mask out padding in scores to prevent invalid insertions
                # (B, L, 1)
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(scores)
                scores = scores * mask_expanded

                # Flatten to find global max per sample
                # (B, L*V)
                scores_flat = scores.view(batch_size, -1)
                best_flat_indices = torch.argmax(scores_flat, dim=1)

                # Convert back to (pos, word_idx)
                best_pos = best_flat_indices // self.vocab_size
                best_word_idx = best_flat_indices % self.vocab_size

                # Reconstruct sentences
                input_ids_cpu = input_ids.cpu().numpy()
                best_pos_cpu = best_pos.cpu().numpy()
                best_word_idx_cpu = best_word_idx.cpu().numpy()

                for i in range(batch_size):
                    rid = row_ids[i]
                    curr_ids = input_ids_cpu[i]
                    pos = best_pos_cpu[i]
                    w_idx = best_word_idx_cpu[i]

                    # Determine real length (exclude padding)
                    # We look for EOS token or use len(curr_ids) if not found (though EOS should be there)
                    eos_mask = curr_ids == self.vocab.stoi.get(EOS_TOKEN, 3)
                    if eos_mask.any():
                        # Include EOS in the valid tokens list initially
                        real_len = np.argmax(eos_mask) + 1
                    else:
                        real_len = len(curr_ids)

                    valid_tokens = list(curr_ids[:real_len])

                    # Insert word
                    # 'pos' is the index *after* which we insert.
                    # list.insert(index, obj) inserts *before* index.
                    # So to insert after 'pos', we insert at 'pos + 1'.
                    valid_tokens.insert(pos + 1, w_idx)

                    # Decode to string
                    decoded_tokens = self.vocab.decode(valid_tokens)

                    # Filter special tokens (SOS, EOS, PAD, MASK)
                    special_tokens = [
                        SOS_TOKEN,
                        EOS_TOKEN,
                        Config.PAD_TOKEN,
                        Config.MASK_TOKEN,
                    ]
                    clean_tokens = [
                        t for t in decoded_tokens if t not in special_tokens
                    ]

                    # Join tokens
                    sentence = " ".join(clean_tokens)

                    # Escape quotes for CSV compliance
                    sentence = sentence.replace('"', '""')

                    results.append(f'{rid},"{sentence}"')

        # Save to file
        with open(Config.SUBMISSION_PATH, "w", encoding="utf-8") as f:
            f.write('id,"sentence"\n')
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
