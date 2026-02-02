import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import get_dataloaders
from library.models import LocatorNetwork, FillerNetwork


class Trainer:
    """
    Manages the training of Locator and Filler models, and generates submissions.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        self.logger = setup_logger("Trainer")

        # Setup directories
        os.makedirs(Config.LOCATOR_MODEL_DIR, exist_ok=True)
        os.makedirs(Config.FILLER_MODEL_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Tokenizer (needed for inference/decoding)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

        # Loss functions
        # Locator: Pointer Head uses CrossEntropy over the sequence dimension.
        # Cite solution_lesson_node_00001
        self.locator_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.filler_criterion = nn.CrossEntropyLoss()

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_checkpoint(self, model, path):
        if os.path.exists(path):
            self.logger.info(f"Loading checkpoint from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False

    def train_locator(self):
        """
        Trains the LocatorNetwork to identify the position of the missing word.
        """
        self.logger.info("Starting Locator Training...")

        # Initialize Model
        model = LocatorNetwork().to(self.device)

        # Hyperparameters
        params = Config.LOCATOR_PARAMS
        optimizer = optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        # DataLoaders
        train_loader, val_loader = get_dataloaders(
            "locator_train", tokenizer=self.tokenizer, debug=self.debug
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.LOCATOR_MODEL_DIR, "best_locator.pth")

        for epoch in range(params["epochs"]):
            model.train()
            train_loss = 0.0

            # Training Loop
            for batch in tqdm(
                train_loader, desc=f"Locator Epoch {epoch+1}/{params['epochs']} [Train]"
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_idx = batch["label_idx"].to(self.device)  # Shape: (B,)

                optimizer.zero_grad()

                # Forward
                logits = model(input_ids, attention_mask)  # Shape: (B, L)

                # Mask padding tokens so they are not selected
                # Set logits at padding positions to a very small number
                logits = logits.masked_fill(attention_mask == 0, -1e9)

                # Loss: CrossEntropy over sequence dimension
                loss = self.locator_criterion(logits, label_idx)

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])

                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total_seqs = 0

            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"Locator Epoch {epoch+1}/{params['epochs']} [Val]"
                ):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    label_idx = batch["label_idx"].to(self.device)

                    logits = model(input_ids, attention_mask)
                    logits = logits.masked_fill(attention_mask == 0, -1e9)

                    loss = self.locator_criterion(logits, label_idx)
                    val_loss += loss.item()

                    # Metrics: Sequence Accuracy
                    preds = torch.argmax(logits, dim=1)  # (B,)

                    # Only count valid samples (where label_idx != -100)
                    valid_mask = label_idx != -100
                    correct += ((preds == label_idx) & valid_mask).sum().item()
                    total_seqs += valid_mask.sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total_seqs if total_seqs > 0 else 0.0

            self.logger.info(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val Seq Acc: {val_acc}"
            )

            # Early Stopping & Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if params["save_best_only"]:
                    self.save_checkpoint(model, best_model_path)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{params['early_stopping_patience']}"
                )
                if patience_counter >= params["early_stopping_patience"]:
                    self.logger.info("Early stopping triggered.")
                    break

        # Load best model for return
        if params["save_best_only"] and os.path.exists(best_model_path):
            self.load_checkpoint(model, best_model_path)

        return model

    def train_filler(self):
        """
        Trains the FillerNetwork (MLM) to predict the missing word.
        """
        self.logger.info("Starting Filler Training...")

        model = FillerNetwork().to(self.device)

        params = Config.FILLER_PARAMS
        optimizer = optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        train_loader, val_loader = get_dataloaders(
            "filler_train", tokenizer=self.tokenizer, debug=self.debug
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.FILLER_MODEL_DIR, "best_filler.pth")

        for epoch in range(params["epochs"]):
            model.train()
            train_loss = 0.0

            for batch in tqdm(
                train_loader, desc=f"Filler Epoch {epoch+1}/{params['epochs']} [Train]"
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                logits = model(input_ids, attention_mask)  # (B, L, V)

                # MLM Loss (CrossEntropy ignores -100 by default)
                loss = self.filler_criterion(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"Filler Epoch {epoch+1}/{params['epochs']} [Val]"
                ):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = model(input_ids, attention_mask)
                    loss = self.filler_criterion(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

            self.logger.info(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val PPL: {perplexity}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if params["save_best_only"]:
                    self.save_checkpoint(model, best_model_path)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{params['early_stopping_patience']}"
                )
                if patience_counter >= params["early_stopping_patience"]:
                    self.logger.info("Early stopping triggered.")
                    break

        if params["save_best_only"] and os.path.exists(best_model_path):
            self.load_checkpoint(model, best_model_path)

        return model

    def generate_submission(self):
        """
        Runs the inference pipeline:
        1. Load best Locator and Filler models.
        2. Predict gap location.
        3. Insert [MASK] token.
        4. Predict missing word.
        5. Reconstruct sentence and save to CSV.
        """
        self.logger.info("Starting Submission Generation...")

        # Load Models
        locator = LocatorNetwork().to(self.device)
        locator_path = os.path.join(Config.LOCATOR_MODEL_DIR, "best_locator.pth")
        if not self.load_checkpoint(locator, locator_path):
            self.logger.warning(
                "Locator checkpoint not found! Training might have failed or skipped."
            )
            return

        filler = FillerNetwork().to(self.device)
        filler_path = os.path.join(Config.FILLER_MODEL_DIR, "best_filler.pth")
        if not self.load_checkpoint(filler, filler_path):
            self.logger.warning(
                "Filler checkpoint not found! Training might have failed or skipped."
            )
            return

        locator.eval()
        filler.eval()

        test_loader = get_dataloaders(
            "submission", tokenizer=self.tokenizer, debug=self.debug
        )

        results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating Predictions"):
                ids = batch["id"]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                raw_sentences = batch["raw_sentence"]  # List of strings

                # --- Step 1: Locate ---
                # Cite solution_lesson_node_00001: Pointer Head Inference
                loc_logits = locator(input_ids, attention_mask)  # (B, L)

                # Mask out padding tokens
                loc_logits = loc_logits.masked_fill(attention_mask == 0, -1e9)

                # Find argmax index
                # This index 'k' means the gap is AFTER token k.
                pred_indices = torch.argmax(loc_logits, dim=1)  # (B,)

                # --- Step 2 & 3: Construct Masked Input & Fill ---
                # We need to insert [MASK] token after the predicted index for each item in batch.
                # Since insertion changes length, and lengths might vary, we process construction carefully.

                # We will process reconstruction and filler prediction row-by-row or using list comprehension
                # because tensor manipulation with variable insertion indices is complex to vectorize efficiently
                # without advanced scatter/gather or padding management.

                batch_preds = []

                for i in range(len(ids)):
                    curr_input_ids = input_ids[i]
                    curr_idx = pred_indices[i].item()
                    curr_len = attention_mask[i].sum().item()

                    # Ensure index is within valid bounds (not at the very end of sequence/padding)
                    if curr_idx >= curr_len - 1:
                        curr_idx = (
                            curr_len - 2
                        )  # Default to before last token if predicted out of bounds

                    # Construct new input_ids
                    # [0...curr_idx] + [MASK] + [curr_idx+1...end]
                    # Note: We must handle truncation if max_len is reached, but usually we have buffer.

                    mask_token_id = self.tokenizer.mask_token_id

                    # Slicing tensors
                    prefix = curr_input_ids[: curr_idx + 1]
                    suffix = curr_input_ids[curr_idx + 1 :]

                    # Create new tensor with mask inserted
                    # We need to ensure we don't exceed model max length, though Config.MAX_LEN is 128
                    # and most sentences are shorter.

                    new_ids_list = torch.cat(
                        [
                            prefix,
                            torch.tensor([mask_token_id], device=self.device),
                            suffix,
                        ]
                    )

                    # Truncate if necessary (rare)
                    if len(new_ids_list) > Config.MAX_LEN:
                        new_ids_list = new_ids_list[: Config.MAX_LEN]

                    # Create new attention mask (all 1s for the valid length)
                    new_mask_list = torch.ones_like(new_ids_list)

                    # Add batch dimension
                    new_ids_list = new_ids_list.unsqueeze(0)
                    new_mask_list = new_mask_list.unsqueeze(0)

                    # --- Step 3: Fill ---
                    # Run filler on single instance (or accumulate and run batch if optimizing speed)
                    # For simplicity and safety, running per instance here inside the batch loop.
                    # (Optimizable, but sufficient for 300k samples in 24h).

                    fill_logits = filler(new_ids_list, new_mask_list)  # (1, L_new, V)

                    # The mask token is at index `curr_idx + 1`
                    mask_pos = curr_idx + 1
                    if mask_pos >= fill_logits.size(1):
                        mask_pos = fill_logits.size(1) - 1

                    pred_token_id = torch.argmax(fill_logits[0, mask_pos]).item()
                    pred_word = self.tokenizer.decode(
                        [pred_token_id], clean_up_tokenization_spaces=True
                    ).strip()

                    # --- Step 4: Reconstruct Sentence ---
                    # We have the raw sentence. We need to insert the word.
                    # The locator prediction `curr_idx` is in Token Space.
                    # Mapping Token Index -> Character Index is tricky.
                    # Strategy: Use the raw tokens, insert word, detokenize/join.

                    # Get original tokens (excluding special tokens [CLS], [SEP], [PAD])
                    # Note: tokenizer.convert_ids_to_tokens keeps '##' subwords.
                    # Better to use offsets if available, but we don't have offsets in SubmissionDataset by default.
                    # Fallback: Use the tokenized list, insert, then decode.

                    # Re-tokenize raw sentence to get tokens matching our indices
                    # (This is redundant but ensures alignment with pred_indices)
                    # Ideally, we trust the `curr_idx` relative to the `curr_input_ids`.

                    # Remove [CLS] (index 0) and [SEP] (last valid) to get pure text tokens
                    # curr_idx includes [CLS]. So if curr_idx=1, it's after the first real word token.

                    # Let's decode the prefix and suffix separately and join.
                    # prefix_ids = curr_input_ids[1 : curr_idx+1] (skip CLS)
                    # suffix_ids = curr_input_ids[curr_idx+1 : curr_len-1] (skip SEP)

                    # Handle edge cases where curr_idx is 0 (after CLS)
                    start_pos = 1
                    end_pos = curr_len - 1  # Position of SEP

                    if curr_idx < start_pos:
                        curr_idx = start_pos  # Insert at start

                    prefix_ids = curr_input_ids[start_pos : curr_idx + 1]
                    suffix_ids = curr_input_ids[curr_idx + 1 : end_pos]

                    prefix_str = self.tokenizer.decode(
                        prefix_ids, clean_up_tokenization_spaces=True
                    )
                    suffix_str = self.tokenizer.decode(
                        suffix_ids, clean_up_tokenization_spaces=True
                    )

                    # Construct final string
                    # Handle spacing heuristics
                    if prefix_str and not prefix_str.endswith(" "):
                        prefix_str += " "

                    final_sent = f"{prefix_str}{pred_word} {suffix_str}".strip()

                    # Clean up double spaces if any
                    final_sent = " ".join(final_sent.split())

                    # Escape double quotes for CSV format as per requirement
                    # "Use double quotes to escape the sentence text and two double quotes ("") for double quotes within a sentence."
                    # Pandas to_csv handles the outer quoting, we just need to escape inner quotes.
                    # Actually, pandas handles inner quote escaping automatically if we set quoting correctly.
                    # We just store the raw string.

                    results.append({"id": ids[i].item(), "sentence": final_sent})

        # Save to CSV
        df_sub = pd.DataFrame(results)
        # Ensure correct column order
        df_sub = df_sub[["id", "sentence"]]

        # Write CSV
        # quoting=1 is csv.QUOTE_ALL (quote all fields)
        # doublequote=True is default (escape " with "")
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False, quoting=1)

        self.logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    # Set global seed
    set_seed(Config.SEED)

    # Initialize Trainer
    # Set debug=Config.DEBUG to use smaller subset if configured
    trainer = Trainer(debug=Config.DEBUG)

    # Train Locator
    trainer.train_locator()

    # Train Filler
    trainer.train_filler()

    # Generate Submission
    trainer.generate_submission()


if __name__ == "__main__":
    main()
