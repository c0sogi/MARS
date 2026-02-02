import os
import csv
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from library.config import Config
from library.models import PointerLocator, get_filler_model
from library.utils import setup_logger


class Predictor:
    """
    Handles inference for the Two-Stage (Locator + Filler) model.
    Implements the Offset-Insertion strategy to preserve original sentence formatting.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.logger = setup_logger(
            "Predictor", os.path.join(Config.OUTPUT_DIR, "prediction.log")
        )

        self.logger.info(f"Initializing Predictor on {self.device}...")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # Load Locator Model
        self.logger.info(f"Loading Locator from {Config.BEST_LOCATOR_PATH}")
        self.locator = PointerLocator(model_name=Config.MODEL_NAME)

        if os.path.exists(Config.BEST_LOCATOR_PATH):
            state_dict = torch.load(Config.BEST_LOCATOR_PATH, map_location=self.device)
            self.locator.load_state_dict(state_dict)
        else:
            self.logger.warning(
                "Locator checkpoint not found! Using random weights (Expect poor performance)."
            )

        self.locator.to(self.device)
        self.locator.eval()

        # Load Filler Model
        self.logger.info(f"Loading Filler from {Config.BEST_FILLER_PATH}")
        self.filler = get_filler_model(model_name=Config.MODEL_NAME)

        if os.path.exists(Config.BEST_FILLER_PATH):
            state_dict = torch.load(Config.BEST_FILLER_PATH, map_location=self.device)
            self.filler.load_state_dict(state_dict)
        else:
            self.logger.warning("Filler checkpoint not found! Using random weights.")

        self.filler.to(self.device)
        self.filler.eval()

    def predict_batch(self, batch):
        """
        Performs inference on a single batch using the Locator -> Filler pipeline.

        Args:
            batch (dict): Batch dictionary from DataLoader containing inputs and metadata.

        Returns:
            list: List of tuples (id, predicted_sentence).
        """
        # Move inputs to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Metadata (keep on CPU)
        offset_mapping = batch["offset_mapping"]
        original_sentences = batch["sentence"]
        ids = batch["id"]

        batch_size = input_ids.size(0)

        # -------------------------------------------------------
        # 1. Locator Inference
        # -------------------------------------------------------
        # Predict the token index immediately PRECEDING the gap
        with torch.no_grad():
            locator_logits = self.locator(input_ids, attention_mask)
            pred_loc_indices = torch.argmax(locator_logits, dim=1)  # (batch_size,)

        # -------------------------------------------------------
        # 2. Prepare Filler Inputs
        # -------------------------------------------------------
        # Dynamically insert <mask> token into the tensor sequences

        filler_input_ids_list = []
        filler_attention_mask_list = []
        mask_token_id = self.tokenizer.mask_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Track mask positions for extraction
        mask_indices = []

        for i in range(batch_size):
            loc_idx = pred_loc_indices[i].item()
            curr_ids = input_ids[i]
            curr_mask = attention_mask[i]

            # Determine actual sequence length (ignoring padding)
            valid_len = curr_mask.sum().item()

            # Clamp loc_idx to be within valid range
            if loc_idx >= valid_len:
                loc_idx = valid_len - 1

            # Construct new sequence: [Prefix] + [MASK] + [Suffix]
            # Prefix includes the token at loc_idx
            prefix = curr_ids[: loc_idx + 1]
            suffix = curr_ids[loc_idx + 1 : valid_len]

            mask_tensor = torch.tensor([mask_token_id], device=self.device)

            new_seq = torch.cat([prefix, mask_tensor, suffix])

            # Truncate if necessary (rare given MAX_LEN buffer)
            if len(new_seq) > Config.MAX_LEN:
                new_seq = new_seq[: Config.MAX_LEN]

            # Create new attention mask (1s for data)
            new_att = torch.ones_like(new_seq)

            # Pad back to MAX_LEN
            padding_len = Config.MAX_LEN - len(new_seq)
            if padding_len > 0:
                pad_seq = torch.full((padding_len,), pad_token_id, device=self.device)
                new_seq = torch.cat([new_seq, pad_seq])

                pad_att = torch.zeros((padding_len,), device=self.device)
                new_att = torch.cat([new_att, pad_att])

            filler_input_ids_list.append(new_seq)
            filler_attention_mask_list.append(new_att)

            # The mask is inserted at index `len(prefix)` which equals `loc_idx + 1`
            mask_pos = loc_idx + 1
            if mask_pos >= Config.MAX_LEN:
                mask_pos = Config.MAX_LEN - 1
            mask_indices.append(mask_pos)

        # Stack to create batch tensors
        filler_input_ids = torch.stack(filler_input_ids_list)
        filler_attention_mask = torch.stack(filler_attention_mask_list)

        # -------------------------------------------------------
        # 3. Filler Inference
        # -------------------------------------------------------
        with torch.no_grad():
            filler_outputs = self.filler(
                input_ids=filler_input_ids, attention_mask=filler_attention_mask
            )
            logits = filler_outputs.logits  # (batch_size, seq_len, vocab_size)

        # -------------------------------------------------------
        # 4. Reconstruction (Offset-Based)
        # -------------------------------------------------------
        predictions = []

        for i in range(batch_size):
            # A. Decode Predicted Word
            mask_pos = mask_indices[i]
            pred_token_id = torch.argmax(logits[i, mask_pos]).item()
            pred_word = self.tokenizer.decode(
                [pred_token_id], clean_up_tokenization_spaces=True
            ).strip()

            # B. Identify Insertion Character Index
            loc_idx = pred_loc_indices[i].item()
            offsets = offset_mapping[i]

            # Handle edge cases where loc_idx might be out of bounds of offsets
            if loc_idx >= len(offsets):
                loc_idx = len(offsets) - 1

            # Get the end character position of the token preceding the gap
            # offsets[loc_idx] is [start, end]
            end_char_idx = offsets[loc_idx][1].item()

            # C. Insert into Original String
            orig_sent = original_sentences[i]

            # Safety check for string bounds
            if end_char_idx > len(orig_sent):
                end_char_idx = len(orig_sent)

            # Insert space + word.
            # This preserves all original punctuation and spacing of the source sentence.
            final_sent = (
                orig_sent[:end_char_idx] + " " + pred_word + orig_sent[end_char_idx:]
            )

            predictions.append((ids[i], final_sent))

        return predictions

    def generate_submission(self, test_loader):
        """
        Generates predictions for the entire test set and saves to CSV.
        """
        self.logger.info(f"Starting inference on {len(test_loader.dataset)} samples...")

        all_predictions = []

        # Iterate over test loader
        for batch in tqdm(test_loader, desc="Generating Predictions"):
            batch_preds = self.predict_batch(batch)
            all_predictions.extend(batch_preds)

        # Create DataFrame
        df_submission = pd.DataFrame(all_predictions, columns=["id", "sentence"])

        # Save to CSV
        # quoting=csv.QUOTE_NONNUMERIC ensures string fields (sentence) are quoted,
        # while numeric fields (id) are not, matching the submission format.
        df_submission.to_csv(
            Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC
        )

        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Log a few examples for verification
        self.logger.info("--- Sample Predictions ---")
        for i in range(min(3, len(df_submission))):
            self.logger.info(
                f"{df_submission.iloc[i]['id']}: {df_submission.iloc[i]['sentence']}"
            )
