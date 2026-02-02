import os
import csv
import torch
import pandas as pd
import logging
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import get_dataloaders
from library.models import LocatorNetwork, FillerNetwork


class InferencePipeline:
    """
    Manages the inference process: loading models, predicting missing words,
    and generating the submission file.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        self.logger = setup_logger("InferencePipeline")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
        self.locator = None
        self.filler = None

    def load_models(self):
        """
        Loads the trained Locator and Filler models from checkpoints.
        """
        self.logger.info("Loading models...")

        # Load Locator
        self.locator = LocatorNetwork().to(self.device)
        locator_path = os.path.join(Config.LOCATOR_MODEL_DIR, "best_locator.pth")

        if os.path.exists(locator_path):
            self.locator.load_state_dict(
                torch.load(locator_path, map_location=self.device)
            )
            self.locator.eval()
            self.logger.info(f"Locator loaded from {locator_path}")
        else:
            raise FileNotFoundError(f"Locator checkpoint not found at {locator_path}")

        # Load Filler
        self.filler = FillerNetwork().to(self.device)
        filler_path = os.path.join(Config.FILLER_MODEL_DIR, "best_filler.pth")

        if os.path.exists(filler_path):
            self.filler.load_state_dict(
                torch.load(filler_path, map_location=self.device)
            )
            self.filler.eval()
            self.logger.info(f"Filler loaded from {filler_path}")
        else:
            raise FileNotFoundError(f"Filler checkpoint not found at {filler_path}")

    def predict(self, test_loader):
        """
        Runs the two-stage prediction pipeline on the test loader.

        Args:
            test_loader (DataLoader): The dataloader for the submission dataset.

        Returns:
            list: A list of dictionaries containing 'id' and 'sentence'.
        """
        self.logger.info("Starting prediction loop...")
        results = []

        # Disable gradients for inference
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["id"]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # --- Step 1: Locate the Gap ---
                # Forward pass through Locator
                loc_logits = self.locator(input_ids, attention_mask)  # (B, L, 2)

                # Get probability of class 1 (Gap)
                gap_probs = torch.softmax(loc_logits, dim=2)[:, :, 1]

                # Mask out padding to prevent predicting gap in padding
                gap_probs = gap_probs * attention_mask

                # Find the token index after which the gap exists
                pred_indices = torch.argmax(gap_probs, dim=1)  # (B,)

                # --- Step 2 & 3: Construct Masked Input & Fill ---
                # Process each sentence in the batch
                for i in range(len(ids)):
                    curr_input_ids = input_ids[i]
                    curr_idx = pred_indices[i].item()
                    curr_len = attention_mask[i].sum().item()

                    # Boundary checks
                    # We assume [CLS] is at 0 and [SEP] is at curr_len-1
                    # If curr_idx points to [SEP] or padding, clamp it.
                    if curr_idx >= curr_len - 1:
                        curr_idx = curr_len - 2

                    # Construct new input sequence with [MASK] inserted
                    mask_token_id = self.tokenizer.mask_token_id

                    # Slicing: prefix includes the token at curr_idx
                    prefix = curr_input_ids[: curr_idx + 1]
                    suffix = curr_input_ids[curr_idx + 1 :]

                    # Insert mask
                    new_ids = torch.cat(
                        [
                            prefix,
                            torch.tensor([mask_token_id], device=self.device),
                            suffix,
                        ]
                    )

                    # Truncate to max length if necessary
                    if len(new_ids) > Config.MAX_LEN:
                        new_ids = new_ids[: Config.MAX_LEN]

                    # Create attention mask for new sequence
                    new_mask = torch.ones_like(new_ids)

                    # Add batch dimension for Filler
                    new_ids_batch = new_ids.unsqueeze(0)
                    new_mask_batch = new_mask.unsqueeze(0)

                    # Forward pass through Filler
                    fill_logits = self.filler(
                        new_ids_batch, new_mask_batch
                    )  # (1, L_new, V)

                    # The mask token is located at curr_idx + 1
                    mask_pos = curr_idx + 1
                    if mask_pos >= fill_logits.size(1):
                        mask_pos = fill_logits.size(1) - 1

                    # Get predicted token ID
                    pred_token_id = torch.argmax(fill_logits[0, mask_pos]).item()
                    pred_word = self.tokenizer.decode(
                        [pred_token_id], clean_up_tokenization_spaces=True
                    ).strip()

                    # --- Step 4: Reconstruct Sentence ---
                    # We use the original token IDs to reconstruct the text to preserve context
                    # curr_idx is the index of the token *before* the gap.

                    # Define valid range excluding [CLS] (0) and [SEP] (curr_len-1)
                    start_pos = 1
                    end_pos = curr_len - 1

                    # Ensure split point is valid relative to text content
                    if curr_idx < start_pos:
                        curr_idx = start_pos

                    # Decode prefix and suffix
                    prefix_ids = curr_input_ids[start_pos : curr_idx + 1]
                    suffix_ids = curr_input_ids[curr_idx + 1 : end_pos]

                    prefix_str = self.tokenizer.decode(
                        prefix_ids, clean_up_tokenization_spaces=True
                    )
                    suffix_str = self.tokenizer.decode(
                        suffix_ids, clean_up_tokenization_spaces=True
                    )

                    # Join parts
                    # Ensure proper spacing around the inserted word
                    if prefix_str and not prefix_str.endswith(" "):
                        prefix_str += " "

                    final_sent = f"{prefix_str}{pred_word} {suffix_str}".strip()

                    # Normalize whitespace
                    final_sent = " ".join(final_sent.split())

                    results.append({"id": ids[i].item(), "sentence": final_sent})

        return results

    def generate_submission(self):
        """
        Orchestrates the submission generation process.
        """
        set_seed(Config.SEED)

        # Load Resources
        self.load_models()

        # Get DataLoader
        test_loader = get_dataloaders(
            "submission", tokenizer=self.tokenizer, debug=self.debug
        )

        # Run Prediction
        predictions = self.predict(test_loader)

        # Save to CSV
        self.logger.info(
            f"Saving {len(predictions)} predictions to {Config.SUBMISSION_FILE}..."
        )

        df_sub = pd.DataFrame(predictions)
        # Ensure column order
        df_sub = df_sub[["id", "sentence"]]

        # Write CSV with specific quoting to match requirements
        # quoting=2 is csv.QUOTE_NONNUMERIC (quotes non-numeric fields)
        # This matches the format: 1,"Sentence text"
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)

        self.logger.info("Submission generation complete.")
