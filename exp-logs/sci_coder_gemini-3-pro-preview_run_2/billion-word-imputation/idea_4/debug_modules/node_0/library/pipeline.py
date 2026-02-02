import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import setup_logger, save_submission
from library.modeling import GapLocatorModel, InFillerModel
from library.data_factory import create_dataloaders
from library.engine import train_locator, train_infiller

# Initialize module-level logger
logger = setup_logger("pipeline", os.path.join(Config.WORKING_DIR, "pipeline.log"))


class Predictor:
    """
    Inference pipeline for the Two-Stage Cascade model.
    Manages the sequential execution of the Gap Locator and In-Filler models.
    """

    def __init__(
        self, locator_path: str, infiller_path: str, device: str = Config.DEVICE
    ):
        self.device = device

        logger.info(f"Initializing Predictor on {self.device}...")

        # Load Tokenizers
        # We assume both models use the same or compatible tokenizers (Roberta family)
        self.locator_tokenizer = AutoTokenizer.from_pretrained(Config.LOCATOR_MODEL)
        self.infiller_tokenizer = AutoTokenizer.from_pretrained(Config.INFILLER_MODEL)

        # Initialize Models
        self.locator_model = GapLocatorModel(Config.LOCATOR_MODEL)
        self.infiller_model = InFillerModel(Config.INFILLER_MODEL)

        # Load Checkpoints
        self._load_checkpoint(self.locator_model, locator_path)
        self._load_checkpoint(self.infiller_model, infiller_path)

        # Move to device and set to eval mode
        self.locator_model.to(self.device)
        self.infiller_model.to(self.device)
        self.locator_model.eval()
        self.infiller_model.eval()

    def _load_checkpoint(self, model: torch.nn.Module, path: str):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model checkpoint from {path}")
        else:
            logger.warning(
                f"Checkpoint not found at {path}. Using random/pre-trained initialization."
            )

    def predict(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        Runs the full prediction pipeline on the test set.

        Args:
            test_loader (DataLoader): DataLoader providing test samples (id, original_text).

        Returns:
            pd.DataFrame: DataFrame containing 'id' and 'sentence' with predicted words inserted.
        """
        results_ids = []
        results_sentences = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", disable=None):
                original_texts = batch["original_text"]
                ids = batch["id"].tolist()

                # ---------------------------------------------------------
                # Stage 1: Gap Localization
                # ---------------------------------------------------------
                # Tokenize with offsets to map tokens back to character positions
                loc_encodings = self.locator_tokenizer(
                    original_texts,
                    return_offsets_mapping=True,
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_LEN,
                    return_tensors="pt",
                )

                loc_input_ids = loc_encodings["input_ids"].to(self.device)
                loc_attention_mask = loc_encodings["attention_mask"].to(self.device)
                offset_mappings = loc_encodings["offset_mapping"].cpu().numpy()

                # Forward Pass: Locator
                loc_outputs = self.locator_model(loc_input_ids, loc_attention_mask)
                loc_logits = loc_outputs["logits"]  # (Batch, Seq_Len)

                # Mask special tokens (CLS, SEP, PAD) to prevent invalid gap predictions
                # We set their logits to a very small number
                special_tokens_mask = torch.tensor(
                    [
                        self.locator_tokenizer.get_special_tokens_mask(
                            val, already_has_special_tokens=True
                        )
                        for val in loc_input_ids.cpu().tolist()
                    ],
                    device=self.device,
                )
                loc_logits = loc_logits.masked_fill(special_tokens_mask.bool(), -1e9)

                # Predict insertion index: The gap is AFTER this token index
                gap_indices = torch.argmax(loc_logits, dim=1).cpu().numpy()  # (Batch,)

                # ---------------------------------------------------------
                # Stage 2: In-Filling
                # ---------------------------------------------------------
                infiller_input_ids_list = []
                infiller_attention_mask_list = []
                mask_indices = []

                batch_loc_ids = loc_input_ids.cpu().tolist()

                # Construct new input sequences with <mask> inserted
                for i, seq_ids in enumerate(batch_loc_ids):
                    gap_idx = gap_indices[i]

                    # Identify sequence length (ignoring padding)
                    try:
                        sep_idx = seq_ids.index(self.locator_tokenizer.sep_token_id)
                        actual_seq = seq_ids[: sep_idx + 1]  # Include SEP
                    except ValueError:
                        actual_seq = seq_ids

                    # Determine insertion position in the token list
                    # gap_idx is the token index *after* which the word was removed.
                    # So we insert <mask> at gap_idx + 1.
                    insert_pos = gap_idx + 1

                    # Ensure we insert before the final SEP if insert_pos falls at the end
                    if insert_pos >= len(actual_seq):
                        insert_pos = len(actual_seq) - 1

                    # Construct new sequence
                    new_seq = (
                        actual_seq[:insert_pos]
                        + [self.infiller_tokenizer.mask_token_id]
                        + actual_seq[insert_pos:]
                    )

                    # Truncate to MAX_LEN if necessary
                    if len(new_seq) > Config.MAX_LEN:
                        new_seq = new_seq[: Config.MAX_LEN]
                        # Ensure SEP is preserved at the end
                        if new_seq[-1] != self.infiller_tokenizer.sep_token_id:
                            new_seq[-1] = self.infiller_tokenizer.sep_token_id

                    infiller_input_ids_list.append(new_seq)
                    # Track the index of the mask token for extracting predictions
                    mask_indices.append(
                        insert_pos
                        if insert_pos < Config.MAX_LEN
                        else Config.MAX_LEN - 2
                    )

                # Pad the new batch manually
                max_batch_len = max(len(x) for x in infiller_input_ids_list)
                padded_input_ids = []
                padded_att_masks = []

                for seq in infiller_input_ids_list:
                    pad_len = max_batch_len - len(seq)
                    padded_ids = seq + [self.infiller_tokenizer.pad_token_id] * pad_len
                    mask = [1] * len(seq) + [0] * pad_len
                    padded_input_ids.append(padded_ids)
                    padded_att_masks.append(mask)

                fill_input_ids = torch.tensor(padded_input_ids, device=self.device)
                fill_att_mask = torch.tensor(padded_att_masks, device=self.device)

                # Forward Pass: In-Filler
                fill_outputs = self.infiller_model(fill_input_ids, fill_att_mask)
                fill_logits = fill_outputs.logits  # (Batch, Seq_Len, Vocab)

                # Extract predicted tokens at mask positions
                pred_token_ids = []
                for i, mask_pos in enumerate(mask_indices):
                    # Clamp mask_pos to valid range just in case
                    if mask_pos >= fill_logits.size(1):
                        mask_pos = fill_logits.size(1) - 1

                    token_logits = fill_logits[i, mask_pos, :]
                    pred_id = torch.argmax(token_logits).item()
                    pred_token_ids.append(pred_id)

                # Decode predicted tokens to strings
                decoded_words = self.infiller_tokenizer.batch_decode(pred_token_ids)

                # ---------------------------------------------------------
                # Reconstruction
                # ---------------------------------------------------------
                for i, text in enumerate(original_texts):
                    gap_idx = gap_indices[i]
                    pred_word = decoded_words[i]

                    # Get character offset of the token identified by Locator
                    offsets = offset_mappings[i]

                    # We insert AFTER the token at gap_idx.
                    # So we need the 'end' character offset of that token.
                    if gap_idx < len(offsets):
                        start_char, end_char = offsets[gap_idx]
                        # Handle special case where token might be CLS (0,0) or similar
                        if start_char == 0 and end_char == 0 and gap_idx == 0:
                            insert_char_pos = 0
                        else:
                            insert_char_pos = end_char
                    else:
                        insert_char_pos = len(text)

                    # Insert the predicted word into the original string
                    # Note: Roberta tokens often include leading spaces (e.g., " word").
                    # The tokenizer.batch_decode preserves this, so simple concatenation works.
                    new_sentence = (
                        text[:insert_char_pos] + pred_word + text[insert_char_pos:]
                    )

                    results_ids.append(ids[i])
                    results_sentences.append(new_sentence)

        return pd.DataFrame({"id": results_ids, "sentence": results_sentences})


def run_training():
    """
    Orchestrates the training of both the Locator and In-Filler models.

    Returns:
        Tuple[str, str]: Paths to the best saved checkpoints for Locator and In-Filler.
    """
    logger.info("Initializing Data Loaders for training...")
    # Load cached data (or create cache if missing)
    train_loc, val_loc, train_fill, val_fill, _ = create_dataloaders(
        load_cached_data=True
    )

    logger.info("==================================================")
    logger.info(" Stage 1: Training Gap Locator")
    logger.info("==================================================")
    locator_path = train_locator(train_loc, val_loc)

    logger.info("==================================================")
    logger.info(" Stage 2: Training In-Filler")
    logger.info("==================================================")
    infiller_path = train_infiller(train_fill, val_fill)

    return locator_path, infiller_path


def generate_submission(
    locator_path: str = None, infiller_path: str = None, run_train: bool = True
):
    """
    Main pipeline entry point.
    1. Trains models (if run_train is True or models are missing).
    2. Loads test data.
    3. Runs inference.
    4. Saves submission CSV.

    Args:
        locator_path (str, optional): Path to pre-trained locator checkpoint.
        infiller_path (str, optional): Path to pre-trained infiller checkpoint.
        run_train (bool): Whether to execute the training loop. Defaults to True.
    """
    # 1. Model Preparation
    # Define default paths if not provided
    if locator_path is None:
        locator_path = os.path.join(Config.WORKING_DIR, "best_locator.pth")
    if infiller_path is None:
        infiller_path = os.path.join(Config.WORKING_DIR, "best_infiller.pth")

    # Determine if training is needed
    models_exist = os.path.exists(locator_path) and os.path.exists(infiller_path)

    if run_train or not models_exist:
        if not run_train and not models_exist:
            logger.warning("Models not found and run_train=False. Forcing training.")

        logger.info("Starting training phase...")
        locator_path, infiller_path = run_training()
    else:
        logger.info(f"Using existing models at {locator_path} and {infiller_path}")

    # 2. Data Loading
    logger.info("Loading Test Data...")
    # We only need the test loader here
    _, _, _, _, test_loader = create_dataloaders(load_cached_data=True)

    # 3. Inference
    logger.info("Starting Inference phase...")
    predictor = Predictor(locator_path, infiller_path)
    df_results = predictor.predict(test_loader)

    # 4. Submission
    logger.info(f"Saving submission file to {Config.SUBMISSION_PATH}...")
    save_submission(
        df_results["id"].tolist(),
        df_results["sentence"].tolist(),
        Config.SUBMISSION_PATH,
    )

    logger.info("Pipeline completed successfully.")
