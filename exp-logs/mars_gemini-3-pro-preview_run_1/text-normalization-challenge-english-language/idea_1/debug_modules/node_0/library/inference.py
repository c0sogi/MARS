import torch
import pandas as pd
import numpy as np
import os
import sys
from library.config import Config
from library.model import BiLSTMTagger
from library.data_loader import get_dataloaders
from library.utils import get_logger, timer

logger = get_logger("inference")


def normalize_text(token: str, pred_class: str, kb: dict) -> str:
    """
    Normalizes a token based on its predicted class and the Knowledge Base.

    Args:
        token (str): The raw token text.
        pred_class (str): The predicted class label (e.g., "DATE", "PLAIN").
        kb (dict): The Knowledge Base dictionary mapping (token, class) -> normalized_text.

    Returns:
        str: The normalized text.
    """
    # Logic 1: Copy if class indicates no change
    if pred_class in ["PLAIN", "PUNCT"]:
        return token

    # Logic 2: Lookup in Knowledge Base
    # kb keys are tuples (token, class)
    key = (token, pred_class)
    if key in kb:
        return kb[key]

    # Logic 3: Fallback (Copy)
    # Used when the specific token/class pair was not seen in training.
    return token


class InferencePipeline:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # Load resources
        # We use get_dataloaders to ensure we have the exact same vocab and KB as training
        # This also sets up the test loader
        logger.info("Initializing Inference Pipeline...")
        # Note: get_dataloaders does not limit test set size even in debug mode
        _, _, self.test_loader, self.vocab, self.kb = get_dataloaders(
            load_cached_data=True, debug=debug
        )

        # Load raw test data for ID mapping and raw token access
        # The data loader saves grouped data to parquet during preprocessing
        test_grouped_path = os.path.join(Config.WORKING_DIR, "test_grouped.parquet")
        if not os.path.exists(test_grouped_path):
            raise FileNotFoundError(
                f"Test data cache not found at {test_grouped_path}. Run data preparation first."
            )

        self.df_test = pd.read_parquet(test_grouped_path)

        # Initialize Model
        self.model = BiLSTMTagger(
            vocab_size=len(self.vocab.token2id), num_classes=len(self.vocab.class2id)
        )
        self.model.to(self.device)

        # Load Weights
        if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            logger.info(f"Loading model weights from {Config.MODEL_CHECKPOINT_PATH}")
            state_dict = torch.load(
                Config.MODEL_CHECKPOINT_PATH, map_location=self.device
            )
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Using random weights (expect poor performance)."
            )

        self.model.eval()

    def run(self):
        """
        Executes the inference loop and saves the submission file.
        """
        results = []

        # Convert dataframe to list of dicts for fast sequential access
        # This dataframe is grouped by sentence, matching the DataLoader's sequence
        test_records = self.df_test.to_dict("records")
        current_idx = 0
        total_records = len(test_records)

        logger.info(f"Starting inference on {total_records} sentences...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Move inputs to device
                input_ids = batch["input_ids"].to(self.device)
                seq_len = batch[
                    "seq_len"
                ]  # CPU tensor is sufficient for indexing, but model handles it
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward Pass
                logits = self.model(input_ids, seq_len, attention_mask)
                # logits shape: (batch_size, max_seq_len, num_classes)

                # Get predicted class indices
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

                batch_size = input_ids.size(0)

                # Process each sentence in the batch
                for i in range(batch_size):
                    if current_idx >= total_records:
                        break

                    # Retrieve raw data for this sentence from the dataframe
                    record = test_records[current_idx]
                    current_idx += 1

                    raw_tokens = record["before"]
                    token_ids = record["id"]

                    # Retrieve predictions for this sentence
                    # pred_ids[i] is an array of shape (max_seq_len,)
                    sentence_preds = pred_ids[i]

                    # Iterate over tokens in the sentence
                    # We zip raw_tokens and token_ids to ensure alignment
                    for t_idx, (token, tid) in enumerate(zip(raw_tokens, token_ids)):

                        # Determine Class
                        # If the sentence is longer than MAX_LEN, the model output is truncated.
                        # In that case, we fall back to "PLAIN" for the tail.
                        if t_idx < Config.MAX_LEN:
                            class_id = sentence_preds[t_idx]
                            class_name = self.vocab.id2class.get(class_id, "PLAIN")
                        else:
                            class_name = "PLAIN"

                        # Normalize text
                        normalized = normalize_text(token, class_name, self.kb)

                        results.append({"id": tid, "after": normalized})

                # Log progress periodically
                if batch_idx % 100 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(self.test_loader)}")

        # Create Submission DataFrame
        logger.info("Constructing submission DataFrame...")
        df_submission = pd.DataFrame(results)

        # Save to disk
        save_path = Config.SUBMISSION_FILE_PATH
        logger.info(f"Saving submission to {save_path}...")
        df_submission.to_csv(save_path, index=False)
        logger.info(f"Submission saved successfully. Total rows: {len(df_submission)}")


def generate_submission(debug: bool = False):
    """
    Main entry point to generate submission.

    Args:
        debug (bool): If True, enables debug logging and potentially limits data loading
                      (though test set size is usually fixed in get_dataloaders).
    """
    with timer("Submission Generation", logger):
        pipeline = InferencePipeline(debug=debug)
        pipeline.run()
