import os
import csv
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.model import TransformerTagger
from library.data_loader import get_dataloaders


class Normalizer:
    """
    Handles the normalization of tokens based on predicted classes and a Knowledge Base.
    """

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def normalize(self, token, pred_class):
        """
        Normalizes a token given its predicted class.

        Strategy:
        1. Lookup (token, pred_class) in the deterministic Knowledge Base.
        2. If found, return the normalized text.
        3. If not found (OOV):
           - If class is PLAIN or PUNCT, return the token as is.
           - For other classes, fallback to returning the token (copy).
        """
        # 1. Deterministic Lookup
        key = (token, pred_class)
        if key in self.knowledge_base:
            return self.knowledge_base[key]

        # 2. Fallback
        # In a full system, we might use regex or num2words here for DATE, CARDINAL, etc.
        # Given the constraints, we copy the token.
        return token


def run_inference(test_loader, model, vocab_classes, normalizer, device):
    """
    Runs the model on the test_loader and generates normalized predictions.
    """
    model.eval()
    results = []

    print(f"Inference started on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)

            # Forward pass
            logits = model(input_ids)  # [batch_size, seq_len, num_classes]
            predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

            predictions = predictions.cpu().numpy()

            # Handle batch structure for string lists
            # The DataLoader collates lists of strings into a list of tuples (transposed)
            # batch["submission_ids"] is a list of length MAX_LEN, containing tuples of size BATCH_SIZE
            submission_ids_transposed = batch["submission_ids"]
            raw_tokens_transposed = batch["raw_tokens"]

            # Transpose back to [batch_size, seq_len]
            batch_submission_ids = list(zip(*submission_ids_transposed))
            batch_raw_tokens = list(zip(*raw_tokens_transposed))

            # Iterate through each sample in the batch
            for i in range(len(input_ids)):
                pred_indices = predictions[i]
                sample_sub_ids = batch_submission_ids[i]
                sample_tokens = batch_raw_tokens[i]

                # Iterate through tokens in the sequence
                for j, sub_id in enumerate(sample_sub_ids):
                    # Skip padding (empty submission id)
                    if sub_id == "":
                        continue

                    token = sample_tokens[j]
                    class_idx = pred_indices[j]
                    pred_class = vocab_classes.lookup_token(class_idx)

                    # Normalize
                    normalized_text = normalizer.normalize(token, pred_class)

                    results.append({"id": sub_id, "after": normalized_text})

    return pd.DataFrame(results)


def generate_submission(debug=False, load_cached_data=True):
    """
    Main function to generate the submission file.

    Args:
        debug (bool): If True, runs on a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
    """
    # Handle Debug Mode
    if debug:
        print("Debug mode enabled for submission generation.")
        Config.DEBUG = True
        Config.DEBUG_SIZE = 500  # Process a small number of sentences

    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data & Artifacts
    # get_dataloaders handles the loading of test data, vocabs, and building/loading the KB
    print("Loading data and artifacts...")
    _, _, test_loader, vocab_tokens, vocab_classes, knowledge_base = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Load Model
    print("Initializing model...")
    model = TransformerTagger(
        vocab_size=len(vocab_tokens),
        num_classes=len(vocab_classes),
        pad_token_id=vocab_tokens.stoi.get(Config.PAD_TOKEN, 0),
    )

    model_path = Config.MODEL_SAVE_PATH
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model file {model_path} not found. Using random weights.")

    model.to(device)

    # 3. Prepare Normalizer
    normalizer = Normalizer(knowledge_base)

    # 4. Run Inference
    print("Running inference...")
    df_submission = run_inference(test_loader, model, vocab_classes, normalizer, device)

    # 5. Save Submission
    # Ensure correct column order
    df_submission = df_submission[["id", "after"]]

    save_path = Config.SUBMISSION_PATH
    print(f"Saving submission to {save_path}...")

    # Use QUOTE_NONNUMERIC to ensure text fields containing commas are quoted,
    # matching the robust CSV format expected.
    df_submission.to_csv(save_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Submission generated with {len(df_submission)} rows.")
