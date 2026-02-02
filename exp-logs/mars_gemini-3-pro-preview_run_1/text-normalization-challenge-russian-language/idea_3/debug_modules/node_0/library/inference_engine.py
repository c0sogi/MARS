import os
import re
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

from library.config import Config
from library.utils import get_artifact_path, seed_everything
from library.data_processing import load_and_group_data, get_tokenizer
from library.symbolic_model import SymbolicLookup
from library.neural_model import CharSeq2SeqTransformer


class HybridRouter:
    """
    Implements the Hybrid Neuro-Symbolic routing logic for inference.
    Routes tokens to either the Symbolic Solver (N-grams) or the Neural Solver (Transformer)
    based on context and content.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Load Tokenizer
        # We assume the tokenizer was created during training.
        self.tokenizer = get_tokenizer(load_cached=True)

        # 2. Load Symbolic Model (N-gram Stats)
        # This will load from cache or compute if missing (and training data available)
        self.symbolic = SymbolicLookup(load_cached=True)

        # 3. Load Neural Model
        self.model = CharSeq2SeqTransformer(
            vocab_size=Config.VOCAB_SIZE,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            max_seq_len=Config.MAX_SEQ_LEN,
            pad_token_id=self.tokenizer.pad_token_id,
            sos_token_id=self.tokenizer.char_to_id.get("<sos>", 1),
            eos_token_id=self.tokenizer.char_to_id.get("<eos>", 2),
        ).to(self.device)

        # Load best model weights
        model_path = get_artifact_path("neural_normalizer_best.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(
                f"Warning: Neural model checkpoint not found at {model_path}. Neural inference will be skipped/random."
            )

        # Pre-compile regex for digit detection
        self.digit_pattern = re.compile(r"\d")

    def generate_submission(self, output_file="submission.csv"):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            output_file (str): Path to save the submission CSV.
        """
        test_df = load_and_group_data("test", load_cached_data=True)

        # Container for final results: (sentence_id, token_id, predicted_text)
        results = []

        # Queue for tokens requiring neural inference
        # Stores: (result_index, context_data_dict)
        neural_queue = []

        # Iterate over sentences
        # itertuples is faster than iterrows
        for row in test_df.itertuples(index=False):
            sent_id = row.sentence_id
            tokens = row.before
            token_ids = row.token_id
            n_tokens = len(tokens)

            for i in range(n_tokens):
                curr_w = tokens[i]
                t_id = token_ids[i]

                # Define Context
                prev_w = tokens[i - 1] if i > 0 else "<start>"
                next_w = tokens[i + 1] if i < n_tokens - 1 else "<end>"

                # --- Step 1: Specific Memory (Trigram) ---
                # If we have seen this exact sequence, use the memorized result.
                trigram_key = (prev_w, curr_w, next_w)
                if (
                    self.symbolic.stats
                    and trigram_key in self.symbolic.stats["trigram"]
                ):
                    pred = self.symbolic.stats["trigram"][trigram_key]
                    results.append((sent_id, t_id, pred))
                    continue

                # --- Step 2: Generalization (Neural) ---
                # If not memorized, check if it looks like a complex token (digits).
                if self.digit_pattern.search(curr_w):
                    # Prepare context for neural model
                    left_ctx = []
                    for k in range(1, Config.CONTEXT_WINDOW + 1):
                        if i - k >= 0:
                            left_ctx.insert(0, tokens[i - k])
                        else:
                            break

                    right_ctx = []
                    for k in range(1, Config.CONTEXT_WINDOW + 1):
                        if i + k < n_tokens:
                            right_ctx.append(tokens[i + k])
                        else:
                            break

                    neural_data = {
                        "left": " ".join(left_ctx),
                        "center": curr_w,
                        "right": " ".join(right_ctx),
                    }

                    # Add placeholder (None) to results and queue for processing
                    results.append((sent_id, t_id, None))
                    neural_queue.append((len(results) - 1, neural_data))
                    continue

                # --- Step 3: General Memory (Backoff) ---
                # If not complex, check lower-order N-grams.
                if self.symbolic.stats:
                    # Bigram
                    bigram_key = (prev_w, curr_w)
                    if bigram_key in self.symbolic.stats["bigram"]:
                        pred = self.symbolic.stats["bigram"][bigram_key]
                        results.append((sent_id, t_id, pred))
                        continue

                    # Unigram
                    unigram_key = curr_w
                    if unigram_key in self.symbolic.stats["unigram"]:
                        pred = self.symbolic.stats["unigram"][unigram_key]
                        results.append((sent_id, t_id, pred))
                        continue

                # --- Step 4: Identity ---
                # If unknown word and no digits, assume it's a standard word (e.g. name)
                results.append((sent_id, t_id, curr_w))

        # --- Process Neural Queue ---
        if neural_queue:
            self._process_neural_queue(neural_queue, results)

        # --- Save Submission ---
        self._save_results(results, output_file)

    def _process_neural_queue(self, queue, results):
        """
        Batches neural candidates, runs inference, and updates the results list.
        """
        batch_size = Config.BATCH_SIZE

        # Process in batches
        for i in range(0, len(queue), batch_size):
            batch_items = queue[i : i + batch_size]
            batch_indices = [item[0] for item in batch_items]
            batch_data = [item[1] for item in batch_items]

            # Prepare Batch Tensors
            input_tensors = []
            sep_id = self.tokenizer.sep_token_id

            for item in batch_data:
                # Encode: left <sep> center <sep> right
                left_ids = self.tokenizer.encode(item["left"])
                center_ids = self.tokenizer.encode(item["center"])
                right_ids = self.tokenizer.encode(item["right"])

                ids = left_ids + [sep_id] + center_ids + [sep_id] + right_ids
                input_tensors.append(torch.tensor(ids, dtype=torch.long))

            # Pad
            padded_input = torch.nn.utils.rnn.pad_sequence(
                input_tensors,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).to(self.device)

            # Inference
            with torch.no_grad():
                # predict returns [batch, seq_len]
                output_ids = self.model.predict(padded_input)

            # Decode and Update
            for j, out_seq in enumerate(output_ids):
                # Convert to list and decode
                pred_text = self.tokenizer.decode(out_seq.cpu().tolist())

                # Update the results list at the correct index
                original_idx = batch_indices[j]
                sent_id, token_id, _ = results[original_idx]
                results[original_idx] = (sent_id, token_id, pred_text)

    def _save_results(self, results, output_file):
        """
        Formats and saves the results to CSV.
        """
        # Construct ID column: sentence_id + "_" + token_id
        ids = [f"{r[0]}_{r[1]}" for r in results]
        preds = [r[2] for r in results]

        df = pd.DataFrame({"id": ids, "after": preds})

        # Ensure directory exists
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save
        df.to_csv(output_file, index=False)


def generate_submission_file(output_path="./submission/submission.csv"):
    """
    Wrapper function to initialize the engine and run the submission generation.
    """
    seed_everything(Config.SEED)
    engine = HybridRouter()
    engine.generate_submission(output_path)
