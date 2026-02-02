import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_metadata, save_submission, ensure_dir
from library.symbolic_model import NgramLookup
from library.neural_network import NeuralNormalizer
from library.data_processing import CharTokenizer, TextNormalizationDataset, collate_fn


class HybridPredictor:
    """
    Implements the hybrid routing logic for text normalization inference.
    Combines a symbolic N-gram lookup for deterministic cases and a
    Target-Aware Global-Context Transformer for complex cases.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 1. Load Tokenizer
        # We expect the tokenizer to exist from the training phase.
        # If not found, we cannot proceed with neural inference effectively.
        self.tokenizer = CharTokenizer(config)
        if os.path.exists(config.tokenizer_path):
            self.tokenizer.load(config.tokenizer_path)
        else:
            print(
                f"Warning: Tokenizer not found at {config.tokenizer_path}. Neural inference may fail."
            )

        # 2. Initialize Solvers
        # Symbolic Solver
        self.ngram_lookup = NgramLookup(config)

        # Neural Solver
        self.neural_model = NeuralNormalizer(config, self.tokenizer)

    def predict(self, load_cached_data=True):
        """
        Main inference pipeline.

        Args:
            load_cached_data (bool): Used to load N-gram stats from cache if available.
        """
        print("Initializing Hybrid Predictor...")

        # --- Step 1: Prepare Solvers ---

        # Load N-gram statistics
        self.ngram_lookup.fit(load_cached_data=load_cached_data)

        # Load Neural Model weights
        if os.path.exists(self.config.model_best_path):
            self.neural_model.load(self.config.model_best_path)
        else:
            print(
                f"Warning: Neural model weights not found at {self.config.model_best_path}. Using random weights."
            )

        # --- Step 2: Load and Group Test Data ---
        print("Loading test data...")
        df_test = load_metadata(self.config.test_file)

        # Ensure data is sorted by sentence and token id
        if "token_id" in df_test.columns:
            df_test = df_test.sort_values(["sentence_id", "token_id"])

        # Group by sentence_id to get full context
        # We need both the token text and the token_id to reconstruct the submission ID
        print("Grouping data by sentence for context extraction...")

        # Aggregating into lists: sentence_id -> (list_of_token_ids, list_of_tokens)
        # Using a dictionary comprehension for speed
        grouped = df_test.groupby("sentence_id")[["token_id", "before"]].agg(list)

        # Prepare containers for results
        final_predictions = {}  # Map: id_string -> predicted_text
        neural_candidates = []  # List of dicts for neural processing

        print("Executing Pass 1: Symbolic Lookup and Routing...")

        # Pre-compile regex for digit detection (heuristic for "hard" tokens)
        digit_pattern = re.compile(r"\d")

        # Iterate over sentences
        # grouped.index is sentence_id
        # grouped['token_id'] is list of ids, grouped['before'] is list of tokens
        for sentence_id, row in grouped.iterrows():
            token_ids = row["token_id"]
            tokens = row["before"]
            seq_len = len(tokens)

            # Iterate tokens within sentence
            for i in range(seq_len):
                curr_token = str(tokens[i])
                t_id = token_ids[i]
                submission_id = f"{sentence_id}_{t_id}"

                # Define Context
                prev_token = str(tokens[i - 1]) if i > 0 else "<s>"
                next_token = str(tokens[i + 1]) if i < seq_len - 1 else "</s>"

                # 1. Symbolic Lookup
                symbolic_pred = self.ngram_lookup.get_normalization(
                    curr_token, prev_token, next_token
                )

                if symbolic_pred is not None:
                    # Found in lookup
                    final_predictions[submission_id] = symbolic_pred
                else:
                    # Not found in lookup. Check complexity.
                    if digit_pattern.search(curr_token):
                        # Contains digits -> Route to Neural Model
                        # Cite solution_lesson_node_00013: Use Local Window Context
                        # Format: left_context <sep> target <sep> right_context

                        window_size = self.config.context_window
                        start_idx = max(0, i - window_size)
                        end_idx = min(seq_len, i + window_size + 1)

                        left_ctx = tokens[start_idx:i]
                        right_ctx = tokens[i + 1 : end_idx]

                        parts = []
                        if left_ctx:
                            parts.append(" ".join(map(str, left_ctx)))
                        parts.append(self.config.sep_token)
                        parts.append(curr_token)
                        parts.append(self.config.sep_token)
                        if right_ctx:
                            parts.append(" ".join(map(str, right_ctx)))

                        input_text = " ".join(parts)

                        neural_candidates.append(
                            {
                                "id": submission_id,
                                "input_text": input_text,
                                "target_text": "",  # Placeholder for dataset compatibility
                            }
                        )
                    else:
                        # No digits -> "Easy" unknown token -> Identity Fallback
                        # (Assume it's a rare word or punctuation not in lookup but doesn't need normalization)
                        final_predictions[submission_id] = curr_token

        # --- Step 3: Neural Inference (Pass 2) ---

        if neural_candidates:
            print(
                f"Executing Pass 2: Neural Inference on {len(neural_candidates)} complex tokens..."
            )

            # Create DataFrame for candidates
            df_neural = pd.DataFrame(neural_candidates)

            # Create Dataset and Loader
            # We use 'test' mode so target_text is ignored/empty
            neural_dataset = TextNormalizationDataset(
                df_neural, self.tokenizer, self.config, mode="test"
            )

            neural_loader = DataLoader(
                neural_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )

            # Run Inference
            neural_preds = self.neural_model.predict(neural_loader)

            # Merge results
            final_predictions.update(neural_preds)
        else:
            print("No neural candidates found.")

        # --- Step 4: Save Submission ---
        print("Formatting and saving submission...")

        # Convert dictionary to DataFrame
        # The submission format requires 'id' and 'after' columns
        submission_ids = sorted(final_predictions.keys())
        submission_values = [final_predictions[k] for k in submission_ids]

        df_submission = pd.DataFrame({"id": submission_ids, "after": submission_values})

        # Save
        save_submission(df_submission, self.config.submission_path)
        print(f"Submission saved to {self.config.submission_path}")
        print(f"Total predictions: {len(df_submission)}")
