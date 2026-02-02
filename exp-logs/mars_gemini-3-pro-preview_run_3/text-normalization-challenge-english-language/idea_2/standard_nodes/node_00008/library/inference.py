import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.data import process_data, build_tokenizer, NormalizationDataset, collate_fn
from library.symbolic_model import SymbolicMemory
from library.neural_model import NeuralSolver


class HybridNormalizer:
    """
    Hybrid Neuro-Symbolic Normalizer.
    Combines high-precision N-gram lookup tables with a generalization-capable
    Seq2Seq neural network to normalize text.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the hybrid model components.

        Args:
            load_cached_data (bool): Whether to load pre-computed resources from cache.
        """
        self.device = Config.DEVICE
        set_seed()

        print("Initializing HybridNormalizer...")

        # 1. Load Tokenizer
        # We need the tokenizer to decode neural model outputs
        print("Loading Tokenizer...")
        self.tokenizer = build_tokenizer(load_cached_data=load_cached_data)

        # 2. Initialize and Load Symbolic Memory (The "Head")
        # This component handles frequent and context-dependent patterns via memorization
        print("Loading Symbolic Memory...")
        self.symbolic_mem = SymbolicMemory()
        self.symbolic_mem.fit(load_cached_data=load_cached_data)

        # 3. Initialize Neural Solver (The "Tail")
        # This component handles OOV and complex patterns (digits, symbols)
        print("Loading Neural Solver...")
        self.neural_solver = NeuralSolver(self.tokenizer)

        # Load trained weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading neural model weights from {Config.MODEL_SAVE_PATH}")
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.neural_solver.model.load_state_dict(state_dict)
        else:
            print(
                "Warning: No neural model checkpoint found! Neural predictions will be random."
            )

        self.neural_solver.model.eval()

    def predict_dataset(self, load_cached_data=True, debug_sample_size=None):
        """
        Runs the inference pipeline on the test dataset and generates the submission file.

        The pipeline follows a strict specificity-first logic:
        1. Symbolic Lookup (Trigram -> Bigram -> Unigram)
        2. Heuristic Filter (Identity for purely alphabetic tokens)
        3. Neural Inference (Seq2Seq for remaining complex tokens)

        Args:
            load_cached_data (bool): Whether to use cached processed data.
            debug_sample_size (int, optional): Number of samples to process for debugging.
        """
        # 1. Load and Process Test Data
        # process_data handles caching and generates 'prev' and 'next' context columns
        print("Loading and processing test data...")
        df_test = process_data("test", load_cached_data=load_cached_data)

        if debug_sample_size:
            print(f"DEBUG: Subsampling test set to {debug_sample_size} rows.")
            df_test = df_test.iloc[:debug_sample_size].copy()

        total_samples = len(df_test)
        final_predictions = [None] * total_samples

        # Lists to collect samples for batch neural inference
        neural_indices = []
        neural_rows = []

        print("Running Symbolic Lookup and Heuristic Filter...")

        # Convert columns to lists for faster iteration
        prevs = df_test["prev"].fillna("").astype(str).tolist()
        currs = df_test["before"].fillna("").astype(str).tolist()
        nexts = df_test["next"].fillna("").astype(str).tolist()

        for idx in range(total_samples):
            p, c, n = prevs[idx], currs[idx], nexts[idx]

            # Step 1: Symbolic Lookup
            # Check hierarchical N-grams
            symbolic_res = self.symbolic_mem.query(p, c, n)
            if symbolic_res is not None:
                final_predictions[idx] = symbolic_res
                continue

            # Step 2: Heuristic Filter
            # If token is purely alphabetic (e.g., "dog", "Rocky"), it likely doesn't need normalization.
            # This filters out rare proper nouns that might confuse the neural model.
            if c.isalpha():
                final_predictions[idx] = c
                continue

            # Step 3: Neural Fallback Candidate
            # If we reach here, the token is OOV and "complex" (contains digits/symbols).
            # We queue it for the neural model.
            neural_indices.append(idx)
            neural_rows.append({"before": c, "prev": p, "next": n})

        # Run Neural Inference on queued samples
        if neural_indices:
            print(f"Running Neural Inference on {len(neural_indices)} samples...")

            # Create a temporary DataFrame for the neural dataset
            df_neural = pd.DataFrame(neural_rows)

            # Create Dataset and DataLoader
            neural_dataset = NormalizationDataset(df_neural, self.tokenizer)
            neural_loader = DataLoader(
                neural_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=os.cpu_count() or 4,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            # Generate predictions
            neural_preds = self.neural_solver.predict(neural_loader)

            # Sanity check
            if len(neural_preds) != len(neural_indices):
                raise ValueError(
                    f"Mismatch in neural predictions: {len(neural_preds)} preds vs {len(neural_indices)} inputs"
                )

            # Map predictions back to the main results list
            for i, pred in zip(neural_indices, neural_preds):
                final_predictions[i] = pred

        # Final pass: Fill any remaining Nones with Identity (safety fallback)
        filled_count = 0
        for i in range(total_samples):
            if final_predictions[i] is None:
                final_predictions[i] = currs[i]
                filled_count += 1

        if filled_count > 0:
            print(
                f"Warning: {filled_count} samples were not handled by any logic. Defaulted to identity."
            )

        # 2. Generate Submission File
        print("Formatting submission...")
        submission = pd.DataFrame({"id": df_test["id"], "after": final_predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
