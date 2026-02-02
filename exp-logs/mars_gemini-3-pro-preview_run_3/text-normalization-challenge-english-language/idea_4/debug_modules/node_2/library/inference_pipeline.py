import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, List

from library.config import Config
from library.symbolic_layer import SymbolicMemory
from library.retrieval_system import SimilarityIndex
from library.text_utils import CharTokenizer, get_context_window
from library.neural_architecture import RAGTransformer
from library.dataset_factory import RAGDataset, RAGCollator


class CascadeSolver:
    """
    Implements the Hybrid Cascade Inference Pipeline:
    1. Symbolic Memory (Head) -> 2. Heuristic Gate -> 3. Retrieval-Augmented Neural Model (Tail)
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # 1. Initialize Symbolic Memory
        print("Initializing Symbolic Memory...")
        self.symbolic_memory = SymbolicMemory()
        self.symbolic_memory.build_stats(load_cached_data=True)

        # 2. Initialize Retrieval System
        print("Initializing Retrieval System...")
        self.sim_index = SimilarityIndex()
        self.sim_index.build_index(load_cached_data=True)

        # 3. Initialize Tokenizer
        print("Initializing Tokenizer...")
        self.tokenizer = CharTokenizer()
        if os.path.exists(Config.TOKENIZER_PATH):
            self.tokenizer.load(Config.TOKENIZER_PATH)
        else:
            # Fallback or error if tokenizer strictly required
            raise FileNotFoundError(f"Tokenizer not found at {Config.TOKENIZER_PATH}")

        # 4. Initialize Neural Model
        print("Initializing Neural Model...")
        self.model = RAGTransformer(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            d_model=Config.EMBED_DIM,
            nhead=Config.N_HEADS,
            num_encoder_layers=Config.N_ENCODER_LAYERS,
            num_decoder_layers=Config.N_DECODER_LAYERS,
            dim_feedforward=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
        ).to(self.device)

        if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            print(f"Loading model weights from {Config.MODEL_CHECKPOINT_PATH}")
            state_dict = torch.load(
                Config.MODEL_CHECKPOINT_PATH, map_location=self.device
            )
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Using random weights."
            )

        self.model.eval()

    def solve(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Main inference method applying the cascade logic.
        """
        print(f"Starting inference on {len(df_test)} samples...")

        # Ensure index is standard RangeIndex for iloc alignment in context extraction
        results = df_test.copy().reset_index(drop=True)
        results["after"] = None  # Initialize predictions column

        # Ensure string types
        results["before"] = results["before"].astype(str)

        # ==========================================
        # Stage 1: Symbolic Memory (The "Head")
        # ==========================================
        print("Stage 1: Symbolic Memory Lookup...")

        # Pre-calculate context for fast lookup
        # We need prev_token and next_token respecting sentence boundaries
        results["prev_sent"] = results["sentence_id"].shift(1)
        results["next_sent"] = results["sentence_id"].shift(-1)

        results["prev_token"] = results["before"].shift(1)
        results["next_token"] = results["before"].shift(-1)

        # Apply boundaries: if sentence ID changes, context is None (SymbolicMemory handles this)
        results.loc[results["prev_sent"] != results["sentence_id"], "prev_token"] = None
        results.loc[results["next_sent"] != results["sentence_id"], "next_token"] = None

        # Fill NaNs with None for consistency
        results["prev_token"] = results["prev_token"].where(
            results["prev_token"].notna(), None
        )
        results["next_token"] = results["next_token"].where(
            results["next_token"].notna(), None
        )

        # Apply symbolic lookup row-wise
        def symbolic_lookup(row):
            return self.symbolic_memory.query(
                token=row["before"],
                prev_token=row["prev_token"],
                next_token=row["next_token"],
            )

        results["after"] = results.apply(symbolic_lookup, axis=1)

        resolved_count = results["after"].notna().sum()
        print(
            f"Symbolic Memory resolved {resolved_count} ({resolved_count/len(results):.2%}) tokens."
        )

        # ==========================================
        # Stage 2: Heuristic Router (The "Gate")
        # ==========================================
        print("Stage 2: Heuristic Router (Alpha Identity)...")

        # Mask for currently unresolved tokens
        unresolved_mask = results["after"].isna()

        # Check isalpha on unresolved tokens
        alpha_mask = results.loc[unresolved_mask, "before"].str.isalpha()

        # Apply identity
        indices_to_fill = alpha_mask[alpha_mask].index
        results.loc[indices_to_fill, "after"] = results.loc[indices_to_fill, "before"]

        resolved_after_stage_2 = results["after"].notna().sum()
        stage_2_count = resolved_after_stage_2 - resolved_count
        print(
            f"Heuristic Router resolved {stage_2_count} ({stage_2_count/len(results):.2%}) tokens."
        )

        # ==========================================
        # Stage 3: Retrieval-Augmented Transformer (The "Tail")
        # ==========================================
        print("Stage 3: Neural Inference...")

        unresolved_mask = results["after"].isna()
        df_neural = results[unresolved_mask].copy()

        if len(df_neural) > 0:
            print(f"Running neural model on {len(df_neural)} remaining tokens...")

            # 1. Context Extraction
            # We must extract context from the FULL dataframe 'results' to find neighbors
            contexts = []
            neural_indices = df_neural.index.tolist()

            for idx in neural_indices:
                # idx corresponds to the position in 'results' because we reset index at start
                ctx = get_context_window(results, idx, window_size=2)
                contexts.append(ctx)

            df_neural["context"] = contexts

            # 2. Retrieval
            print("Retrieving similar examples...")
            queries = df_neural["before"].astype(str).tolist()
            retrieval_results = self.sim_index.retrieve_batch(
                queries, k=Config.RETRIEVAL_K
            )

            ret_sources = []
            ret_targets = []

            for res_list in retrieval_results:
                if res_list:
                    ret_sources.append(res_list[0]["source"])
                    ret_targets.append(res_list[0]["target"])
                else:
                    ret_sources.append("")
                    ret_targets.append("")

            df_neural["retrieved_source"] = ret_sources
            df_neural["retrieved_target"] = ret_targets

            # 3. Create Dataset & Loader
            neural_dataset = RAGDataset(
                df_neural, self.tokenizer, Config.MAX_SEQ_LEN, mode="test"
            )
            collator = RAGCollator(self.tokenizer.pad_token_id)

            neural_loader = DataLoader(
                neural_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collator,
            )

            # 4. Inference Loop
            print("Generating predictions...")
            predictions = []

            with torch.no_grad():
                for batch in neural_loader:
                    src = batch["input_ids"].to(self.device)

                    # Greedy generation
                    generated_ids = self.model.generate(
                        src,
                        sos_token_id=self.tokenizer.sos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_len=Config.MAX_SEQ_LEN,
                    )

                    # Decode
                    gen_list = generated_ids.cpu().tolist()
                    for ids in gen_list:
                        decoded_text = self.tokenizer.decode(
                            ids, skip_special_tokens=True
                        )
                        predictions.append(decoded_text)

            # 5. Assign back to results
            results.loc[neural_indices, "after"] = predictions

        else:
            print("No tokens required neural inference.")

        # Final check for any remaining NaNs
        remaining_nans = results["after"].isna().sum()
        if remaining_nans > 0:
            print(
                f"Warning: {remaining_nans} tokens still unresolved. Filling with identity."
            )
            results["after"] = results["after"].fillna(results["before"])

        return results[["id", "after"]]


def generate_submission():
    """
    Wrapper function to execute the pipeline and save submission.
    """
    # Load Test Data
    print("Loading test data...")
    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {Config.TEST_DATA_PATH}")

    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Initialize Solver
    solver = CascadeSolver()

    # Run Prediction
    submission_df = solver.solve(df_test)

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
