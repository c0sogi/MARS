import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.symbolic_model import SymbolicMemory
from library.neural_model import Seq2SeqModel
from library.data_utils import Tokenizer, process_context, TextNormalizerDataset


class CascadePredictor:
    """
    Orchestrates the text normalization pipeline using a Specificity-Based Cascade:
    1. Symbolic Memory (Trigram -> Bigram -> Unigram)
    2. Heuristic Router (Identity for purely alphabetic OOV tokens)
    3. Neural Model (Factored Seq2Seq for complex/ambiguous cases)
    """

    def __init__(self, model_path=Config.MODEL_CHECKPOINT_PATH, device=Config.DEVICE):
        self.device = device
        self.tokenizer = Tokenizer()

        # 1. Initialize and Load Symbolic Memory
        print("Loading Symbolic Memory...")
        self.symbolic_memory = SymbolicMemory()
        # This will load from cache if available, or compute from train metadata
        self.symbolic_memory.fit(load_cached_data=True)

        # 2. Initialize and Load Neural Model
        print("Loading Neural Model...")
        self.neural_model = Seq2SeqModel().to(self.device)

        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.neural_model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model checkpoint not found at {model_path}. Using random initialization (expect poor performance)."
            )

        self.neural_model.eval()

    def predict(self, df):
        """
        Runs the cascade prediction on the provided DataFrame.

        Args:
            df (pd.DataFrame): Must contain 'sentence_id', 'token_id', 'before'.
                               'prev' and 'next' columns are optional; if missing,
                               context will be processed.

        Returns:
            list: A list of normalized strings corresponding to the input rows.
        """
        # Ensure context exists
        if "prev" not in df.columns or "next" not in df.columns:
            print("Processing context for inference...")
            df = process_context(df)

        # Prepare results container
        num_samples = len(df)
        predictions = [None] * num_samples

        # Indices requiring neural inference
        neural_indices = []

        print(f"Running Symbolic Lookup and Heuristics on {num_samples} samples...")

        # Iterate to apply Symbolic + Heuristic
        # Using itertuples for performance
        for row in df.itertuples():
            idx = row.Index  # This is the index in the dataframe

            # Extract context
            token = str(row.before)
            prev_token = str(row.prev)
            next_token = str(row.next)

            # 1. Symbolic Memory Lookup
            # Priority: Trigram -> Left Bigram -> Right Bigram -> Unigram
            sym_pred = self.symbolic_memory.predict(token, prev_token, next_token)
            if sym_pred is not None:
                predictions[idx] = sym_pred
                continue

            # 2. Heuristic Router
            # If the token is not in symbolic memory (OOV) but is purely alphabetic,
            # we assume it is a standard word/name and return identity.
            # This prevents the neural model from hallucinating on rare proper nouns.
            if token.isalpha():
                predictions[idx] = token
                continue

            # 3. Mark for Neural Inference
            # If neither symbolic nor heuristic resolved it, delegate to the neural model.
            neural_indices.append(idx)

        print(
            f"Symbolic/Heuristic covered {num_samples - len(neural_indices)} samples."
        )
        print(f"Neural Inference required for {len(neural_indices)} samples.")

        # 4. Neural Inference
        if neural_indices:
            self._run_neural_inference(df, neural_indices, predictions)

        return predictions

    def _run_neural_inference(self, df, indices, predictions):
        """
        Runs the neural model on a subset of the dataframe specified by indices.
        Updates the predictions list in-place.
        """
        # Create subset DataFrame for neural candidates
        # Note: We use .loc to retrieve the specific rows.
        # TextNormalizerDataset will reset the index, so we must track mapping manually.
        df_neural = df.loc[indices].copy()

        # Create Dataset and DataLoader
        dataset = TextNormalizerDataset(df_neural, self.tokenizer, mode="test")
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Pointer to track which original index corresponds to the current batch item
        current_idx_ptr = 0

        with torch.no_grad():
            for batch in dataloader:
                src_char = batch["src_char"].to(self.device)
                src_case = batch["src_case"].to(self.device)
                src_type = batch["src_type"].to(self.device)

                # Forward pass
                # When tgt is None, the model performs greedy decoding up to MAX_LEN
                outputs, _ = self.neural_model(
                    src_char, src_case, src_type, tgt=None, teacher_forcing_ratio=0.0
                )

                # outputs: [batch, max_len, vocab_size]
                # Get predicted token IDs
                preds_ids = outputs.argmax(dim=2)  # [batch, max_len]

                # Decode to strings
                batch_size = preds_ids.size(0)
                for i in range(batch_size):
                    # Detokenize (handles removal of special tokens like SOS, EOS, PAD)
                    pred_str = self.tokenizer.detokenize(preds_ids[i])

                    # Map back to the original dataframe index
                    original_idx = indices[current_idx_ptr]
                    predictions[original_idx] = pred_str

                    current_idx_ptr += 1


def generate_submission(load_cached_data=True, limit=None):
    """
    Generates the submission file for the test set.

    Args:
        load_cached_data (bool): Whether to use cached intermediate data (passed to SymbolicMemory).
        limit (int, optional): Limit the number of test samples for debugging.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 1. Load Test Data
    print(f"Loading test data from {Config.TEST_META_PATH}...")
    if not os.path.exists(Config.TEST_META_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META_PATH}")

    df_test = pd.read_parquet(Config.TEST_META_PATH)

    # 2. Process Context
    # This sorts the dataframe by sentence_id and token_id to ensure correct context
    print("Processing context...")
    df_test = process_context(df_test)

    # Apply limit for debugging if requested
    if limit is not None:
        print(f"Limiting test set to first {limit} samples.")
        df_test = df_test.iloc[:limit].copy()

    # 3. Initialize Predictor
    predictor = CascadePredictor()

    # 4. Run Prediction
    print("Starting prediction pipeline...")
    preds = predictor.predict(df_test)

    # 5. Create Submission DataFrame
    print("Creating submission file...")
    # We use the 'id' column from the processed dataframe to ensure alignment
    submission = pd.DataFrame({"id": df_test["id"], "after": preds})

    # 6. Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission)}")
