import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from library.config import Config
from library.tokenizer import CharTokenizer
from library.neural_model import TransformerSeq2Seq
from library.data_manager import build_ngram_stats


class InferenceDataset(Dataset):
    """
    Dataset wrapper for batch inference with the Neural Model.
    """

    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Encode without target (inference mode)
        # We still add special tokens (SOS/EOS) as the model expects them in src
        enc = self.tokenizer.encode(
            text,
            max_len=self.max_len,
            add_special_tokens=True,
            return_tensor=True,
        )
        return enc


class HybridPredictor:
    """
    Orchestrates the hybrid text normalization pipeline:
    1. Router: Decides between Memory (N-gram) and Generalization (Neural).
    2. Memory: Hierarchical N-gram lookup.
    3. Generalization: Seq2Seq Transformer.
    """

    def __init__(self, load_cached_data=True):
        self.device = Config.DEVICE
        self.load_cached_data = load_cached_data

        # 1. Load N-gram Statistics
        # This function handles caching internally as per requirements
        print("Loading N-gram statistics...")
        self.ngram_stats = build_ngram_stats(
            Config.TRAIN_FILE, load_cached=load_cached_data
        )
        self.unigram = self.ngram_stats.get("unigram", {})
        self.bigram = self.ngram_stats.get("bigram", {})
        self.trigram = self.ngram_stats.get("trigram", {})

        # 2. Load Tokenizer
        print("Loading Tokenizer...")
        self.tokenizer = CharTokenizer()
        # Ensure vocab is built/loaded
        self.tokenizer.build_vocab(Config.TRAIN_FILE, load_cached=load_cached_data)

        # 3. Load Neural Model
        print("Loading Neural Model...")
        self.model = self._load_model()

    def _load_model(self):
        model = TransformerSeq2Seq(
            vocab_size=self.tokenizer.vocab_size,
            d_model=Config.EMBED_DIM,
            nhead=Config.N_HEADS,
            num_encoder_layers=Config.N_LAYERS,
            num_decoder_layers=Config.N_LAYERS,
            dim_feedforward=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
            pad_token_id=self.tokenizer.pad_token_id,
            sos_token_id=self.tokenizer.sos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_len=Config.MAX_INPUT_LEN,
        ).to(self.device)

        if os.path.exists(Config.MODEL_CHECKPOINT):
            print(f"Loading model weights from {Config.MODEL_CHECKPOINT}")
            state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(
                f"Warning: Model checkpoint not found at {Config.MODEL_CHECKPOINT}. Neural predictions will be random/untrained."
            )
            # We do not raise error to allow pipeline testing/fallback, but performance will be bad if not trained.

        return model

    def _prepare_context(self, df):
        """
        Adds context columns (prev, prev2, next, etc.) to the dataframe.
        """
        print("Generating context columns...")
        # Ensure sorted
        df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

        # Group by sentence to prevent context bleeding
        g = df.groupby("sentence_id")["before"]

        # Previous tokens (for N-grams and Neural)
        df["prev"] = g.shift(1).fillna("")
        df["prev2"] = g.shift(2).fillna("")

        # Next tokens (for Neural context window)
        # Assuming Context Window = 1 based on Config
        if Config.CONTEXT_WINDOW >= 1:
            df["next"] = g.shift(-1).fillna("")

        return df

    def _get_neural_candidates(self, df):
        """
        Identifies which tokens should be routed to the Neural Model.
        Logic: Contains Digit AND (Trigram Not Found).
        """
        print("Routing tokens...")

        # 1. Regex Filter
        has_digit = (
            df["before"]
            .astype(str)
            .str.contains(Config.DIGIT_REGEX, regex=True, na=False)
        )

        # 2. Trigram Memory Check
        # We prefer Memory if the exact sequence (prev2, prev, curr) was seen in training.
        # Vectorized lookup is tricky with dicts. We use list comprehension.

        # Extract tuples
        # Ensure strings
        prev2 = df["prev2"].astype(str).tolist()
        prev = df["prev"].astype(str).tolist()
        curr = df["before"].astype(str).tolist()

        trigram_keys = list(zip(prev2, prev, curr))

        # Check existence
        in_trigram = np.array([k in self.trigram for k in trigram_keys])

        # Decision: Neural if (Digit AND NOT in_trigram)
        # If it has a digit, we want to use the Neural model primarily, UNLESS we have a high-confidence memory match.
        neural_mask = has_digit & (~in_trigram)

        return neural_mask

    def _predict_ngrams(self, df_subset):
        """
        Predicts using N-gram hierarchy for a subset of data.
        Returns a Series of predictions.
        """
        # Extract columns
        prev2 = df_subset["prev2"].astype(str).values
        prev = df_subset["prev"].astype(str).values
        curr = df_subset["before"].astype(str).values

        preds = []

        # Iterate (list comprehension is fast enough for 1M rows compared to .apply)
        for p2, p1, c in zip(prev2, prev, curr):
            # 1. Trigram
            res = self.trigram.get((p2, p1, c))
            if res is not None:
                preds.append(res)
                continue

            # 2. Bigram
            res = self.bigram.get((p1, c))
            if res is not None:
                preds.append(res)
                continue

            # 3. Unigram
            res = self.unigram.get(c)
            if res is not None:
                preds.append(res)
                continue

            # 4. Identity
            preds.append(c)

        return pd.Series(preds, index=df_subset.index)

    def _predict_neural(self, df_subset):
        """
        Predicts using Neural Model for a subset of data.
        """
        if len(df_subset) == 0:
            return pd.Series([], dtype=object)

        print(f"Running Neural Inference on {len(df_subset)} tokens...")

        # 1. Construct Input Strings
        # Format: "prev curr next" (Context Window=1)
        # Note: Tokenizer splits by character, so spaces are just characters.
        inputs = []

        # Vectorized string concat
        p_list = df_subset["prev"].astype(str).tolist()
        c_list = df_subset["before"].astype(str).tolist()
        n_list = df_subset["next"].astype(str).tolist()

        # Logic from NormalizationDataset: left + [curr] + right
        for p, c, n in zip(p_list, c_list, n_list):
            parts = []
            if p:
                parts.append(p)
            parts.append(c)
            if n:
                parts.append(n)
            inputs.append(" ".join(parts))

        # 2. Dataset & DataLoader
        dataset = InferenceDataset(inputs, self.tokenizer, max_len=Config.MAX_INPUT_LEN)
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE * 2,  # Inference can handle larger batches
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device == "cuda"),
        )

        # 3. Inference Loop
        preds = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Neural Inference"):
                src = batch.to(self.device)

                # Generate
                generated_ids = self.model.generate(src, max_len=Config.MAX_TARGET_LEN)

                # Decode
                # generated_ids is (batch, seq_len)
                for i in range(generated_ids.size(0)):
                    # Remove special tokens
                    decoded = self.tokenizer.decode(
                        generated_ids[i], remove_special_tokens=True
                    )
                    preds.append(decoded)

        return pd.Series(preds, index=df_subset.index)

    def generate_submission(
        self, test_file=Config.TEST_FILE, submission_path=Config.SUBMISSION_PATH
    ):
        """
        Main entry point to generate submission file.
        """
        print(f"Reading test data from {test_file}...")
        df = pd.read_csv(test_file, dtype=str)
        df["before"] = df["before"].fillna("")
        df["sentence_id"] = df["sentence_id"].astype(int)
        df["token_id"] = df["token_id"].astype(int)

        # Prepare Context
        df = self._prepare_context(df)

        # Identify Routes
        neural_mask = self._get_neural_candidates(df)
        ngram_mask = ~neural_mask

        print(f"Split: {neural_mask.sum()} Neural vs {ngram_mask.sum()} N-gram")

        # Initialize result column
        df["after"] = ""

        # Path A: N-gram
        if ngram_mask.any():
            print("Processing N-gram candidates...")
            df.loc[ngram_mask, "after"] = self._predict_ngrams(df[ngram_mask])

        # Path B: Neural
        if neural_mask.any():
            print("Processing Neural candidates...")
            df.loc[neural_mask, "after"] = self._predict_neural(df[neural_mask])

        # Format Submission
        print("Formatting submission...")
        # id = sentence_id + "_" + token_id
        df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)

        submission = df[["id", "after"]].copy()

        # Verify length
        print(f"Submission shape: {submission.shape}")

        # Save
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission.to_csv(
            submission_path, index=False, quoting=1
        )  # quote all non-numeric
        print(f"Submission saved to {submission_path}")
