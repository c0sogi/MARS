import torch
import pandas as pd
import re
import os
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.tokenizer import HybridTokenizer
from library.hfbb import HierarchicalBackoff
from library.transformer_model import TransformerTrainer, NormalizationDataset
from library.data_factory import _add_context


class HybridPredictor:
    """
    Implements the inference logic for the Text Normalization task.
    Uses a Hybrid Cascade strategy:
    1. Tier 1: Hierarchical Backoff (HFBB) for retrieval of stable/context-specific tokens.
    2. Tier 2: Transformer (Char-to-BPE) for generalization on ambiguous/semiotic tokens.
    """

    def __init__(self, load_cached_data: bool = True):
        """
        Initializes the predictor by loading the Tokenizer, HFBB, and Transformer model.

        Args:
            load_cached_data (bool): Whether to load artifacts from cache.
        """
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # 1. Load Tokenizer
        # We assume the tokenizer has been trained during the training phase.
        self.tokenizer = HybridTokenizer()
        self.tokenizer.train(load_cached_data=load_cached_data)

        # 2. Load HFBB (Tier 1)
        self.hfbb = HierarchicalBackoff()
        self.hfbb.fit(load_cached_data=load_cached_data)

        # 3. Load Transformer (Tier 2)
        # We use TransformerTrainer as a wrapper to initialize the model architecture
        self.trainer = TransformerTrainer(self.tokenizer)

        # Load model weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"Loading model weights from {Config.BEST_MODEL_PATH}...")
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.trainer.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model checkpoint not found at {Config.BEST_MODEL_PATH}. Predictions may be random."
            )

        self.trainer.model.eval()

        # Regex for Semiotic check (Digits or Latin characters)
        self.semiotic_pattern = re.compile(r"\d|[a-zA-Z]")

    def predict(self, df: pd.DataFrame):
        """
        Generates normalized predictions for the input DataFrame.

        Args:
            df (pd.DataFrame): Input data containing ['sentence_id', 'token_id', 'before'].
                               Context columns 'prev' and 'next' will be generated if missing.

        Returns:
            list: A list of predicted normalized strings corresponding to the input rows.
        """
        # Ensure context exists
        if "prev" not in df.columns or "next" not in df.columns:
            # Uses the shared logic from data_factory to ensure consistency with training
            df = _add_context(df)

        # Prepare containers
        predictions = [None] * len(df)
        nn_indices = []  # Indices of rows requiring Neural Network prediction

        # Extract lists for faster iteration
        befores = df["before"].fillna("").astype(str).tolist()
        prevs = df["prev"].fillna("<START>").astype(str).tolist()
        nexts = df["next"].fillna("<END>").astype(str).tolist()

        # ==========================================
        # Phase 1: Routing Logic (HFBB & Heuristics)
        # ==========================================
        for idx, (before, prev, nxt) in enumerate(zip(befores, prevs, nexts)):
            # Step A: HFBB Query
            pred, conf, level = self.hfbb.query(before, prev, nxt)

            # Decision Logic:
            # Accept HFBB prediction if:
            # 1. It is a context-specific match (trigram, bigram_prev, bigram_next) -> High Specificity
            # 2. It is a Unigram match with High Confidence -> High Stability
            if pred is not None and (
                level != "unigram" or conf > Config.HFBB_CONFIDENCE_THRESHOLD
            ):
                predictions[idx] = pred
            else:
                # Step B: Tier 2 Candidate Check
                # If the token contains digits or latin characters, it's "Semiotic" and needs normalization via NN
                if self.semiotic_pattern.search(before):
                    nn_indices.append(idx)
                else:
                    # Step C: Identity Fallback
                    # Plain words usually don't change (e.g., punctuation, standard Russian words)
                    predictions[idx] = before

        # ==========================================
        # Phase 2: Neural Network Inference
        # ==========================================
        if nn_indices:
            print(f"Routing {len(nn_indices)} tokens to Transformer...")

            # Create subset dataframe for NN
            df_nn = df.iloc[nn_indices].copy()

            # Create Dataset and Loader
            # is_train=False ensures we only get 'src' tensors
            nn_dataset = NormalizationDataset(df_nn, self.tokenizer, is_train=False)
            nn_loader = DataLoader(
                nn_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            nn_preds = []
            with torch.no_grad():
                for batch in nn_loader:
                    src = batch["src"].to(self.device)

                    # Predict using greedy decoding from TransformerTrainer
                    tgt_indices = self.trainer.predict(src)

                    # Decode BPE IDs to String
                    decoded_batch = [self.tokenizer.decode(t) for t in tgt_indices]
                    nn_preds.extend(decoded_batch)

            # Map predictions back to the original list
            for i, pred in enumerate(nn_preds):
                original_idx = nn_indices[i]
                predictions[original_idx] = pred

        return predictions
