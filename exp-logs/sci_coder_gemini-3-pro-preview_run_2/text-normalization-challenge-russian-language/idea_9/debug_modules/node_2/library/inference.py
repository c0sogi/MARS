import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, is_semiotic, clean_text
from library.tokenizer import CharTokenizer, TargetBPETokenizer
from library.hfbb import HFBBModel
from library.model import Seq2SeqTransformer
from library.dataset import ContextWindowDataset


class CascadePredictor:
    """
    Inference engine implementing the Confidence-Gated Curriculum Cascade.
    Routes tokens between the statistical HFBB model and the neural Transformer.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.char_tokenizer = None
        self.target_tokenizer = None
        self.hfbb_model = None
        self.transformer_model = None

    def load_resources(self):
        """
        Loads tokenizers, the statistical model, and the neural model.
        """
        print("Loading resources...")

        # 1. Load Tokenizers
        self.char_tokenizer = CharTokenizer()
        self.char_tokenizer.load(Config.CHAR_VOCAB_PATH)

        self.target_tokenizer = TargetBPETokenizer()
        self.target_tokenizer.load(Config.TARGET_TOKENIZER_MODEL)

        # 2. Load HFBB Model (Tier 1)
        # We need the model fitted on the full training set.
        # We assume the cache might exist, if not we load train data and fit.
        self.hfbb_model = HFBBModel()

        # Check if HFBB cache exists
        cache_exists = all(
            os.path.exists(p) for p in self.hfbb_model.cache_files.values()
        )

        if cache_exists:
            self.hfbb_model.fit(None, load_cached_data=True)
        else:
            print("HFBB cache not found. Fitting on training data...")
            if not os.path.exists(Config.TRAIN_DATA_PATH):
                raise FileNotFoundError(
                    f"Training data not found at {Config.TRAIN_DATA_PATH}"
                )
            df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
            self.hfbb_model.fit(df_train, load_cached_data=False)

        # 3. Load Transformer Model (Tier 2)
        print("Loading Transformer model...")
        src_vocab_size = self.char_tokenizer.get_vocab_size()
        tgt_vocab_size = self.target_tokenizer.get_vocab_size()

        self.transformer_model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_idx=self.char_tokenizer.char2id[self.char_tokenizer.pad_token],
            tgt_pad_idx=self.target_tokenizer.pad_token_id,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        ).to(self.device)

        if os.path.exists(Config.BEST_MODEL_PATH):
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.transformer_model.load_state_dict(state_dict)
            self.transformer_model.eval()
        else:
            print(
                f"Warning: Checkpoint not found at {Config.BEST_MODEL_PATH}. Using initialized weights (random)."
            )

    def _prepare_test_data(self, df_test):
        """
        Adds context columns (prev, next) to the test dataframe.
        """
        # Ensure string types
        df_test["before"] = df_test["before"].fillna("").astype(str)

        # Shift for context respecting sentence boundaries
        # Assuming df_test is sorted by sentence_id, token_id

        # Prev
        df_test["prev"] = df_test["before"].shift(1).fillna("<start>")
        mask_start = df_test["sentence_id"] != df_test["sentence_id"].shift(1)
        df_test.loc[mask_start, "prev"] = "<start>"

        # Next
        df_test["next"] = df_test["before"].shift(-1).fillna("<end>")
        mask_end = df_test["sentence_id"] != df_test["sentence_id"].shift(-1)
        df_test.loc[mask_end, "next"] = "<end>"

        return df_test

    def _greedy_decode(self, src, max_len):
        """
        Performs greedy decoding for a batch of source sequences.
        """
        batch_size = src.size(0)

        # Encode
        memory = self.transformer_model.encode(src)

        # Initialize decoder input with BOS token
        bos_id = self.target_tokenizer.bos_token_id
        eos_id = self.target_tokenizer.eos_token_id

        ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            # Decode one step
            # Note: transformer.decoder expects tgt to have shape (batch, seq_len)
            out = self.transformer_model.decode(ys, memory)

            # Get logits for the last token
            # out shape: (batch, seq_len, d_model)
            prob = self.transformer_model.generator(out[:, -1])

            # Greedy choice
            _, next_word = torch.max(prob, dim=1)

            # Update inputs
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Update finished status
            finished |= next_word == eos_id

            if finished.all():
                break

        return ys

    def generate_submission(self):
        """
        Main execution flow: Load data, predict, save submission.
        """
        set_seed(Config.SEED)
        self.load_resources()

        print("Loading test data...")
        df_test = pd.read_csv(Config.TEST_DATA_PATH)

        if Config.DEBUG:
            print(f"DEBUG: Using subset of {Config.DEBUG_SIZE} test samples.")
            df_test = df_test.iloc[: Config.DEBUG_SIZE].copy()

        # Prepare context
        df_test = self._prepare_test_data(df_test)

        # Initialize predictions with None
        predictions = [None] * len(df_test)

        # Indices requiring Tier 2 (Transformer)
        tier2_indices = []

        print("Running Tier 1 (HFBB) Inference...")

        # Vectorized-ish iteration
        # Extract arrays for speed
        prev_arr = df_test["prev"].values
        curr_arr = df_test["before"].values
        next_arr = df_test["next"].values

        # Direct map access
        trigram_map = self.hfbb_model.trigram_map
        bigram_prev_map = self.hfbb_model.bigram_prev_map
        bigram_next_map = self.hfbb_model.bigram_next_map
        unigram_map = self.hfbb_model.unigram_map

        confidence_threshold = Config.CONFIDENCE_THRESHOLD

        for idx, (p, c, n) in enumerate(zip(prev_arr, curr_arr, next_arr)):
            pred = None
            source = "none"
            conf = 0.0

            # 1. Trigram
            if (p, c, n) in trigram_map:
                pred = trigram_map[(p, c, n)]
                source = "trigram"
            # 2. Bigram Prev
            elif (p, c) in bigram_prev_map:
                pred = bigram_prev_map[(p, c)]
                source = "bigram_prev"
            # 3. Bigram Next
            elif (c, n) in bigram_next_map:
                pred = bigram_next_map[(c, n)]
                source = "bigram_next"
            # 4. Unigram
            elif c in unigram_map:
                pred, conf = unigram_map[c]
                source = "unigram"

            # Routing Logic
            route_to_tier2 = False

            if source in ["trigram", "bigram_prev", "bigram_next"]:
                # High context match: Accept
                predictions[idx] = pred
            elif source == "unigram":
                # Unigram match
                if conf > confidence_threshold:
                    # High confidence: Accept
                    predictions[idx] = pred
                else:
                    # Low confidence
                    if is_semiotic(c):
                        route_to_tier2 = True
                    else:
                        # Not semiotic, likely just a word. Accept HFBB mode.
                        predictions[idx] = pred
            else:
                # No match (OOV)
                if is_semiotic(c):
                    route_to_tier2 = True
                else:
                    # Identity fallback
                    predictions[idx] = c

            if route_to_tier2:
                tier2_indices.append(idx)

        print(
            f"Tier 1 complete. {len(tier2_indices)} tokens routed to Tier 2 (Transformer)."
        )

        # Run Tier 2
        if tier2_indices:
            print("Running Tier 2 (Transformer) Inference...")

            # Create Dataset for specific indices
            # Note: ContextWindowDataset expects numpy array of indices relative to the dataframe
            dataset = ContextWindowDataset(
                df=df_test,
                indices=np.array(tier2_indices),
                char_tokenizer=self.char_tokenizer,
                target_tokenizer=None,  # No targets in test
                mode="test",
            )

            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
            )

            # We need to map batch results back to original indices
            # The dataset returns 'id' which corresponds to the dataframe index

            with torch.no_grad():
                for batch in loader:
                    src = batch["src_ids"].to(self.device)
                    original_indices = batch["id"].numpy()

                    # Generate
                    generated_ids = self._greedy_decode(
                        src, max_len=Config.MAX_LEN_SUBWORD
                    )

                    # Decode to string
                    # generated_ids is (batch, seq_len)
                    generated_lists = generated_ids.cpu().tolist()

                    decoded_strs = [
                        self.target_tokenizer.decode(ids, remove_special_tokens=True)
                        for ids in generated_lists
                    ]

                    # Assign back
                    for i, original_idx in enumerate(original_indices):
                        predictions[original_idx] = decoded_strs[i]

        # Final cleanup: Ensure no Nones remain (shouldn't happen per logic, but safety check)
        # If any None, fallback to identity
        for i, p in enumerate(predictions):
            if p is None:
                predictions[i] = df_test.iloc[i]["before"]

        # Create Submission DataFrame
        print("Formatting submission...")
        # Construct ID: sentence_id + "_" + token_id
        # df_test has these columns
        sub_ids = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )

        submission_df = pd.DataFrame({"id": sub_ids, "after": predictions})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

        # Validation if possible (check against sample format)
        print(f"Submission shape: {submission_df.shape}")
        print(submission_df.head())
