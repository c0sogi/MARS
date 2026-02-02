import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import load_checkpoint, is_semiotic
from library.hfbb_model import HFBBEngine
from library.transformer_model import Seq2SeqTransformer, Trainer, Generator
from library.data_factory import (
    build_dataloaders,
    _add_context_and_filter,
    TransformerDataset,
    CharTokenizer,
)


class HybridPredictor:
    """
    Implements the Tiered Context-Aware Hybrid System.
    Tier 1: HFBB (Memory/Lookup)
    Tier 2: Transformer (Generative for Semiotic tokens)
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.hfbb = HFBBEngine()
        self.transformer = None
        self.tokenizer = None
        self.generator = None
        set_seed(Config.SEED)

    def train_systems(
        self,
        load_cached_data=True,
        epochs=None,
        max_train_samples=None,
        force_retrain=False,
    ):
        """
        Trains or loads both Tier 1 and Tier 2 models.

        Args:
            load_cached_data (bool): Whether to use cached intermediate files.
            epochs (int, optional): Override Config.EPOCHS.
            max_train_samples (int, optional): Limit training data size for debugging.
            force_retrain (bool): Force retraining of the Transformer even if checkpoint exists.
        """
        # 1. Fit HFBB Engine (Tier 1)
        print("\n=== Initializing Tier 1: HFBB Engine ===")
        self.hfbb.fit(load_cached_data=load_cached_data)

        # 2. Train/Load Transformer (Tier 2)
        print("\n=== Initializing Tier 2: Transformer ===")

        # Override Config epochs if provided
        if epochs is not None:
            Config.EPOCHS = epochs

        # Check if we need to train
        checkpoint_exists = os.path.exists(Config.MODEL_CHECKPOINT)
        should_train = force_retrain or not checkpoint_exists

        # Load Tokenizer & Data
        # We need the tokenizer to initialize the model structure
        if should_train:
            print("Preparing DataLoaders for training...")
            if max_train_samples is not None:
                # Custom loading for debugging/subset
                df_train_raw = pd.read_csv(Config.TRAIN_CSV)
                df_train = _add_context_and_filter(
                    df_train_raw, is_train=True, load_cached_data=load_cached_data
                )
                df_train = df_train.iloc[:max_train_samples]

                df_val_raw = pd.read_csv(Config.VAL_CSV)
                df_val = _add_context_and_filter(
                    df_val_raw, is_train=True, load_cached_data=load_cached_data
                )
                if max_train_samples < len(df_val):
                    df_val = df_val.iloc[:max_train_samples]

                # Build vocab manually or load
                tokenizer = CharTokenizer()
                if load_cached_data and os.path.exists(Config.VOCAB_PATH):
                    tokenizer.load(Config.VOCAB_PATH)
                else:
                    # Quick fit on subset
                    texts = (
                        df_train["before"].astype(str).tolist()
                        + df_train["after"].astype(str).tolist()
                    )
                    tokenizer.fit_on_texts(texts)

                train_dataset = TransformerDataset(df_train, tokenizer)
                val_dataset = TransformerDataset(df_val, tokenizer)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=Config.NUM_WORKERS,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                )
                self.tokenizer = tokenizer
            else:
                # Standard loading
                train_loader, val_loader, tokenizer = build_dataloaders(
                    load_cached_data=load_cached_data
                )
                self.tokenizer = tokenizer

        else:
            # Just load tokenizer
            print(f"Loading tokenizer from {Config.VOCAB_PATH}...")
            self.tokenizer = CharTokenizer()
            self.tokenizer.load(Config.VOCAB_PATH)

        # Initialize Model
        print("Initializing Transformer Model...")
        self.transformer = Seq2SeqTransformer(
            num_encoder_layers=Config.NUM_LAYERS,
            num_decoder_layers=Config.NUM_LAYERS,
            emb_size=Config.EMBED_DIM,
            nhead=Config.NUM_HEADS,
            src_vocab_size=len(self.tokenizer),
            tgt_vocab_size=len(self.tokenizer),
            dim_feedforward=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
        ).to(self.device)

        if should_train:
            print(f"Starting training for {Config.EPOCHS} epochs...")
            trainer = Trainer(
                self.transformer,
                train_loader,
                val_loader,
                len(self.tokenizer),
                self.device,
            )
            trainer.fit()
            # Reload best
            print("Loading best checkpoint...")
            load_checkpoint(
                Config.MODEL_CHECKPOINT, self.transformer, device=self.device
            )
        else:
            print(f"Loading pre-trained model from {Config.MODEL_CHECKPOINT}...")
            load_checkpoint(
                Config.MODEL_CHECKPOINT, self.transformer, device=self.device
            )

        self.generator = Generator(self.transformer, self.tokenizer, self.device)
        print("Tier 2 Ready.")

    def generate_submission(self, load_cached_data=True):
        """
        Runs the inference cascade on the test set and generates the submission file.
        """
        print("\n=== Generating Submission ===")

        # 1. Load Test Data (Full, with context)
        test_cache_full = os.path.join(
            Config.WORKING_DIR, "test_full_processed.parquet"
        )

        if load_cached_data and os.path.exists(test_cache_full):
            print(f"Loading processed test data from {test_cache_full}...")
            df_test = pd.read_parquet(test_cache_full)
        else:
            print("Processing raw test data...")
            df_test_raw = pd.read_csv(Config.TEST_CSV)
            # is_train=False ensures NO filtering, we need all tokens
            df_test = _add_context_and_filter(
                df_test_raw, is_train=False, load_cached_data=False
            )
            df_test.to_parquet(test_cache_full)

        # 2. Iterate and Cascade
        print("Running Cascade Inference...")

        # Pre-fetch columns for speed
        ids = df_test["id_str"].tolist()
        befores = df_test["before"].fillna("").astype(str).tolist()
        prevs = df_test["prev"].fillna("").astype(str).tolist()
        nexts = df_test["next"].fillna("").astype(str).tolist()

        final_preds = {}  # idx -> prediction
        transformer_indices = []  # indices requiring Tier 2

        for idx, (curr, p, n) in enumerate(zip(befores, prevs, nexts)):
            # Step 1: Tier 1 (HFBB)
            res, level = self.hfbb.query(curr, p, n)

            # Priority: Trigram > Bigram
            if level in ["trigram", "bigram_prev", "bigram_next"]:
                final_preds[idx] = res
                continue

            # Step 2: Semiotic Check
            # If HFBB failed (or only found unigram), check if we need Tier 2
            if is_semiotic(curr):
                # Queue for Transformer
                transformer_indices.append(idx)
                continue

            # Step 3: Fallback
            # Not semiotic. If HFBB had a unigram match, use it.
            if res is not None:
                final_preds[idx] = res
            else:
                # Identity
                final_preds[idx] = curr

        # 3. Process Tier 2 (Transformer) Batches
        if transformer_indices:
            print(f"Tier 2: Processing {len(transformer_indices)} semiotic tokens...")

            # Create subset dataframe
            df_subset = df_test.iloc[transformer_indices].copy()

            # Create Dataset and Loader
            dataset = TransformerDataset(df_subset, self.tokenizer, is_test=True)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Inference
            batch_preds = []
            # We don't print progress bar as per instructions, but we process in batches
            for batch in loader:
                src = batch["input_ids"]
                decoded_texts = self.generator.predict_batch(src)
                batch_preds.extend(decoded_texts)

            # Map back to indices
            for i, original_idx in enumerate(transformer_indices):
                final_preds[original_idx] = batch_preds[i]

        # 4. Write Submission
        print("Writing submission file...")
        submission_rows = []
        for idx, id_str in enumerate(ids):
            # Retrieve prediction, default to identity (safety net)
            pred = final_preds.get(idx, befores[idx])
            submission_rows.append({"id": id_str, "after": pred})

        df_sub = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(f"Total predictions: {len(df_sub)}")
