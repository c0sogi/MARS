import os
import pandas as pd
import torch
from library.config import Config
from library.utils import set_seed, load_data
from library.tokenizers import HeterogeneousTokenizer
from library.data_factory import get_dataloaders
from library.model import CharToSubwordTransformer, train_model


class Trainer:
    """
    Orchestrator for the Heterogeneous Transformer training pipeline.
    Manages Tokenizer preparation, Data Loading, Model Initialization, and Training execution.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (str, optional): Computation device ('cuda' or 'cpu').
                                    Defaults to Config.DEVICE.
        """
        self.device = device if device else Config.DEVICE
        self.tokenizer = HeterogeneousTokenizer()

    def run(self, epochs=Config.EPOCHS, load_cached_data=True):
        """
        Executes the full training pipeline.

        Args:
            epochs (int): Number of training epochs.
            load_cached_data (bool): If True, attempts to load processed data and
                                     tokenizer artifacts from cache.

        Returns:
            nn.Module: The trained PyTorch model (loaded with best weights).
        """
        set_seed()
        print(f"Trainer: Starting run on device {self.device}")

        # ==========================================
        # 1. Tokenizer Preparation
        # ==========================================
        # We optimize memory usage by only loading the raw training data if strictly necessary.
        # The tokenizer needs raw data only if it hasn't been fitted/cached yet.
        vocab_path = Config.VOCAB_PATH
        bpe_model_path = f"{Config.BPE_MODEL_PREFIX}.model"
        artifacts_exist = os.path.exists(vocab_path) and os.path.exists(bpe_model_path)

        if load_cached_data and artifacts_exist:
            print("Trainer: Tokenizer artifacts found. Skipping raw data load.")
            # Pass an empty DataFrame; the tokenizer will load from disk and ignore this.
            train_df_for_tokenizer = pd.DataFrame()
        else:
            print(
                "Trainer: Artifacts missing or cache disabled. Loading raw training data..."
            )
            train_df_for_tokenizer = load_data("train")

        self.tokenizer.fit(train_df_for_tokenizer, load_cached_data=load_cached_data)

        # Free memory immediately
        del train_df_for_tokenizer

        # ==========================================
        # 2. Data Loading
        # ==========================================
        print("Trainer: Initializing DataLoaders...")
        # get_dataloaders handles the processing and caching of the neural dataset
        train_loader, val_loader = get_dataloaders(
            self.tokenizer,
            batch_size=Config.BATCH_SIZE,
            load_cached_data=load_cached_data,
        )

        # ==========================================
        # 3. Model Initialization
        # ==========================================
        print("Trainer: Initializing Heterogeneous Transformer Model...")
        src_vocab_size = self.tokenizer.get_source_vocab_size()
        tgt_vocab_size = self.tokenizer.get_target_vocab_size()

        # Ensure padding indices match the tokenizer's special tokens
        src_pad_idx = self.tokenizer.pad_id
        # SentencePiece typically uses ID 0 for padding, but we verify via tokenizer
        tgt_pad_idx = 0

        model = CharToSubwordTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            d_model=Config.ENC_EMB_DIM,
            nhead=Config.ENC_HEADS,
            num_encoder_layers=Config.ENC_LAYERS,
            num_decoder_layers=Config.DEC_LAYERS,
            dim_feedforward=Config.ENC_HIDDEN_DIM,
            dropout=Config.DROPOUT,
            max_len=Config.MAX_SEQ_LEN,
        )

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainer: Model initialized with {param_count} trainable parameters.")

        # ==========================================
        # 4. Training Loop
        # ==========================================
        print("Trainer: Launching training loop...")
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            self.device,
            epochs=epochs,
            patience=Config.PATIENCE,
        )

        print("Trainer: Run complete.")
        return trained_model
