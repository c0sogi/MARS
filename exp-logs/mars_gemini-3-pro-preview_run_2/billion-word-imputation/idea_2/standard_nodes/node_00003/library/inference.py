import os
import torch
import pandas as pd
import csv
from torch.utils.data import DataLoader

from library.config import Config
from library.model import BiLSTMDualHead
from library.tokenizer import get_tokenizer
from library.dataset import MissingWordDataset


class Predictor:
    """
    Predictor class for generating predictions on the test set using the
    trained Bi-Directional LSTM Dual-Head model.
    """

    def __init__(self, config=Config, load_cached_data=True):
        """
        Initialize the Predictor.

        Args:
            config: Configuration class containing paths and hyperparameters.
            load_cached_data (bool): Whether to attempt loading cached tokenizer data.
        """
        self.config = config
        self.device = torch.device(config.DEVICE)

        # 1. Load Tokenizer
        # We use the factory function from the library to ensure consistency
        self.tokenizer = get_tokenizer(load_cached_data=load_cached_data)

        # 2. Initialize Model Architecture
        self.model = BiLSTMDualHead(
            vocab_size=self.tokenizer.get_vocab_size(),
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            lstm_layers=self.config.LSTM_LAYERS,
            dropout=self.config.DROPOUT,
        ).to(self.device)

        # 3. Load Model Weights
        self._load_weights()

    def _load_weights(self):
        """Loads the trained model weights from the configured path."""
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            state_dict = torch.load(
                self.config.MODEL_SAVE_PATH, map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {self.config.MODEL_SAVE_PATH}")
        else:
            print(
                f"Warning: No checkpoint found at {self.config.MODEL_SAVE_PATH}. "
                "Using initialized random weights."
            )

    def predict(self, output_path=None):
        """
        Runs the inference loop on the test dataset and generates the submission file.

        Args:
            output_path (str, optional): Path to save the submission CSV.
                                         Defaults to Config.SUBMISSION_PATH.
        """
        self.model.eval()

        # 1. Prepare Test Data
        # We read the test parquet directly to create a lightweight loader
        # without triggering the full training data processing pipeline.
        print(f"Loading test data from {self.config.TEST_DATA_PATH}...")
        df_test = pd.read_parquet(self.config.TEST_DATA_PATH)

        # Apply debug sampling if configured
        if self.config.DEBUG_SAMPLE_SIZE is not None:
            if len(df_test) > self.config.DEBUG_SAMPLE_SIZE:
                print(f"Sampling {self.config.DEBUG_SAMPLE_SIZE} test samples...")
                df_test = df_test.iloc[: self.config.DEBUG_SAMPLE_SIZE]

        # Create Dataset and DataLoader
        test_dataset = MissingWordDataset(
            df_test, self.tokenizer, self.config.MAX_SEQ_LEN, mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=(self.config.DEVICE == "cuda"),
            drop_last=False,
        )

        results = []
        print(f"Starting inference on {len(test_dataset)} samples...")

        with torch.no_grad():
            for batch in test_loader:
                ids = batch["id"].numpy()
                input_ids = batch["input_ids"].to(self.device)

                # Forward Pass
                # loc_logits: (batch, seq_len, 1)
                # word_logits: (batch, seq_len, vocab_size)
                loc_logits, word_logits = self.model(input_ids)

                # --- Decode Predictions ---

                # 1. Identify Gap Location
                # Apply sigmoid to get probabilities
                loc_probs = torch.sigmoid(loc_logits.squeeze(-1))

                # Mask padding tokens to ensure we don't predict a gap in the padding
                padding_mask = input_ids != self.tokenizer.pad_token_id
                loc_probs = loc_probs * padding_mask.float()

                # Find the index with the maximum probability for each sample
                best_loc_indices = torch.argmax(loc_probs, dim=1)  # Shape: (batch,)

                # 2. Identify Missing Word
                # Select the word logits corresponding to the predicted gap location
                batch_indices = torch.arange(input_ids.size(0), device=self.device)
                best_word_logits = word_logits[batch_indices, best_loc_indices, :]

                # Find the word index with the maximum probability
                pred_word_ids = torch.argmax(best_word_logits, dim=1)  # Shape: (batch,)

                # --- Reconstruct Sentences ---

                # Move tensors to CPU for string processing
                input_ids_np = input_ids.cpu().numpy()
                best_loc_indices_np = best_loc_indices.cpu().numpy()
                pred_word_ids_np = pred_word_ids.cpu().numpy()

                for i in range(len(ids)):
                    sample_id = ids[i]
                    curr_input_ids = input_ids_np[i]
                    gap_idx = best_loc_indices_np[i]
                    pred_word_id = pred_word_ids_np[i]

                    # Decode the predicted word
                    pred_word = self.tokenizer.idx2word.get(
                        pred_word_id, self.tokenizer.unk_token
                    )

                    # Decode the input sentence (excluding padding)
                    tokens = []
                    for tid in curr_input_ids:
                        if tid == self.tokenizer.pad_token_id:
                            break
                        tokens.append(
                            self.tokenizer.idx2word.get(tid, self.tokenizer.unk_token)
                        )

                    # Insert the predicted word
                    # gap_idx represents the token *before* the missing word.
                    # Therefore, we insert at gap_idx + 1.
                    insert_pos = min(gap_idx + 1, len(tokens))
                    tokens.insert(insert_pos, pred_word)

                    final_sentence = " ".join(tokens)

                    results.append({"id": sample_id, "sentence": final_sentence})

        # 3. Save Submission
        save_path = output_path if output_path else self.config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df_sub = pd.DataFrame(results)

        # Save to CSV with appropriate quoting
        df_sub.to_csv(
            save_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        print(f"Submission saved successfully to {save_path}")
