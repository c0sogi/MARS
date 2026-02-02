import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.data_utils import (
    build_vocabularies,
    build_knowledge_base,
    load_and_group_data,
)
from library.datasets import TaggerDataset, tagger_collate_fn
from library.models import BiLSTMTagger, Seq2SeqModel


class NormalizationPipeline:
    """
    End-to-end inference pipeline for Text Normalization.
    Combines Bi-LSTM Tagger, Knowledge Base Lookup, and Seq2Seq Neural Fallback.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.vocab_words = None
        self.vocab_chars = None
        self.vocab_classes = None
        self.knowledge_base = None
        self.tagger_model = None
        self.seq2seq_model = None

        # Identity classes where before == after usually
        self.identity_classes = {"PLAIN", "PUNCT", "VERBATIM"}

        self._load_resources()
        self._load_models()

    def _load_resources(self):
        """Loads vocabularies and knowledge base."""
        print("Loading vocabularies...")
        self.vocab_words, self.vocab_chars, self.vocab_classes = build_vocabularies(
            None, load_cached_data=True
        )

        print("Loading Knowledge Base...")
        # We pass None for df_train because we expect cached data to exist for inference
        self.knowledge_base = build_knowledge_base(None, load_cached_data=True)

    def _load_models(self):
        """Initializes and loads trained model weights."""
        print("Loading models...")

        # 1. Tagger
        self.tagger_model = BiLSTMTagger(
            vocab_size=len(self.vocab_words),
            num_classes=len(self.vocab_classes),
            char_vocab_size=len(self.vocab_chars),
        )
        if os.path.exists(Config.TAGGER_MODEL_PATH):
            state_dict = torch.load(Config.TAGGER_MODEL_PATH, map_location=self.device)
            self.tagger_model.load_state_dict(state_dict)
            self.tagger_model.to(self.device)
            self.tagger_model.eval()
        else:
            print(f"Warning: Tagger model not found at {Config.TAGGER_MODEL_PATH}")

        # 2. Seq2Seq
        self.seq2seq_model = Seq2SeqModel(num_chars=len(self.vocab_chars))
        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            state_dict = torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            self.seq2seq_model.load_state_dict(state_dict)
            self.seq2seq_model.to(self.device)
            self.seq2seq_model.eval()
        else:
            print(f"Warning: Seq2Seq model not found at {Config.SEQ2SEQ_MODEL_PATH}")

    def predict(self, batch_size=Config.BATCH_SIZE):
        """
        Runs the full prediction pipeline on the test set.
        Generates submission.csv.
        """
        # 1. Load Test Data
        df_test_grouped = load_and_group_data("test", load_cached_data=True)

        dataset = TaggerDataset(
            df_test_grouped,
            self.vocab_words,
            self.vocab_chars,
            self.vocab_classes,
            split="test",
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=tagger_collate_fn,
        )

        results = []

        print(f"Starting inference on {len(dataset)} sentences...")

        # Special tokens for Seq2Seq generation
        sos_idx = self.vocab_chars.stoi[Config.SOS_TOKEN]
        eos_idx = self.vocab_chars.stoi[Config.EOS_TOKEN]

        with torch.no_grad():
            for batch in loader:
                # Unpack batch
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                row_ids = batch["row_ids"]  # List of lists

                # --- Stage 1: Tagger Classification ---
                logits = self.tagger_model(word_ids, char_ids)
                pred_classes = torch.argmax(logits, dim=-1)  # (Batch, Seq)

                # Prepare for processing
                batch_results = []

                # Collect items that need Seq2Seq fallback
                seq2seq_inputs = []
                seq2seq_indices = []  # (batch_idx_in_loop, token_idx)

                # Iterate through the batch
                # We need to reconstruct the raw tokens to check KB
                # Since we don't have raw tokens in the collated batch tensor,
                # we rely on the fact that we can't easily reverse hash the word_ids if UNK.
                # Ideally, the dataset should return raw tokens.
                # However, TaggerDataset implementation in library/datasets.py *does not* return raw tokens in the dict.
                # We must modify our approach or rely on the grouped dataframe index if strictly necessary,
                # but the dataset is sequential.
                # Actually, looking at TaggerDataset, it doesn't pass raw tokens.
                # But we have `df_test_grouped` and the loader is sequential (shuffle=False).
                # We can retrieve raw tokens from the dataframe using a global index tracker or
                # by augmenting the dataset. Since I cannot modify datasets.py, I will iterate
                # the dataframe in parallel or assume the batch aligns.
                # The loader aligns with the dataframe.

                # Calculate global index for dataframe access
                # This is slightly risky with num_workers > 0 if order isn't guaranteed,
                # but PyTorch DataLoader with shuffle=False preserves order.
                # A safer way is to just trust the order.

                # Wait, I can't easily access the raw text from the batch dictionary provided by `tagger_collate_fn`.
                # However, I need the raw text for KB lookup.
                # Strategy: The `row_ids` (e.g., "12_5") contain sentence_id and token_id.
                # I can use these to look up the raw token if I have a map, or I can infer it
                # if I had the raw data.
                # Efficient approach: Create a map of id -> raw_token before loop.
                pass

        # Optimization: To avoid mapping overhead inside the loop, let's pre-build an ID->Token map.
        # This fits in memory (1M tokens is small).
        print("Building ID to Token map for inference...")
        df_test_raw = pd.read_csv(
            Config.TEST_DATA_PATH, dtype=str, keep_default_na=False
        )
        id_to_token = dict(zip(df_test_raw["id"], df_test_raw["before"]))

        # Re-initialize results list
        final_submission_data = []

        with torch.no_grad():
            for batch in loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                batch_row_ids = batch["row_ids"]  # List of lists of strings

                # Tagger Prediction
                logits = self.tagger_model(word_ids, char_ids)
                pred_class_indices = torch.argmax(logits, dim=-1).cpu().numpy()

                # Buffers for Seq2Seq
                oov_tokens = []
                oov_coords = (
                    []
                )  # (list_index, token_index_in_list) to place result back

                # Intermediate storage for this batch
                # We will store (id, normalization)
                batch_preds = []

                # Iterate over sentences in batch
                for i, sent_row_ids in enumerate(batch_row_ids):
                    # sent_row_ids is a list of "sent_token" ids
                    # The model output is padded. We iterate up to len(sent_row_ids)
                    seq_len = len(sent_row_ids)

                    for j in range(seq_len):
                        row_id = sent_row_ids[j]
                        raw_token = id_to_token.get(row_id, "")

                        class_idx = pred_class_indices[i, j]
                        class_str = self.vocab_classes.itos.get(
                            class_idx, Config.UNK_TOKEN
                        )

                        # --- Logic Flow ---

                        # 1. Knowledge Base Lookup
                        if (raw_token, class_str) in self.knowledge_base:
                            norm_text = self.knowledge_base[(raw_token, class_str)]
                            batch_preds.append((row_id, norm_text))

                        # 2. Identity / Trivial Classes
                        elif class_str in self.identity_classes:
                            # Fallback to raw token
                            batch_preds.append((row_id, raw_token))

                        # 3. Neural Fallback (OOV + Non-Trivial Class)
                        else:
                            # We need to generate this.
                            # Store placeholder and queue for Seq2Seq
                            oov_tokens.append(raw_token)
                            oov_coords.append(
                                len(batch_preds)
                            )  # Index in batch_preds to update later
                            batch_preds.append((row_id, None))  # Placeholder

                # --- Process Seq2Seq Queue for this batch ---
                if len(oov_tokens) > 0:
                    # Convert OOV tokens to tensor
                    src_indices_list = []
                    for t in oov_tokens:
                        chars = list(t)
                        indices = self.vocab_chars.lookup_indices(
                            chars, unk_token=Config.UNK_TOKEN
                        )
                        src_indices_list.append(torch.tensor(indices, dtype=torch.long))

                    # Pad
                    src_tensor = pad_sequence(
                        src_indices_list, batch_first=True, padding_value=0
                    ).to(self.device)

                    # Generate
                    generated_indices = self.seq2seq_model.generate(
                        src_tensor,
                        sos_idx=sos_idx,
                        eos_idx=eos_idx,
                        max_len=Config.SEQ2SEQ_MAX_LEN,
                    )

                    # Decode
                    generated_indices = generated_indices.cpu().numpy()
                    for k, indices in enumerate(generated_indices):
                        # Convert to string, stop at EOS
                        chars = []
                        for idx in indices:
                            if idx == eos_idx:
                                break
                            # Skip SOS if present (generate usually doesn't return SOS as first if coded that way, but let's check)
                            if idx == sos_idx:
                                continue
                            token = self.vocab_chars.itos.get(idx, "")
                            chars.append(token)

                        norm_text = "".join(chars)

                        # Update batch_preds
                        list_idx = oov_coords[k]
                        r_id, _ = batch_preds[list_idx]
                        batch_preds[list_idx] = (r_id, norm_text)

                # Add to final list
                final_submission_data.extend(batch_preds)

        # Save Submission
        print(f"Saving submission to {Config.SUBMISSION_OUTPUT_PATH}...")
        df_sub = pd.DataFrame(final_submission_data, columns=["id", "after"])

        # Ensure exact format: id, after. Quotes are handled by pandas if needed.
        # The example shows quotes around text. Pandas default quoting usually suffices.
        # We force quoting for non-numeric to match "the" -> "the" style if pandas detects it.
        # However, standard competition CSVs are usually standard CSV.
        df_sub.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
        print("Inference complete.")
