import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    SUBMISSION_PATH,
    TEST_DATA_PATH,
    TAGGER_EMBEDDING_DIM,
    TAGGER_CHAR_EMBEDDING_DIM,
    TAGGER_CHAR_CNN_FILTERS,
    TAGGER_CHAR_CNN_KERNEL_SIZE,
    TAGGER_HIDDEN_DIM,
    TAGGER_NUM_LAYERS,
    TAGGER_DROPOUT,
    SEQ2SEQ_EMBEDDING_DIM,
    SEQ2SEQ_HIDDEN_DIM,
    SEQ2SEQ_NUM_LAYERS,
    SEQ2SEQ_DROPOUT,
    SEQ2SEQ_MAX_LEN,
    EOS_TOKEN,
    SOS_TOKEN,
    PAD_TOKEN,
)
from library.utils import set_seed
from library.vocabulary import build_vocabularies, Vocabulary
from library.models import RegexBiLSTMTagger, CharLSTMSeq2Seq
from library.data_loader import get_test_dataloader, build_knowledge_base


class InferencePipeline:
    def __init__(self):
        self.device = DEVICE
        print(f"Initializing Inference Pipeline on {self.device}...")

        # 1. Load Vocabularies
        print("Loading vocabularies...")
        # We assume training has run and cached these
        self.vocab_words, self.vocab_chars, self.vocab_classes = build_vocabularies(
            df_train=None, load_cached_data=True
        )

        # 2. Load Knowledge Base
        print("Loading Knowledge Base...")
        kb_df = build_knowledge_base(load_cached_data=True)
        # Convert to dict for O(1) lookup: (before, class) -> after
        self.kb_dict = {}
        for before, cls, after in zip(kb_df["before"], kb_df["class"], kb_df["after"]):
            self.kb_dict[(str(before), str(cls))] = str(after)

        # 3. Load Test Data Map (ID -> Raw Text)
        # We need this because the DataLoader gives us IDs, but we need raw text for KB/Seq2Seq
        print("Loading Test Metadata for text mapping...")
        if not os.path.exists(TEST_DATA_PATH):
            raise FileNotFoundError(f"Test metadata not found at {TEST_DATA_PATH}")

        test_df = pd.read_csv(TEST_DATA_PATH, dtype=str, keep_default_na=False)
        # Map id -> before
        self.test_text_map = dict(zip(test_df["id"], test_df["before"]))

        # 4. Initialize and Load Models
        self._load_models()

    def _load_models(self):
        print("Loading models...")

        # --- Tagger ---
        self.tagger = RegexBiLSTMTagger(
            vocab_size_words=len(self.vocab_words),
            vocab_size_chars=len(self.vocab_chars),
            vocab_size_classes=len(self.vocab_classes),
        ).to(self.device)

        tagger_path = os.path.join(CHECKPOINT_DIR, "tagger_best_model.pth")
        if os.path.exists(tagger_path):
            self.tagger.load_state_dict(
                torch.load(tagger_path, map_location=self.device)
            )
            print(f"Tagger weights loaded from {tagger_path}")
        else:
            print(
                f"WARNING: Tagger checkpoint not found at {tagger_path}. Using random weights."
            )

        self.tagger.eval()

        # --- Seq2Seq ---
        sos_idx = self.vocab_chars.token2id.get(SOS_TOKEN)
        eos_idx = self.vocab_chars.token2id.get(EOS_TOKEN)

        self.seq2seq = CharLSTMSeq2Seq(
            vocab_size_chars=len(self.vocab_chars),
            vocab_size_classes=len(self.vocab_classes),
            sos_idx=sos_idx,
            eos_idx=eos_idx,
        ).to(self.device)

        seq2seq_path = os.path.join(CHECKPOINT_DIR, "seq2seq_best_model.pth")
        if os.path.exists(seq2seq_path):
            self.seq2seq.load_state_dict(
                torch.load(seq2seq_path, map_location=self.device)
            )
            print(f"Seq2Seq weights loaded from {seq2seq_path}")
        else:
            print(
                f"WARNING: Seq2Seq checkpoint not found at {seq2seq_path}. Using random weights."
            )

        self.seq2seq.eval()

    def _decode_seq2seq(self, char_indices_batch: np.ndarray) -> List[str]:
        """
        Decodes a batch of character indices from Seq2Seq output into strings.
        """
        decoded_strings = []
        eos_idx = self.vocab_chars.token2id.get(EOS_TOKEN)

        for row in char_indices_batch:
            chars = []
            for idx in row:
                if idx == eos_idx:
                    break
                # Skip padding or special tokens if they appear (SOS shouldn't be in output)
                token = self.vocab_chars.id2token.get(idx, "")
                if token not in [SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]:
                    chars.append(token)
            decoded_strings.append("".join(chars))
        return decoded_strings

    def run(self):
        print("Starting Inference...")

        # Get DataLoader
        test_loader = get_test_dataloader(
            self.vocab_words,
            self.vocab_chars,
            self.vocab_classes,
            load_cached_data=True,
        )

        results = []  # List of (id, after)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # 1. Tagger Prediction
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                regex_features = batch["regex_features"].to(self.device)

                # Forward pass
                logits = self.tagger(
                    word_ids, char_ids, regex_features
                )  # (batch, seq, num_classes)
                pred_class_indices = (
                    torch.argmax(logits, dim=2).cpu().numpy()
                )  # (batch, seq)

                # Metadata for reconstruction
                batch_token_ids = batch["token_ids"]  # List of lists
                batch_sentence_ids = batch["sentence_id"]  # List of ints

                # Buffers for Seq2Seq batching
                seq2seq_inputs_text = []
                seq2seq_class_indices = []
                seq2seq_result_indices = []  # Index in `results` list to update later

                # Iterate through batch
                batch_size_curr = len(batch_sentence_ids)

                for i in range(batch_size_curr):
                    sentence_id = batch_sentence_ids[i]
                    token_ids = batch_token_ids[i]
                    preds = pred_class_indices[i]

                    # Iterate through tokens in the sentence
                    # Note: preds might be padded, token_ids is the true length
                    seq_len = len(token_ids)

                    for t in range(seq_len):
                        token_id = token_ids[t]
                        pred_class_idx = preds[t]

                        # Construct ID
                        row_id = f"{sentence_id}_{token_id}"

                        # Get Raw Text
                        raw_text = self.test_text_map.get(row_id, "")

                        # Get Class Name
                        class_name = self.vocab_classes.id2token.get(
                            pred_class_idx, "PLAIN"
                        )

                        # --- Logic: Classify -> Retrieve -> Generate ---

                        # 1. Knowledge Base Lookup
                        kb_key = (raw_text, class_name)
                        if kb_key in self.kb_dict:
                            normalized_text = self.kb_dict[kb_key]
                            results.append((row_id, normalized_text))
                            continue

                        # 2. PLAIN/PUNCT Copy Fallback
                        # If it's a simple class and not in KB, we assume it's self-normalizing
                        if class_name in ["PLAIN", "PUNCT"]:
                            results.append((row_id, raw_text))
                            continue

                        # 3. Seq2Seq Generation for Complex/OOV
                        # We defer this to run in a sub-batch
                        seq2seq_inputs_text.append(raw_text)
                        seq2seq_class_indices.append(pred_class_idx)

                        # Placeholder in results, will be updated
                        results.append((row_id, None))
                        seq2seq_result_indices.append(len(results) - 1)

                # --- Run Seq2Seq for the accumulated OOV tokens in this batch ---
                if len(seq2seq_inputs_text) > 0:
                    # Convert text to char IDs
                    src_char_ids_list = []
                    for text in seq2seq_inputs_text:
                        ids = self.vocab_chars.lookup_indices(list(text))
                        src_char_ids_list.append(torch.tensor(ids, dtype=torch.long))

                    # Pad
                    src_char_ids = torch.nn.utils.rnn.pad_sequence(
                        src_char_ids_list, batch_first=True, padding_value=0
                    ).to(self.device)

                    class_ids_tensor = torch.tensor(
                        seq2seq_class_indices, dtype=torch.long
                    ).to(self.device)

                    # Predict
                    # Returns (batch, max_len) indices
                    generated_indices = self.seq2seq.predict(
                        src_char_ids, class_ids_tensor
                    )

                    # Decode
                    generated_strings = self._decode_seq2seq(generated_indices)

                    # Update results
                    for res_idx, gen_str in zip(
                        seq2seq_result_indices, generated_strings
                    ):
                        row_id, _ = results[res_idx]
                        results[res_idx] = (row_id, gen_str)

        # Save Submission
        self._save_submission(results)

    def _save_submission(self, results: List[Tuple[str, str]]):
        print(f"Saving submission to {SUBMISSION_PATH}...")
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        df_sub = pd.DataFrame(results, columns=["id", "after"])

        # Ensure " after" is quoted properly if it contains commas (pandas handles this)
        # The sample submission shows quotes around text. Pandas to_csv with quoting=csv.QUOTE_NONNUMERIC or similar helps,
        # but default is usually fine. The requirements say: 0_0,"the"

        # Explicitly quoting all non-numeric fields is safer for text data containing commas
        import csv

        df_sub.to_csv(SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print("Submission saved successfully.")
