import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.vocabulary import build_vocabularies
from library.data_manager import DataManager
from library.datasets import TaggerDataset, TaggerCollator
from library.models import BiLSTMTagger, TransformerSeq2Seq
from library.utils import load_checkpoint


class InferencePipeline:
    def __init__(self):
        """
        Initializes the Inference Pipeline:
        1. Sets device.
        2. Loads Vocabularies.
        3. Loads Knowledge Base.
        4. Loads and prepares Models (Tagger & Seq2Seq).
        """
        self.device = Config.DEVICE
        print(f"Initializing Inference Pipeline on {self.device}...")

        # 1. Load Vocabularies
        self.vocab_tokens, self.vocab_chars, self.vocab_classes = build_vocabularies(
            load_cached=True
        )

        # 2. Load Knowledge Base
        self.data_manager = DataManager(
            self.vocab_tokens, self.vocab_chars, self.vocab_classes
        )
        self.kb = self.data_manager.get_knowledge_base(load_cached=True)

        # 3. Initialize and Load Models
        self._load_models()

    def _load_models(self):
        """
        Loads the trained Bi-LSTM Tagger and Transformer Seq2Seq models.
        """
        # --- Load Tagger ---
        self.tagger = BiLSTMTagger(
            vocab_size=len(self.vocab_tokens),
            char_vocab_size=len(self.vocab_chars),
            num_classes=len(self.vocab_classes),
            token_pad_idx=self.vocab_tokens.stoi[Config.PAD_TOKEN],
            char_pad_idx=self.vocab_chars.stoi[Config.PAD_TOKEN],
        )
        tagger_epoch = load_checkpoint(
            Config.TAGGER_MODEL_PATH, self.tagger, device=self.device
        )
        print(f"Loaded Tagger model from epoch {tagger_epoch}")
        self.tagger.to(self.device)
        self.tagger.eval()

        # --- Load Seq2Seq ---
        self.seq2seq = TransformerSeq2Seq(
            char_vocab_size=len(self.vocab_chars),
            num_classes=len(self.vocab_classes),
            pad_idx=self.vocab_chars.stoi[Config.PAD_TOKEN],
            sos_idx=self.vocab_chars.stoi[Config.SOS_TOKEN],
            eos_idx=self.vocab_chars.stoi[Config.EOS_TOKEN],
        )
        seq2seq_epoch = load_checkpoint(
            Config.SEQ2SEQ_MODEL_PATH, self.seq2seq, device=self.device
        )
        print(f"Loaded Seq2Seq model from epoch {seq2seq_epoch}")
        self.seq2seq.to(self.device)
        self.seq2seq.eval()

    def _prepare_seq2seq_batch(self, tokens, classes):
        """
        Prepares a batch of OOV tokens for the Seq2Seq model.

        Args:
            tokens (list[str]): List of raw token strings.
            classes (list[str]): List of predicted class strings.

        Returns:
            tuple: (src_ids_tensor, class_ids_tensor)
        """
        src_indices_list = []
        class_indices_list = []

        for token, cls_name in zip(tokens, classes):
            # Numericalize chars
            chars = list(token)[: Config.MAX_SEQ_LEN]
            c_ids = self.vocab_chars.numericalize(chars)
            src_indices_list.append(torch.tensor(c_ids, dtype=torch.long))

            # Numericalize class
            cls_id = self.vocab_classes.stoi.get(cls_name, 0)
            class_indices_list.append(torch.tensor(cls_id, dtype=torch.long))

        # Pad source sequences
        src_padded = pad_sequence(
            src_indices_list,
            batch_first=True,
            padding_value=self.vocab_chars.stoi[Config.PAD_TOKEN],
        )

        class_tensor = torch.stack(class_indices_list)

        return src_padded, class_tensor

    def _decode_seq2seq_output(self, output_indices):
        """
        Converts Seq2Seq output indices back to strings.
        """
        decoded_strings = []
        eos_idx = self.vocab_chars.stoi[Config.EOS_TOKEN]

        for seq in output_indices:
            chars = []
            for idx in seq:
                idx_item = idx.item()
                if idx_item == eos_idx:
                    break
                # Skip SOS if present (though predict usually strips it) or PAD
                if idx_item not in [
                    self.vocab_chars.stoi[Config.SOS_TOKEN],
                    self.vocab_chars.stoi[Config.PAD_TOKEN],
                ]:
                    chars.append(self.vocab_chars.itos.get(idx_item, ""))
            decoded_strings.append("".join(chars))

        return decoded_strings

    def predict(self, batch_size=Config.BATCH_SIZE):
        """
        Runs the inference pipeline on the test set.

        1. Tagger predicts classes.
        2. KB Lookup.
        3. Identity mapping for PLAIN/PUNCT.
        4. Seq2Seq generation for remaining OOV.
        """
        print("Starting prediction on test set...")

        # 1. Load and Prepare Test Data
        df_test_grouped = self.data_manager.get_tagger_data(
            split="test", load_cached=True
        )

        dataset = TaggerDataset(
            df_test_grouped,
            self.vocab_tokens,
            self.vocab_chars,
            self.vocab_classes,
            split="test",
        )

        collator = TaggerCollator(self.vocab_tokens, self.vocab_chars)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=Config.NUM_WORKERS,
        )

        results = []  # List of dicts: {'id': ..., 'after': ...}

        # 2. Batch Processing
        with torch.no_grad():
            for batch in loader:
                # --- Step A: Tagger Prediction ---
                token_ids = batch["token_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                lengths = batch["lengths"]
                raw_tokens_batch = batch["raw_tokens"]  # List of lists
                ids_batch = batch["ids"]  # List of lists

                # Forward Pass
                logits = self.tagger(token_ids, char_ids, lengths)
                # Get class indices: (Batch, Seq_Len)
                pred_class_indices = torch.argmax(logits, dim=2).cpu().numpy()

                # --- Step B: Hybrid Normalization Logic ---

                # We will collect OOV items for batch Seq2Seq processing
                seq2seq_tasks = []  # (batch_idx, token_idx, raw_token, class_name)

                # Temporary storage for batch results
                batch_predictions = []  # List of lists of strings

                for b_i in range(len(raw_tokens_batch)):
                    sent_tokens = raw_tokens_batch[b_i]
                    sent_ids = ids_batch[b_i]
                    sent_preds = []

                    for t_i, token in enumerate(sent_tokens):
                        # Get predicted class name
                        cls_idx = pred_class_indices[b_i, t_i]
                        cls_name = self.vocab_classes.itos.get(cls_idx, "PLAIN")

                        # 1. KB Lookup
                        kb_key = (token, cls_name)
                        if kb_key in self.kb:
                            norm_text = self.kb[kb_key]
                            sent_preds.append(norm_text)
                        else:
                            # 2. Identity Mapping (PLAIN/PUNCT)
                            if cls_name in ["PLAIN", "PUNCT"]:
                                sent_preds.append(token)
                            else:
                                # 3. Fallback: Queue for Seq2Seq
                                # Place holder, will fill later
                                sent_preds.append(None)
                                seq2seq_tasks.append((b_i, t_i, token, cls_name))

                    batch_predictions.append(sent_preds)

                # --- Step C: Run Seq2Seq for OOV items ---
                if seq2seq_tasks:
                    oov_tokens = [x[2] for x in seq2seq_tasks]
                    oov_classes = [x[3] for x in seq2seq_tasks]

                    src_tensor, cls_tensor = self._prepare_seq2seq_batch(
                        oov_tokens, oov_classes
                    )
                    src_tensor = src_tensor.to(self.device)
                    cls_tensor = cls_tensor.to(self.device)

                    # Generate
                    generated_indices = self.seq2seq.predict(
                        src_tensor, cls_tensor, max_len=Config.MAX_SEQ_LEN
                    )
                    generated_strings = self._decode_seq2seq_output(generated_indices)

                    # Fill back into predictions
                    for idx, gen_str in enumerate(generated_strings):
                        b_i, t_i, _, _ = seq2seq_tasks[idx]
                        batch_predictions[b_i][t_i] = gen_str

                # --- Step D: Flatten and Store Results ---
                for b_i in range(len(raw_tokens_batch)):
                    sent_ids = ids_batch[b_i]
                    sent_final_preds = batch_predictions[b_i]

                    for row_id, pred in zip(sent_ids, sent_final_preds):
                        # Safety check for None (should be filled by now)
                        if pred is None:
                            pred = ""
                        results.append({"id": row_id, "after": pred})

        # 3. Save Submission
        print(f"Generated predictions for {len(results)} tokens.")
        df_submission = pd.DataFrame(results)

        # Ensure correct column order
        df_submission = df_submission[["id", "after"]]

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Done.")
