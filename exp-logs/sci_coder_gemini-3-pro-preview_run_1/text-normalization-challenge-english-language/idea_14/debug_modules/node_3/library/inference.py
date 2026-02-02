import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings

from library.config import Config
from library.utils import set_seed
from library.data_processing import (
    Vocab,
    BPETokenizer,
    KnowledgeBase,
    process_tagger_data,
    TaggerDataset,
)
from library.models import PriorInformedTagger, CharLSTMSeq2Seq

# Suppress warnings
warnings.filterwarnings("ignore")


class InferencePipeline:
    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # Placeholders for resources
        self.vocab = None
        self.bpe = None
        self.kb = None
        self.tagger = None
        self.seq2seq = None

    def load_resources(self):
        print("Loading resources...")

        # 1. Load Vocab
        self.vocab = Vocab()
        # We assume vocab files exist from training
        self.vocab.load()

        # 2. Load BPE
        self.bpe = BPETokenizer()
        self.bpe.load()

        # 3. Load Knowledge Base
        self.kb = KnowledgeBase(self.vocab)
        self.kb.load()

        # 4. Load Tagger Model
        print("Loading Tagger model...")
        self.tagger = PriorInformedTagger(
            vocab_words=self.vocab.word2id,
            vocab_classes=self.vocab.class2id,
            bpe_vocab_size=len(self.bpe),
            vocab_chars=self.vocab.char2id,
        ).to(self.device)

        if os.path.exists(Config.TAGGER_MODEL_PATH):
            state_dict = torch.load(Config.TAGGER_MODEL_PATH, map_location=self.device)
            self.tagger.load_state_dict(state_dict)
        else:
            print(f"Warning: Tagger model not found at {Config.TAGGER_MODEL_PATH}")

        self.tagger.eval()

        # 5. Load Seq2Seq Model
        print("Loading Seq2Seq model...")
        self.seq2seq = CharLSTMSeq2Seq(
            vocab_chars=self.vocab.char2id, vocab_classes=self.vocab.class2id
        ).to(self.device)

        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            state_dict = torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            self.seq2seq.load_state_dict(state_dict)
        else:
            print(f"Warning: Seq2Seq model not found at {Config.SEQ2SEQ_MODEL_PATH}")

        self.seq2seq.eval()
        print("Resources loaded successfully.")

    def decode_seq2seq(self, char_indices):
        """
        Convert batch of char indices to list of strings.
        Truncates at EOS or PAD.
        """
        # char_indices: (batch, seq_len)
        decoded_strings = []
        for seq in char_indices:
            chars = []
            for idx in seq:
                idx = idx.item()
                if idx == self.vocab.char2id[Config.EOS_TOKEN]:
                    break
                if idx == self.vocab.char2id[Config.PAD_TOKEN]:
                    continue  # Ignore pads, though usually EOS comes first
                if idx == self.vocab.char2id[Config.SOS_TOKEN]:
                    continue
                # 0 is PAD, 1 is UNK, 2 is SOS, 3 is EOS
                if idx in self.vocab.id2char:
                    chars.append(self.vocab.id2char[idx])
            decoded_strings.append("".join(chars))
        return decoded_strings

    def predict(
        self, test_csv_path=Config.TEST_CSV, batch_size=Config.TAGGER_BATCH_SIZE
    ):
        print(f"Starting prediction on {test_csv_path}...")

        # 1. Load and Sort Data
        df_test = pd.read_csv(test_csv_path, dtype=str, keep_default_na=False)
        # Ensure correct sorting to align with process_tagger_data grouping
        df_test["sentence_id"] = df_test["sentence_id"].astype(int)
        df_test["token_id"] = df_test["token_id"].astype(int)
        df_test = df_test.sort_values(["sentence_id", "token_id"])

        # Pre-group for fast retrieval during iteration
        # We need to access raw tokens by sentence_id
        sent_groups = {k: v for k, v in df_test.groupby("sentence_id")}
        unique_sent_ids = sorted(df_test["sentence_id"].unique())

        # 2. Process Features
        # Use a large max_sent_len to avoid truncation during inference
        print("Processing test features...")
        processed_data = process_tagger_data(
            df_test,
            self.vocab,
            self.bpe,
            self.kb,
            max_sent_len=512,  # Large buffer for inference
        )
        dataset = TaggerDataset(processed_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        results = []
        sent_idx_pointer = 0

        # IDs for special classes
        plain_cls_id = self.vocab.class2id.get("PLAIN", -1)
        punct_cls_id = self.vocab.class2id.get("PUNCT", -1)

        # 3. Batch Inference
        with torch.no_grad():
            for batch in dataloader:
                # --- Tagger Step ---
                word_ids = batch["word_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                regex_features = batch["regex_features"].to(self.device)
                prior_features = batch["prior_features"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Get Tagger predictions
                logits = self.tagger(
                    word_ids, bpe_ids, char_ids, regex_features, prior_features
                )
                pred_class_ids = torch.argmax(logits, dim=-1)  # (batch, seq_len)

                # Prepare for Seq2Seq batching
                seq2seq_inputs = []  # (char_ids, class_id, original_idx_tuple)
                batch_results = []  # Store partial results

                # Iterate through sentences in the batch
                current_batch_size = word_ids.size(0)

                for b_i in range(current_batch_size):
                    # Retrieve metadata for this sentence
                    sent_id = unique_sent_ids[sent_idx_pointer]
                    sent_df = sent_groups[sent_id]
                    sent_idx_pointer += 1

                    # Valid length of this sentence
                    valid_len = mask[b_i].sum().item()

                    # Ensure alignment
                    if len(sent_df) != valid_len:
                        # Fallback if lengths mismatch (e.g. strict truncation in process_tagger_data despite large limit)
                        # We use the min length to avoid index errors
                        iter_len = min(len(sent_df), valid_len)
                    else:
                        iter_len = valid_len

                    tokens = sent_df["before"].values
                    token_ids = sent_df["token_id"].values
                    submission_ids = sent_df["id"].values

                    for t_i in range(iter_len):
                        token = tokens[t_i]
                        cls_id = pred_class_ids[b_i, t_i].item()
                        cls_name = self.vocab.id2class.get(cls_id, Config.UNK_TOKEN)
                        sub_id = submission_ids[t_i]

                        # 1. KB Lookup
                        kb_norm = self.kb.get_normalization(token, cls_name)

                        if kb_norm is not None:
                            batch_results.append((sub_id, kb_norm))
                        elif cls_id == plain_cls_id or cls_id == punct_cls_id:
                            # 2. Copy Logic
                            batch_results.append((sub_id, token))
                        else:
                            # 3. Fallback (Seq2Seq)
                            # We need to run this token through Seq2Seq
                            # Store input and index to fill later
                            # char_ids for this token: batch['char_ids'][b_i, t_i] -> (max_token_len,)
                            src_chars = char_ids[b_i, t_i]
                            seq2seq_inputs.append(
                                {
                                    "src_char_ids": src_chars,
                                    "class_id": cls_id,
                                    "result_idx": len(
                                        batch_results
                                    ),  # Index in batch_results to update
                                }
                            )
                            # Append placeholder
                            batch_results.append((sub_id, None))

                # --- Seq2Seq Step (Batch Processing) ---
                if seq2seq_inputs:
                    # Collate inputs
                    s2s_src = torch.stack([x["src_char_ids"] for x in seq2seq_inputs])
                    s2s_cls = torch.tensor(
                        [x["class_id"] for x in seq2seq_inputs], device=self.device
                    )

                    # Predict
                    # output: (batch, max_len, 1) -> squeeze to (batch, max_len)
                    generated_ids = self.seq2seq.predict(s2s_src, s2s_cls).squeeze(-1)

                    # Decode
                    decoded_texts = self.decode_seq2seq(generated_ids)

                    # Fill placeholders
                    for i, text in enumerate(decoded_texts):
                        res_idx = seq2seq_inputs[i]["result_idx"]
                        sub_id, _ = batch_results[res_idx]
                        batch_results[res_idx] = (sub_id, text)

                results.extend(batch_results)

        # 4. Create Submission DataFrame
        print(f"Generated {len(results)} predictions.")
        df_submission = pd.DataFrame(results, columns=["id", "after"])

        # Post-processing: Handle quotes/spaces if necessary (basic cleanup)
        # The metric requires exact string match.

        return df_submission

    def generate_submission(self):
        self.load_resources()
        df_sub = self.predict()

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub.to_csv(
            Config.SUBMISSION_PATH, index=False, quoting=1
        )  # quoting=1 is csv.QUOTE_ALL usually, or minimal?
        # Pandas default is minimal. The sample shows quotes around text.
        # Let's stick to default pandas to_csv, it handles quoting correctly for strings containing delimiters.
        # Sample format: 0_0,"the"
        # We should ensure 'after' is treated as string.
        print("Submission saved.")
