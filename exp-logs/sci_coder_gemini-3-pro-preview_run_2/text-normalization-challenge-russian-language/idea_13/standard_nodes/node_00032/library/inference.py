import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.utils import get_device, is_semiotic, load_metadata, save_submission
from library.hfbb_model import HFBBStats
from library.tokenizer import HybridTokenizer
from library.transformer_model import Seq2SeqTransformer


class InferenceDataset(Dataset):
    """
    Simple Dataset wrapper for inference batches.
    """

    def __init__(self, enc_ids):
        self.enc_ids = enc_ids

    def __len__(self):
        return len(self.enc_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.enc_ids[idx], dtype=torch.long)


def collate_inference(batch, pad_id):
    """
    Pads inference batches.
    """
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)


class HybridPredictor:
    """
    Implements the Robust Density-Maximized Hybrid Cascade inference pipeline.
    Routes tokens between Statistical Memory (Tier 1) and Neural Network (Tier 2).
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()

        print("=== Initializing HybridPredictor ===")

        # 1. Load Tokenizer
        print("Loading Tokenizer...")
        self.tokenizer = HybridTokenizer(config)
        self.tokenizer.fit(load_cached_data=True)

        # 2. Load HFBB Stats (Tier 1)
        print("Loading HFBB Stats...")
        self.hfbb = HFBBStats(config)
        self.hfbb.fit(load_cached_data=True)

        # 3. Load Transformer (Tier 2)
        print("Loading Transformer Model...")
        self.src_vocab_size = self.tokenizer.char_vocab_size_actual
        self.tgt_vocab_size = self.tokenizer.bpe_vocab_size
        self.src_pad_idx = self.tokenizer.char2id[self.tokenizer.PAD_TOKEN]
        self.tgt_pad_idx = self.tokenizer.bpe_pad_id

        self.model = Seq2SeqTransformer(
            config,
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.src_pad_idx,
            self.tgt_pad_idx,
        ).to(self.device)

        # Load Weights
        checkpoint_path = os.path.join(
            config.base_working_dir, "checkpoints", "transformer_best.pth"
        )
        if os.path.exists(checkpoint_path):
            print(f"Loading model weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights."
            )

        self.model.eval()

    def generate_submission(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        # 1. Load Test Data
        print("Loading Test Data...")
        df = load_metadata("test")

        # Ensure correct order for context generation
        if "token_id" in df.columns:
            df["token_id_int"] = df["token_id"].astype(int)
            df.sort_values(["sentence_id", "token_id_int"], inplace=True)

        # 2. Generate Context
        print("Generating Context...")
        df["prev_token"] = df["before"].shift(1).fillna("<START>")
        df["next_token"] = df["before"].shift(-1).fillna("<END>")
        df["prev_sent"] = df["sentence_id"].shift(1)
        df["next_sent"] = df["sentence_id"].shift(-1)

        # Apply sentence boundaries
        mask_start = df["prev_sent"] != df["sentence_id"]
        df.loc[mask_start, "prev_token"] = "<START>"

        mask_end = df["next_sent"] != df["sentence_id"]
        df.loc[mask_end, "next_token"] = "<END>"

        # 3. Routing Logic
        print("Running Hybrid Routing Logic...")
        predictions = [None] * len(df)
        tier2_indices = []

        # Extract records for fast iteration
        records = df[["before", "prev_token", "next_token"]].to_dict("records")

        for idx, row in enumerate(records):
            token = str(row["before"])
            prev_t = str(row["prev_token"])
            next_t = str(row["next_token"])

            # Step 1 & 2: HFBB Query
            # Returns (prediction, confidence)
            # Confidence is 1.0 for Trigram/Bigram matches
            pred, conf = self.hfbb.query(token, prev_t, next_t)

            if pred is not None:
                if conf > self.config.hfbb_confidence_threshold:
                    # High confidence match (Context or Stable Unigram)
                    predictions[idx] = pred
                else:
                    # Low confidence Unigram -> Route to Tier 2
                    tier2_indices.append(idx)
            else:
                # Step 3: OOV Fallback
                if is_semiotic(token):
                    # Semiotic OOV -> Route to Tier 2
                    tier2_indices.append(idx)
                else:
                    # Non-semiotic OOV -> Identity Fallback
                    predictions[idx] = token

        # 4. Tier 2 Inference (Neural Network)
        if tier2_indices:
            print(f"Routing {len(tier2_indices)} tokens to Tier 2 (Neural Network)...")
            tier2_preds = self._run_transformer_inference(df.iloc[tier2_indices])

            # Merge predictions
            for idx, pred in zip(tier2_indices, tier2_preds):
                predictions[idx] = pred

        # 5. Format and Save
        print("Formatting Submission...")
        df["after"] = predictions

        # Construct required 'id' column: sentence_id + "_" + token_id
        df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)

        submission_df = df[["id", "after"]]
        save_submission(submission_df, self.config.submission_path)
        print(f"Submission saved successfully to {self.config.submission_path}")

    def _run_transformer_inference(self, df_subset):
        """
        Runs the Transformer model on a subset of the dataframe.
        """
        # Tokenize Inputs (Character Level with Context)
        enc_ids_list = []
        SEP_ID = self.tokenizer.SEP_ID
        START_ID = self.tokenizer.START_ID
        END_ID = self.tokenizer.END_ID

        # Using zip for speed
        iterator = zip(
            df_subset["prev_token"].astype(str),
            df_subset["before"].astype(str),
            df_subset["next_token"].astype(str),
        )

        for prev, curr, next_tok in iterator:
            p_ids = self.tokenizer.encode_char(prev)
            c_ids = self.tokenizer.encode_char(curr)
            n_ids = self.tokenizer.encode_char(next_tok)

            # Truncation Logic (Must match training)
            overhead = 4
            available = self.config.max_enc_len - overhead - len(c_ids)

            if available < 0:
                c_ids = c_ids[: self.config.max_enc_len - overhead]
                p_ids = []
                n_ids = []
            else:
                half = available // 2
                if len(p_ids) > half:
                    p_ids = p_ids[-half:]
                remaining = available - len(p_ids)
                if len(n_ids) > remaining:
                    n_ids = n_ids[:remaining]

            full_enc = p_ids + [SEP_ID, START_ID] + c_ids + [END_ID, SEP_ID] + n_ids
            enc_ids_list.append(full_enc)

        # Create DataLoader
        dataset = InferenceDataset(enc_ids_list)
        collate = lambda b: collate_inference(b, self.src_pad_idx)

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate,
        )

        results = []

        # Inference Loop
        with torch.no_grad():
            for batch in loader:
                src = batch.to(self.device)  # [Batch, Seq]

                # Run Greedy Decoding
                generated_ids = self._greedy_decode(src)

                # Decode BPE to String
                for i in range(generated_ids.size(0)):
                    # Convert tensor to list
                    seq = generated_ids[i].tolist()

                    # Extract valid tokens (remove BOS, stop at EOS)
                    valid_seq = []
                    for token_id in seq:
                        if token_id == self.tokenizer.bpe_bos_id:
                            continue
                        if token_id == self.tokenizer.bpe_eos_id:
                            break
                        valid_seq.append(token_id)

                    decoded_str = self.tokenizer.decode_bpe(valid_seq)
                    results.append(decoded_str)

        return results

    def _greedy_decode(self, src):
        """
        Performs greedy decoding for a batch of inputs.
        """
        batch_size = src.size(0)
        max_len = self.config.max_dec_len

        # Encode source
        memory = self.model.encode(src)  # [Seq, Batch, Dim]

        # Initialize decoder input with <BOS>
        ys = torch.full(
            (batch_size, 1),
            self.tokenizer.bpe_bos_id,
            dtype=torch.long,
            device=self.device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            # Decode step
            # decode returns [Seq, Batch, Dim]
            out = self.model.decode(ys, memory)

            # Get logits for the last token in the sequence
            # out[-1] is [Batch, Dim]
            logits = self.model.generator(out[-1])

            # Greedy selection
            next_word = torch.argmax(logits, dim=1)  # [Batch]

            # Update finished status
            finished |= next_word == self.tokenizer.bpe_eos_id

            # Append to sequence
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Early exit if all sequences are finished
            if finished.all():
                break

        return ys
