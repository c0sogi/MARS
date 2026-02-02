import os
import math
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library import config
from library import utils
from library import hfbb
from library import tokenizers
from library import model as model_lib


class TestDataset(Dataset):
    """
    Dataset for inference on test data.
    Encodes 'prev <SEP> curr <SEP> next' into character IDs.
    """

    def __init__(self, df, char_tokenizer, max_input_len):
        self.df = df.reset_index(drop=True)
        self.char_tokenizer = char_tokenizer
        self.max_input_len = max_input_len

        # Ensure strings
        self.befores = self.df["before"].astype(str).tolist()
        self.prevs = self.df["prev"].astype(str).tolist()
        self.nexts = self.df["next"].astype(str).tolist()

        # Store original indices to map predictions back
        self.original_indices = self.df["orig_index"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prev_tok = self.prevs[idx]
        curr_tok = self.befores[idx]
        next_tok = self.nexts[idx]

        sep = self.char_tokenizer.sep_token
        input_text = f"{prev_tok}{sep}{curr_tok}{sep}{next_tok}"

        input_ids = self.char_tokenizer.encode(input_text, add_special_tokens=False)

        # Truncate
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[: self.max_input_len]

        # Pad
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_input_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.char_tokenizer.pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "original_index": self.original_indices[idx],
        }


class HybridNormalizer:
    def __init__(self, load_cached_data=True):
        self.device = config.DEVICE
        print(f"HybridNormalizer: Initializing on {self.device}...")

        # 1. Load Tokenizers
        self.char_tokenizer, self.bpe_tokenizer = tokenizers.build_tokenizers(
            load_cached_data=load_cached_data
        )

        # 2. Load HFBB Stats
        self.hfbb_engine = hfbb.HFBB()
        self.hfbb_engine.build_stats(load_cached_data=load_cached_data)

        # 3. Load Transformer Model
        self.model = self._load_model()

    def _load_model(self):
        print("HybridNormalizer: Loading Transformer model...")
        model = model_lib.SemioticTransformer(
            src_vocab_size=self.char_tokenizer.vocab_size,
            tgt_vocab_size=self.bpe_tokenizer.vocab_size,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
        ).to(self.device)

        if os.path.exists(config.BEST_MODEL_PATH):
            state_dict = torch.load(config.BEST_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            print("HybridNormalizer: Model weights loaded successfully.")
        else:
            print(
                f"Warning: Checkpoint {config.BEST_MODEL_PATH} not found. Using initialized weights (expect poor performance)."
            )
            model.eval()

        return model

    def greedy_decode_batch(self, src, src_key_padding_mask):
        """
        Performs greedy autoregressive decoding for a batch of inputs.
        """
        batch_size = src.size(0)
        max_len = config.MAX_OUTPUT_LEN

        # 1. Run Encoder
        # src: [batch, seq_len] -> transpose to [seq_len, batch]
        src_t = src.transpose(0, 1)

        # Embed and Add Position Info
        src_emb = self.model.src_embedding(src_t) * math.sqrt(self.model.d_model)
        src_emb = self.model.pos_encoder(src_emb)

        # Encode
        memory = self.model.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # 2. Initialize Decoder Input with SOS
        ys = torch.full(
            (batch_size, 1),
            self.bpe_tokenizer.sos_id,
            dtype=torch.long,
            device=self.device,
        )

        # 3. Autoregressive Loop
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            # Prepare Target
            tgt_t = ys.transpose(0, 1)  # [seq, batch]
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_t.size(0)).to(
                self.device
            )

            # Embed Target
            tgt_emb = self.model.tgt_embedding(tgt_t) * math.sqrt(self.model.d_model)
            tgt_emb = self.model.pos_decoder(tgt_emb)

            # Decode
            out = self.model.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Generator (Project to Vocab)
            # Take the last token output: [batch, vocab]
            logits = self.model.generator(out[-1])

            # Greedy choice
            _, next_word = torch.max(logits, dim=1)

            # Update finished status
            finished |= next_word == self.bpe_tokenizer.eos_id

            # Append
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if finished.all():
                break

        return ys

    def predict_transformer_batch(self, batch):
        src = batch["input_ids"].to(self.device)
        # Create padding mask (True where pad_id)
        src_key_padding_mask = src == self.char_tokenizer.pad_id

        with torch.no_grad():
            output_ids = self.greedy_decode_batch(src, src_key_padding_mask)

        # Decode BPE to strings
        predictions = []
        output_ids_list = output_ids.cpu().tolist()
        for ids in output_ids_list:
            # Decode, skipping specials (SOS, EOS, PAD)
            pred_text = self.bpe_tokenizer.decode(ids, skip_special_tokens=True)
            predictions.append(pred_text)

        return predictions

    def generate_submission(self):
        print("HybridNormalizer: Starting submission generation...")

        # 1. Load Test Data
        try:
            df_test = pd.read_csv(config.TEST_FILE)
        except FileNotFoundError:
            print(f"Error: Test file {config.TEST_FILE} not found.")
            return

        # Ensure types
        df_test["before"] = df_test["before"].fillna("").astype(str)

        # 2. Preprocess Context (Prev/Next)
        print("HybridNormalizer: Generating context...")
        # Sort to ensure correct context
        df_test.sort_values(["sentence_id", "token_id"], inplace=True)

        # Prev
        df_test["prev"] = df_test["before"].shift(1).fillna("<START>")
        df_test.loc[df_test["token_id"] == 0, "prev"] = "<START>"

        # Next
        df_test["next"] = df_test["before"].shift(-1).fillna("<END>")
        next_token_id = df_test["token_id"].shift(-1).fillna(0)
        df_test.loc[next_token_id == 0, "next"] = "<END>"

        # 3. HFBB Pass
        print("HybridNormalizer: Running HFBB pass...")

        predictions = [None] * len(df_test)
        transformer_indices = []

        # Iterate efficiently
        # Extract columns to lists for speed
        prevs = df_test["prev"].tolist()
        currs = df_test["before"].tolist()
        nexts = df_test["next"].tolist()

        for i, (p, c, n) in enumerate(zip(prevs, currs, nexts)):
            pred, conf, level = self.hfbb_engine.query(p, c, n)

            # Logic:
            # 1. Context Match -> Accept
            if level in ["TRIGRAM", "BIGRAM_PREV", "BIGRAM_NEXT"]:
                predictions[i] = pred

            # 2. Unigram Match
            elif level == "UNIGRAM":
                if conf > config.CONFIDENCE_THRESHOLD:
                    predictions[i] = pred
                else:
                    # Low confidence check
                    if utils.is_semiotic(c):
                        transformer_indices.append(i)
                    else:
                        # Low confidence but not semiotic (likely regular word) -> Accept Unigram
                        predictions[i] = pred

            # 3. OOV
            else:  # OOV
                if utils.is_semiotic(c):
                    transformer_indices.append(i)
                else:
                    # OOV and not semiotic -> Identity Fallback
                    predictions[i] = c

        # 4. Transformer Pass
        n_transformer = len(transformer_indices)
        print(f"HybridNormalizer: {n_transformer} tokens routed to Transformer.")

        if n_transformer > 0:
            # Create subset dataframe
            df_trans = df_test.iloc[transformer_indices].copy()
            df_trans["orig_index"] = transformer_indices

            # Create Dataset and Loader
            ds = TestDataset(df_trans, self.char_tokenizer, config.MAX_INPUT_LEN)
            loader = DataLoader(
                ds,
                batch_size=config.BATCH_SIZE * 2,  # Inference can handle larger batches
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=(self.device == "cuda"),
            )

            print("HybridNormalizer: Running neural inference...")

            for batch in loader:
                batch_preds = self.predict_transformer_batch(batch)
                orig_idxs = batch["original_index"].tolist()

                for idx, txt in zip(orig_idxs, batch_preds):
                    predictions[idx] = txt

        # 5. Finalize and Save
        print("HybridNormalizer: Finalizing submission...")

        # Fill any remaining Nones (safety fallback)
        for i in range(len(predictions)):
            if predictions[i] is None:
                predictions[i] = currs[i]

        # Construct Submission DataFrame
        # Format: id, after
        # id = sentence_id + "_" + token_id
        sub_ids = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )

        submission = pd.DataFrame({"id": sub_ids, "after": predictions})

        # Save
        utils.save_cache(
            submission, config.SUBMISSION_PATH
        )  # save_cache handles parquet, but here we need CSV
        # Override to save as CSV per competition format
        submission.to_csv(
            config.SUBMISSION_PATH, index=False, quoting=1
        )  # quoting=1 is QUOTE_ALL usually safer, or default
        # The sample uses quotes for strings. Pandas default is usually fine.
        # Let's use strict CSV saving.
        submission.to_csv(config.SUBMISSION_PATH, index=False)

        print(f"Submission saved to {config.SUBMISSION_PATH}")


def generate_submission(load_cached_data=True):
    """
    Wrapper function to execute the full inference pipeline.
    """
    utils.set_seed(config.SEED)
    normalizer = HybridNormalizer(load_cached_data=load_cached_data)
    normalizer.generate_submission()
