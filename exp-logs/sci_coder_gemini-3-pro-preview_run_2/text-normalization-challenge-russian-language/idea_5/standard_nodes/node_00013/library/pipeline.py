import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_data, is_semiotic
from library.hfbb import HFBBEngine
from library.tokenizers import HeterogeneousTokenizer
from library.model import CharToSubwordTransformer
from library.train_eval import Trainer
from torch.nn.utils.rnn import pad_sequence


class HybridInference:
    """
    Encapsulates the inference logic for the Text Normalization task.
    Implements a strict priority cascade:
    1. HFBB (Memory)
    2. Neural Network (Semiotic Tokens)
    3. Identity (Fallback)
    """

    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE
        self.hfbb = HFBBEngine()
        self.tokenizer = HeterogeneousTokenizer()
        self.model = None
        set_seed()

    def load_resources(self, load_cached_data=True):
        """
        Loads the HFBB engine, Tokenizer, and Neural Model.
        """
        print("HybridInference: Loading resources...")

        # 1. Load HFBB (Tier 1)
        # We need training data to fit HFBB if not cached
        if load_cached_data and all(
            os.path.exists(p) for p in self.hfbb.cache_files.values()
        ):
            self.hfbb.fit(None, load_cached_data=True)
        else:
            print(
                "HybridInference: HFBB cache missing. Loading training data to fit HFBB..."
            )
            train_df = load_data("train")
            self.hfbb.fit(train_df, load_cached_data=load_cached_data)
            del train_df

        # 2. Load Tokenizer
        # Tokenizer handles its own cache checking
        self.tokenizer.fit(None, load_cached_data=True)

        # 3. Load Model (Tier 2)
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"HybridInference: Loading model from {Config.BEST_MODEL_PATH}")
            src_vocab = self.tokenizer.get_source_vocab_size()
            tgt_vocab = self.tokenizer.get_target_vocab_size()

            self.model = CharToSubwordTransformer(
                src_vocab_size=src_vocab,
                tgt_vocab_size=tgt_vocab,
                src_pad_idx=self.tokenizer.pad_id,
                tgt_pad_idx=0,
                d_model=Config.ENC_EMB_DIM,
                nhead=Config.ENC_HEADS,
                num_encoder_layers=Config.ENC_LAYERS,
                num_decoder_layers=Config.DEC_LAYERS,
                dim_feedforward=Config.ENC_HIDDEN_DIM,
                dropout=Config.DROPOUT,
                max_len=Config.MAX_SEQ_LEN,
            )
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
        else:
            print(
                "HybridInference: Warning - Model checkpoint not found. Tier 2 will fail if needed."
            )

    def _generate_context(self, df):
        """
        Adds prev_word and next_word columns to the test dataframe.
        """
        print("HybridInference: Generating context for test data...")
        df = df.sort_values(["sentence_id", "token_id"]).copy()

        # Shift to get context
        # GroupBy sentence_id to ensure isolation
        df["prev_word"] = (
            df.groupby("sentence_id")["before"].shift(1).fillna(Config.PAD_TOKEN)
        )
        df["next_word"] = (
            df.groupby("sentence_id")["before"].shift(-1).fillna(Config.PAD_TOKEN)
        )

        return df

    def _batch_greedy_decode(self, src_batch, max_len=None):
        """
        Performs greedy decoding for a batch of source sequences.
        """
        if max_len is None:
            max_len = Config.MAX_SEQ_LEN

        batch_size = src_batch.size(0)
        src_batch = src_batch.to(self.device)

        # Encode
        memory = self.model.encode(src_batch)

        # Initialize target with SOS
        tgt = torch.full(
            (batch_size, 1), self.tokenizer.sos_id, device=self.device, dtype=torch.long
        )

        # Tracking finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Decoding loop
        for _ in range(max_len):
            # Create padding mask for memory attention
            src_padding_mask = src_batch == self.tokenizer.pad_id

            # Decode step
            out = self.model.decode(
                tgt, memory, memory_key_padding_mask=src_padding_mask
            )

            # Project to vocab
            logits = self.model.fc_out(out)  # (B, Seq, Vocab)

            # Greedy selection
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # (B,)

            # Append
            tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

            # Check EOS
            finished |= next_tokens == self.tokenizer.eos_id

            if finished.all():
                break

        return tgt

    def predict(self, test_df, batch_size=Config.BATCH_SIZE):
        """
        Runs the hybrid inference pipeline on the test set.
        """
        # 1. Prepare Data
        df = self._generate_context(test_df)

        # Initialize results container: {index: prediction}
        results = {}

        # Indices requiring neural inference
        neural_indices = []
        neural_inputs = []  # List of (prev, token, next)

        print("HybridInference: Executing Tier 1 (HFBB) and Tier 3 (Identity)...")

        # Iterate efficiently using records
        records = df.to_dict("records")

        for idx, row in enumerate(records):
            token = str(row["before"])
            prev_w = str(row["prev_word"])
            next_w = str(row["next_word"])

            # Tier 1: HFBB
            hfbb_res = self.hfbb.query(token, prev_w, next_w)

            if hfbb_res is not None:
                results[idx] = hfbb_res
            else:
                # Check for Tier 2: Neural
                if is_semiotic(token):
                    neural_indices.append(idx)
                    neural_inputs.append((prev_w, token, next_w))
                else:
                    # Tier 3: Identity
                    results[idx] = token

        print(
            f"HybridInference: Tier 1 & 3 complete. {len(results)} resolved. {len(neural_indices)} sent to Tier 2 (Neural)."
        )

        # 2. Tier 2: Neural Inference
        if neural_indices and self.model is not None:
            print("HybridInference: Executing Tier 2 (Neural)...")

            # Process in batches
            num_samples = len(neural_indices)
            num_batches = (num_samples + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                batch_inputs = neural_inputs[start_idx:end_idx]
                batch_indices = neural_indices[start_idx:end_idx]

                # Prepare Batch
                src_ids_list = []
                for prev, tok, nxt in batch_inputs:
                    src_text = f"{prev}{Config.SEP_TOKEN}{tok}{Config.SEP_TOKEN}{nxt}"
                    encoded_ids = self.tokenizer.encode_source(src_text)
                    if len(encoded_ids) > Config.MAX_SEQ_LEN:
                        encoded_ids = encoded_ids[: Config.MAX_SEQ_LEN]
                    src_ids_list.append(torch.tensor(encoded_ids, dtype=torch.long))

                # Pad
                src_batch = pad_sequence(
                    src_ids_list, batch_first=True, padding_value=self.tokenizer.pad_id
                )

                # Inference
                with torch.no_grad():
                    generated_ids = self._batch_greedy_decode(src_batch)

                # Decode
                generated_ids = generated_ids.cpu().numpy()
                for j, seq_ids in enumerate(generated_ids):
                    # Remove SOS (first token)
                    seq_ids = seq_ids[1:]

                    # Find EOS and truncate
                    eos_locs = np.where(seq_ids == self.tokenizer.eos_id)[0]
                    if len(eos_locs) > 0:
                        seq_ids = seq_ids[: eos_locs[0]]

                    pred_text = self.tokenizer.decode_target(seq_ids.tolist())

                    # Store result
                    original_idx = batch_indices[j]
                    results[original_idx] = pred_text

                if (i + 1) % 100 == 0:
                    print(f"  Processed batch {i+1}/{num_batches}")

        elif neural_indices and self.model is None:
            print(
                "HybridInference: Model missing. Fallback to Identity for semiotic tokens."
            )
            for idx in neural_indices:
                results[idx] = df.iloc[idx]["before"]

        # 3. Compile Final DataFrame
        print("HybridInference: Compiling final predictions...")
        # Sort results by index to match original order
        final_preds = [results[i] for i in range(len(df))]

        submission_df = df[["sentence_id", "token_id"]].copy()
        submission_df["after"] = final_preds

        # Create 'id' column as required: sentence_id_token_id
        submission_df["id"] = (
            submission_df["sentence_id"].astype(str)
            + "_"
            + submission_df["token_id"].astype(str)
        )

        # Select required columns
        submission_df = submission_df[["id", "after"]]

        return submission_df


def run_pipeline(epochs=Config.EPOCHS, load_cached_data=True):
    """
    Orchestrates the full pipeline:
    1. Checks/Trains Model.
    2. Runs Inference.
    3. Saves Submission.
    """
    set_seed()
    Config.setup_directories()

    # 1. Training (if needed)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Pipeline: Model not found. Initiating training...")
        trainer = Trainer(device=Config.DEVICE)
        trainer.run(epochs=epochs, load_cached_data=load_cached_data)
    else:
        print("Pipeline: Model found. Skipping training.")

    # 2. Inference
    inference_engine = HybridInference(device=Config.DEVICE)
    inference_engine.load_resources(load_cached_data=load_cached_data)

    print("Pipeline: Loading test data...")
    test_df = load_data("test")

    print("Pipeline: Running inference...")
    submission_df = inference_engine.predict(test_df)

    # 3. Save
    print(f"Pipeline: Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Pipeline: Done.")
