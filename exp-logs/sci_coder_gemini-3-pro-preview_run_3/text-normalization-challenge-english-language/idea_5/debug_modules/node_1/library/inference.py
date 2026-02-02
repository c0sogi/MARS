import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.symbolic_stats import HierarchicalLookup
from library.neural_data import CharTokenizer, prepare_neural_data
from library.transformer import Seq2SeqTransformer, generate_square_subsequent_mask


class CascadePredictor:
    """
    Implements the Specificity-Based Cascade inference pipeline:
    1. Symbolic Lookup (Trigram -> Bigram -> Unigram)
    2. Heuristic Gate (Identity for Alpha OOV)
    3. Neural Model (Class-Conditioned Transformer)
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(Config.DEVICE)

        # 1. Load Symbolic Memory
        print("Initializing Symbolic Memory...")
        self.lookup = HierarchicalLookup()

        # 2. Load Tokenizer
        print("Loading Tokenizer...")
        self.tokenizer = CharTokenizer()
        if os.path.exists(Config.TOKENIZER_PATH):
            self.tokenizer.load(Config.TOKENIZER_PATH)
        else:
            # This should ideally not happen if training/prep was run
            print(f"Warning: Tokenizer not found at {Config.TOKENIZER_PATH}")

        # 3. Load Neural Model
        print("Loading Neural Model...")
        vocab_size = len(self.tokenizer)
        self.model = Seq2SeqTransformer(
            num_tokens=vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        ).to(self.device)

        if os.path.exists(Config.MODEL_CHECKPOINT):
            state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {Config.MODEL_CHECKPOINT}")
        else:
            print(
                "Warning: Model checkpoint not found. Neural predictions will be random."
            )

    def greedy_decode(self, src, max_len=128):
        """
        Performs greedy decoding for a batch of source sequences.
        """
        batch_size = src.size(0)

        # Identify padding in source to mask memory
        pad_id = self.tokenizer.get_id(Config.PAD_TOKEN)
        src_padding_mask = (src == pad_id).to(self.device)

        # Encode source
        # src_mask is None because we allow full attention in encoder
        memory = self.model.encode(
            src, src_mask=None, src_padding_mask=src_padding_mask
        )

        # Initialize decoder input with SOS
        sos_id = self.tokenizer.get_id(Config.SOS_TOKEN)
        eos_id = self.tokenizer.get_id(Config.EOS_TOKEN)

        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=self.device)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(max_len):
            # Create causal mask for target
            tgt_mask = generate_square_subsequent_mask(ys.size(1), self.device)

            # Decode
            out = self.model.decode(
                ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask
            )

            # Predict next token
            # out shape: (batch, seq_len, d_model) -> take last step
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            # Check for EOS
            finished |= next_word.squeeze() == eos_id
            if finished.all():
                break

        return ys

    def predict(self, df_test):
        """
        Runs the cascade prediction logic on the test dataframe.
        df_test must contain: 'prev', 'before', 'next', 'id'
        """
        results = {}  # id -> prediction_string

        neural_indices = []
        neural_inputs = []

        print("Running Symbolic Lookup and Heuristic Gate...")

        sep_id = self.tokenizer.get_id(Config.SEP_TOKEN)

        # Iterate efficiently
        for row in df_test.itertuples():
            r_id = row.id
            prev = str(row.prev)
            curr = str(row.before)
            nxt = str(row.next)

            # --- Stage 1: Symbolic Lookup ---
            res = self.lookup.query(prev, curr, nxt)
            if res is not None:
                results[r_id] = res
                continue

            # --- Stage 2: Heuristic Gate ---
            # If OOV and purely alphabetic, assume identity
            if curr.isalpha():
                results[r_id] = curr
                continue

            # --- Stage 3: Prepare for Neural Model ---
            neural_indices.append(r_id)

            # Input Format: [prev] <SEP> [curr] <SEP> [next]
            ids = (
                self.tokenizer.encode(prev)
                + [sep_id]
                + self.tokenizer.encode(curr)
                + [sep_id]
                + self.tokenizer.encode(nxt)
            )

            # Truncate if necessary
            if len(ids) > Config.MAX_SEQ_LEN:
                ids = ids[: Config.MAX_SEQ_LEN]
            neural_inputs.append(ids)

        print(f"Symbolic/Heuristic handled {len(results)} samples.")
        print(f"Neural model queued for {len(neural_inputs)} samples.")

        # Run Neural Inference in Batches
        if neural_inputs:
            batch_size = Config.BATCH_SIZE
            num_batches = (len(neural_inputs) + batch_size - 1) // batch_size
            pad_id = self.tokenizer.get_id(Config.PAD_TOKEN)

            print(f"Processing neural candidates in {num_batches} batches...")

            for i in range(num_batches):
                batch_ids = neural_inputs[i * batch_size : (i + 1) * batch_size]
                batch_indices = neural_indices[i * batch_size : (i + 1) * batch_size]

                # Pad batch
                max_len_batch = max(len(x) for x in batch_ids)
                src_tensor = torch.full(
                    (len(batch_ids), max_len_batch), pad_id, dtype=torch.long
                )
                for j, seq in enumerate(batch_ids):
                    src_tensor[j, : len(seq)] = torch.tensor(seq, dtype=torch.long)

                src_tensor = src_tensor.to(self.device)

                # Inference
                with torch.no_grad():
                    output_seqs = self.greedy_decode(src_tensor)

                # Decode and Post-process
                for j, seq_ids in enumerate(output_seqs):
                    # Convert tensor to list
                    seq_list = seq_ids.cpu().tolist()

                    # Remove SOS
                    if seq_list[0] == self.tokenizer.get_id(Config.SOS_TOKEN):
                        seq_list = seq_list[1:]

                    # Truncate at EOS
                    eos_id = self.tokenizer.get_id(Config.EOS_TOKEN)
                    if eos_id in seq_list:
                        seq_list = seq_list[: seq_list.index(eos_id)]

                    # Decode to string
                    decoded_str = self.tokenizer.decode(seq_list)

                    # Post-processing: Format is <CLASS><SEP>text
                    # We need to extract 'text'
                    final_text = decoded_str

                    if Config.SEP_TOKEN in decoded_str:
                        parts = decoded_str.split(Config.SEP_TOKEN, 1)
                        if len(parts) > 1:
                            final_text = parts[1]
                    else:
                        # Fallback: Try to strip class token if SEP is missing
                        for cls_tag in Config.CLASS_TOKENS:
                            if final_text.startswith(cls_tag):
                                final_text = final_text[len(cls_tag) :]
                                break

                    results[batch_indices[j]] = final_text

        return results

    def generate_submission(self, load_cached_data=True):
        """
        Main entry point to generate the submission file.
        1. Prepares/Loads test data.
        2. Runs prediction cascade.
        3. Saves CSV.
        """
        # Ensure data is ready (Tokenizer built, Test data processed)
        print("Preparing data for inference...")
        prepare_neural_data(load_cached_data=load_cached_data)

        if not os.path.exists(Config.PROCESSED_TEST):
            raise FileNotFoundError(
                f"Processed test data not found at {Config.PROCESSED_TEST}"
            )

        print(f"Loading test data from {Config.PROCESSED_TEST}...")
        df_test = pd.read_parquet(Config.PROCESSED_TEST)

        # Run prediction
        results_map = self.predict(df_test)

        # Align with submission format
        print("Formatting submission...")
        ids = df_test["id"].tolist()
        predictions = [results_map.get(uid, "") for uid in ids]

        submission = pd.DataFrame({"id": ids, "after": predictions})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
