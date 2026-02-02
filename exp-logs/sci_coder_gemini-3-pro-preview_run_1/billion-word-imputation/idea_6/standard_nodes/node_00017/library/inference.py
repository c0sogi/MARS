import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.vocab import Vocabulary
from library.model import BifurcatedTransformer
from library.data import InterleavedDataset, collate_fn


class Predictor:
    """
    Handles inference for the Bifurcated Interleaved Transformer.
    Generates predictions for the test set and saves the submission file.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.set_seed(Config.SEED)

        # Initialize Vocabulary
        self.vocab = Vocabulary()
        self.vocab.build(load_cached_data=True)

        # Initialize Model structure
        self.model = BifurcatedTransformer().to(self.device)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def predict(self):
        """
        Runs the inference pipeline:
        1. Loads model weights.
        2. Processes test data.
        3. Computes fusion scores.
        4. Reconstructs sentences.
        5. Writes submission file.
        """
        print("Initializing prediction pipeline...")

        # Load best model weights
        if os.path.exists(Config.MODEL_PATH):
            print(f"Loading model weights from {Config.MODEL_PATH}")
            state_dict = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model file not found at {Config.MODEL_PATH}. Using random weights."
            )

        self.model.eval()

        # Load Test Data
        # load_cached_data=True ensures we use the parquet cache if generated during training/setup
        test_dataset = InterleavedDataset("test", self.vocab, load_cached_data=True)

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predictions = []
        ids = []

        print(f"Starting inference on {len(test_dataset)} samples...")

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_ids = batch["id"]

                # Forward Pass
                # loc_logits: (B, S, 1)
                # id_logits: (B, S, V)
                loc_logits, id_logits = self.model(input_ids, attention_mask)

                # Compute Probabilities
                loc_probs = torch.sigmoid(loc_logits).squeeze(-1)  # (B, S)
                id_probs = torch.softmax(id_logits, dim=-1)  # (B, S, V)

                # Structural Masking
                # In the interleaved sequence [w0, GAP, w1, GAP, ...], gaps are at indices 1, 3, 5...
                # We mask out even indices (0, 2, 4...) because a missing word cannot be inside an existing word.
                seq_len = input_ids.shape[1]
                gap_mask = torch.zeros((seq_len,), device=self.device)
                gap_mask[1::2] = 1.0

                # Apply masks to localization probabilities
                # 1. Structural mask (only odd positions)
                # 2. Attention mask (ignore padding)
                loc_probs = loc_probs * gap_mask.unsqueeze(0)
                loc_probs = loc_probs * attention_mask

                # Score Fusion
                # Score(i, w) = P(Location=i) * P(Word=w | Location=i)
                # We broadcast loc_probs to (B, S, 1) to multiply with id_probs (B, S, V)
                scores = loc_probs.unsqueeze(-1) * id_probs  # (B, S, V)

                # Find the global maximum score for each sample in the batch
                batch_size = input_ids.shape[0]
                # Flatten (S, V) dimensions
                scores_flat = scores.view(batch_size, -1)
                best_flat_indices = torch.argmax(scores_flat, dim=1)

                # Decode flat indices back to (gap_index, word_index)
                best_gap_indices = best_flat_indices // Config.VOCAB_SIZE
                best_word_indices = best_flat_indices % Config.VOCAB_SIZE

                # Move to CPU for string manipulation
                input_ids_cpu = input_ids.cpu().numpy()
                best_gap_indices = best_gap_indices.cpu().numpy()
                best_word_indices = best_word_indices.cpu().numpy()

                # Reconstruct Sentences
                for k in range(batch_size):
                    curr_input = input_ids_cpu[k]
                    curr_gap_idx = best_gap_indices[k]
                    curr_word_idx = best_word_indices[k]

                    # 1. Identify the predicted word
                    pred_word = self.vocab.itos.get(curr_word_idx, Config.UNK_TOKEN)

                    # 2. Extract the original words from the interleaved input
                    # The input is [w0, GAP, w1, GAP, w2, PAD...].
                    # Words are located at even indices: 0, 2, 4...
                    words = []
                    for idx, token_id in enumerate(curr_input):
                        if token_id == 0:  # PAD token
                            break
                        if idx % 2 == 0:
                            word = self.vocab.itos.get(token_id, Config.UNK_TOKEN)
                            words.append(word)

                    # 3. Determine insertion index
                    # If the gap is at index `g` (odd), it lies between word `(g-1)/2` and word `(g+1)/2`.
                    # In a Python list `words`, this corresponds to `insert` at index `(g+1)//2`.
                    # Example: Gap at 1 (between w0, w1) -> insert at 1. Result: w0, NEW, w1.
                    insert_idx = (curr_gap_idx + 1) // 2

                    # Safety clamp (should not be needed if logic is correct, but good for robustness)
                    if insert_idx > len(words):
                        insert_idx = len(words)

                    # 4. Insert and Join
                    words.insert(insert_idx, pred_word)
                    sentence = " ".join(words)

                    ids.append(batch_ids[k])
                    predictions.append(sentence)

        # Save Submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        with open(Config.SUBMISSION_PATH, "w", encoding="utf-8") as f:
            f.write('id,"sentence"\n')
            for pid, sent in zip(ids, predictions):
                # Escape double quotes according to CSV spec: " -> ""
                sent_escaped = sent.replace('"', '""')
                # Wrap the sentence in double quotes
                f.write(f'{pid},"{sent_escaped}"\n')

        print("Submission generation complete.")
