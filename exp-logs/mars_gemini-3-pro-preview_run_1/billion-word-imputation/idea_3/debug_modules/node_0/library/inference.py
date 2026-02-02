import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.model import DecoupledTransformer
from library.dataset import get_dataloaders
from library.utils import get_or_build_vocab, SOS_TOKEN, EOS_TOKEN


class InferenceEngine:
    """
    Inference Engine for the Decoupled Localization-Classification Transformer.
    Handles model loading, prediction generation, and submission file creation.
    """

    def __init__(self):
        # Ensure reproducibility
        Config.set_seed()

        self.device = torch.device(Config.DEVICE)

        # Load Vocabulary
        # We assume vocabulary exists (created during training)
        self.vocab = get_or_build_vocab(load_cached_data=True)
        self.vocab_size = len(self.vocab)

        # Initialize Model Architecture
        self.model = DecoupledTransformer(self.vocab_size).to(self.device)

        # Load Model Weights
        if os.path.exists(Config.MODEL_PATH):
            print(f"Loading model weights from {Config.MODEL_PATH}")
            state_dict = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"WARNING: Model checkpoint not found at {Config.MODEL_PATH}. Using random weights."
            )

        self.model.eval()

    def predict(self, loader):
        """
        Runs inference on the provided loader and returns a list of formatted CSV lines.

        Args:
            loader (DataLoader): DataLoader containing test data.

        Returns:
            list[str]: List of strings formatted as 'id,"sentence"'.
        """
        results = []
        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=-1)

        # Define special tokens to remove during reconstruction
        special_tokens = {SOS_TOKEN, EOS_TOKEN, Config.PAD_TOKEN, Config.MASK_TOKEN}
        eos_idx = self.vocab.stoi.get(EOS_TOKEN)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                # row_id is provided by the test collate_fn
                row_ids = batch["row_id"].cpu().numpy()

                batch_size = input_ids.size(0)

                # Forward Pass
                loc_logits, id_logits = self.model(input_ids, attention_mask)

                # --- Joint Scoring Strategy ---
                # P(Location) = Sigmoid(Loc_Logits) -> Shape: (B, L, 1)
                p_loc = sigmoid(loc_logits).unsqueeze(-1)

                # P(Word) = Softmax(ID_Logits) -> Shape: (B, L, V)
                p_word = softmax(id_logits)

                # Score = P(Location) * P(Word) -> Shape: (B, L, V)
                scores = p_loc * p_word

                # Mask out padding positions in the scores to prevent invalid insertions
                # attention_mask is (B, L), need (B, L, 1) for broadcasting
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(scores)
                scores = scores * mask_expanded

                # Find the global maximum score for each sample in the batch
                # Flatten to (B, L*V)
                scores_flat = scores.view(batch_size, -1)
                best_flat_indices = torch.argmax(scores_flat, dim=1)

                # Decode flat index back to (Position, Word_Index)
                best_pos = best_flat_indices // self.vocab_size
                best_word_idx = best_flat_indices % self.vocab_size

                # Move results to CPU for string processing
                input_ids_cpu = input_ids.cpu().numpy()
                best_pos_cpu = best_pos.cpu().numpy()
                best_word_idx_cpu = best_word_idx.cpu().numpy()

                # Reconstruct Sentences
                for i in range(batch_size):
                    rid = row_ids[i]
                    curr_ids = input_ids_cpu[i]
                    pos = best_pos_cpu[i]
                    w_idx = best_word_idx_cpu[i]

                    # Determine the actual length of the sentence (excluding padding)
                    # We look for the EOS token
                    eos_locs = np.where(curr_ids == eos_idx)[0]
                    if len(eos_locs) > 0:
                        real_len = (
                            eos_locs[0] + 1
                        )  # Include EOS to keep structure valid until cleaning
                    else:
                        real_len = len(curr_ids)

                    # Extract valid tokens
                    valid_tokens = list(curr_ids[:real_len])

                    # Insert the predicted word
                    # 'pos' is the index of the token *before* the gap.
                    # We want to insert *after* 'pos'.
                    # list.insert(i, x) inserts x before index i.
                    # So we insert at pos + 1.
                    valid_tokens.insert(pos + 1, w_idx)

                    # Decode indices to strings
                    decoded_tokens = self.vocab.decode(valid_tokens)

                    # Remove special tokens
                    clean_tokens = [
                        t for t in decoded_tokens if t not in special_tokens
                    ]

                    # Join to form sentence
                    sentence = " ".join(clean_tokens)

                    # Escape double quotes for CSV format (double double quotes)
                    sentence = sentence.replace('"', '""')

                    # Format: id,"sentence"
                    results.append(f'{rid},"{sentence}"')

        return results

    def generate_submission(self, load_cached_data=True):
        """
        Generates the submission file for the test set.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
        """
        print("Preparing test data...")
        # We only need the test loader here
        _, _, test_loader = get_dataloaders(
            self.vocab, load_cached_data=load_cached_data
        )

        print("Running inference...")
        submission_lines = self.predict(test_loader)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        print(f"Writing results to {Config.SUBMISSION_PATH}...")
        with open(Config.SUBMISSION_PATH, "w", encoding="utf-8") as f:
            # Write Header
            f.write('id,"sentence"\n')
            # Write Rows
            for line in submission_lines:
                f.write(line + "\n")

        print("Submission generation complete.")
