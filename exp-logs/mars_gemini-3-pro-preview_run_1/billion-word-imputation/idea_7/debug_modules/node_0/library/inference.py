import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.model import SyntaxAwareTransformer
from library.vocab import load_or_build_artifacts
from library.data import get_test_dataloader


class ConsistencyDecoder:
    """
    Decodes model outputs using consistency scoring:
    Score = P(Loc) * P(Word) * P(Syntax=Tag(Word))

    This decoder enforces that the predicted word is consistent with both
    the structural gap location and the predicted grammatical category.
    """

    def __init__(self, pos_map, device):
        """
        Args:
            pos_map (np.array): Mapping from word_id to pos_tag_id.
            device (torch.device): Device for tensor operations.
        """
        self.pos_map = torch.tensor(pos_map, device=device, dtype=torch.long)
        self.device = device

    def decode(self, loc_logits, syntax_logits, word_logits, gap_mask):
        """
        Computes the best (gap_idx, word_idx) pair for each item in the batch.

        Args:
            loc_logits: (B, S) - Logits for gap localization.
            syntax_logits: (B, S, Num_Tags) - Logits for POS tag prediction.
            word_logits: (B, S, Vocab_Size) - Logits for word identification.
            gap_mask: (B, S) - 1 where tokens are GAPs, 0 otherwise.

        Returns:
            best_gap_indices: (B,) - Index of the gap to insert into.
            best_word_indices: (B,) - Index of the word to insert.
        """
        # 1. Compute Probabilities
        loc_probs = torch.sigmoid(loc_logits)  # (B, S)
        syntax_probs = torch.softmax(syntax_logits, dim=-1)  # (B, S, T)
        word_probs = torch.softmax(word_logits, dim=-1)  # (B, S, V)

        # 2. Expand Syntax Probabilities
        # We map every word in the vocabulary to its static POS tag probability.
        # syntax_probs: (B, S, T)
        # pos_map: (V) -> indices into T
        # Result: (B, S, V)
        syntax_probs_expanded = syntax_probs[:, :, self.pos_map]

        # 3. Compute Final Consistency Score
        # Score(b, s, w) = P(Loc at s) * P(Word w) * P(Tag of w at s)
        # loc_probs: (B, S) -> (B, S, 1) for broadcasting
        final_scores = (
            loc_probs.unsqueeze(-1) * word_probs * syntax_probs_expanded
        )  # (B, S, V)

        # 4. Mask Invalid Locations
        # We can only insert at GAP tokens.
        # gap_mask: (B, S) -> (B, S, 1)
        mask = gap_mask.unsqueeze(-1).bool()
        final_scores = final_scores.masked_fill(~mask, -1.0)

        # 5. Select Best Candidates
        batch_size, seq_len, vocab_size = final_scores.shape
        # Flatten (S, V) dimensions to find global max per batch item
        final_scores_flat = final_scores.view(batch_size, -1)  # (B, S*V)

        best_indices_flat = torch.argmax(final_scores_flat, dim=1)  # (B,)

        # Convert flat index back to (gap_idx, word_idx)
        best_gap_indices = best_indices_flat // vocab_size
        best_word_indices = best_indices_flat % vocab_size

        return best_gap_indices, best_word_indices


def generate_submission(
    batch_size=Config.BATCH_SIZE,
    model_path=Config.MODEL_SAVE_PATH,
    submission_path=Config.SUBMISSION_FILE,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        batch_size (int): Batch size for inference.
        model_path (str): Path to the trained model checkpoint.
        submission_path (str): Path to save the submission CSV.
    """
    device = get_device()
    print(f"Inference Device: {device}")

    # 1. Load Artifacts (Vocab, POS Map)
    # Ensure artifacts exist (created during training or built now)
    vocab, pos_map, _ = load_or_build_artifacts(load_cached_data=True)

    # 2. Initialize Decoder
    decoder = ConsistencyDecoder(pos_map, device)

    # 3. Load Model
    model = SyntaxAwareTransformer().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random initialization."
        )

    model.eval()

    # 4. Prepare Data
    test_loader = get_test_dataloader(batch_size=batch_size)

    predictions = []
    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gap_mask = batch["gap_mask"].to(device)
            row_ids = batch["row_ids"]

            # Forward Pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Decode
            best_gap_indices, best_word_indices = decoder.decode(
                outputs["loc_logits"],
                outputs["syntax_logits"],
                outputs["word_logits"],
                gap_mask,
            )

            # Reconstruct Sentences
            input_ids_cpu = input_ids.cpu().numpy()
            best_gap_indices = best_gap_indices.cpu().numpy()
            best_word_indices = best_word_indices.cpu().numpy()

            for i, row_id in enumerate(row_ids):
                gap_idx = best_gap_indices[i]
                word_idx = best_word_indices[i]

                predicted_word = vocab.lookup_token(word_idx)
                current_seq = input_ids_cpu[i]

                reconstructed_tokens = []

                # Iterate through the input sequence
                # Input format: [SOS, GAP, w1, GAP, w2, ..., GAP, EOS, PAD...]
                for seq_idx, token_id in enumerate(current_seq):
                    # If this index matches the predicted gap, insert the word
                    if seq_idx == gap_idx:
                        reconstructed_tokens.append(predicted_word)

                    # Stop processing if we hit EOS or PAD (EOS usually sufficient)
                    if token_id == Config.EOS_IDX:
                        break

                    # Append existing words (skipping special tokens)
                    # Note: We skip GAP tokens unless it was the target gap handled above
                    if token_id not in [
                        Config.SOS_IDX,
                        Config.GAP_IDX,
                        Config.PAD_IDX,
                        Config.EOS_IDX,
                    ]:
                        word = vocab.lookup_token(token_id)
                        reconstructed_tokens.append(word)

                # Join tokens to form sentence
                pred_sentence = " ".join(reconstructed_tokens)
                predictions.append({"id": row_id, "sentence": pred_sentence})

    # 5. Save Submission
    df_pred = pd.DataFrame(predictions)

    # Ensure sorted by ID
    df_pred = df_pred.sort_values("id")

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Write to CSV with custom formatting
    # Format: id,"sentence"
    # Double quotes within sentence must be escaped as ""
    with open(submission_path, "w", encoding="utf-8") as f:
        f.write('id,"sentence"\n')
        for _, row in df_pred.iterrows():
            sent = row["sentence"]
            # Escape double quotes
            sent = sent.replace('"', '""')
            # Write row
            f.write(f'{row["id"]},"{sent}"\n')

    print(f"Submission saved to {submission_path}")
