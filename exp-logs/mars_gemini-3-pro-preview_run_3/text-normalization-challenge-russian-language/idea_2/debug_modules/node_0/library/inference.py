import os
import torch
import pandas as pd
import numpy as np
import time
from library.config import Config
from library.dataset import (
    prepare_artifacts,
    process_data,
    TextNormalizationDataset,
    get_dataloader,
)
from library.model import TransformerNumNorm
from library.utils import load_checkpoint


def beam_search_decode(
    model, memory, src_mask, tokenizer, beam_width=3, max_len=Config.MAX_SEQ_LEN
):
    """
    Performs beam search decoding for a single sequence.

    Args:
        model: TransformerNumNorm model.
        memory: Encoder output memory (1, src_len, d_model).
        src_mask: Encoder padding mask (1, src_len).
        tokenizer: CharTokenizer instance.
        beam_width: Number of beams to keep.
        max_len: Maximum sequence length.

    Returns:
        str: Decoded string.
    """
    device = memory.device
    sos_idx = Config.SOS_IDX
    eos_idx = Config.EOS_IDX

    # Start with SOS
    # Shape: (1, 1)
    curr_seq = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    # Beams: list of (log_prob_score, sequence_tensor)
    beams = [(0.0, curr_seq)]

    completed_beams = []

    for _ in range(max_len):
        candidates = []

        for score, seq in beams:
            # If beam ended with EOS, it's a candidate for final selection
            if seq[0, -1].item() == eos_idx:
                completed_beams.append((score, seq))
                continue

            # Decode
            # model.decode expects tgt, memory, memory_key_padding_mask
            # It returns output of shape (1, seq_len, d_model)
            with torch.no_grad():
                decoder_output = model.decode(
                    seq, memory, memory_key_padding_mask=src_mask
                )

                # Get logits for the last token
                # Shape: (1, seq_len, vocab_size)
                logits = model.text_head(decoder_output)
                last_token_logits = logits[0, -1, :]  # (vocab_size)

                # Log Softmax
                log_probs = torch.log_softmax(last_token_logits, dim=0)

                # Top K
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for k in range(beam_width):
                    next_score = score + topk_log_probs[k].item()
                    next_idx = topk_indices[k].item()

                    # Append to sequence
                    next_token = torch.tensor(
                        [[next_idx]], dtype=torch.long, device=device
                    )
                    new_seq = torch.cat([seq, next_token], dim=1)

                    candidates.append((next_score, new_seq))

        # If no candidates (all beams finished), break
        if not candidates:
            break

        # Sort candidates by score (descending) and keep top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # Early stopping: if all active beams end with EOS
        if all(b[1][0, -1].item() == eos_idx for b in beams):
            completed_beams.extend(beams)
            break

    # Fallback if loop finishes
    if not completed_beams:
        completed_beams = beams

    # Sort completed beams
    completed_beams.sort(key=lambda x: x[0], reverse=True)

    # Best sequence
    best_seq_tensor = completed_beams[0][1]  # (1, seq_len)

    # Convert to list of indices
    best_indices = best_seq_tensor[0].cpu().numpy().tolist()

    # Decode to string
    return tokenizer.decode(best_indices)


class Predictor:
    """
    Inference class for Text Normalization.
    Implements hybrid logic: Copy if PLAIN/PUNCT, else Beam Search.
    """

    def __init__(self, model_path, tokenizer, class_map, device=Config.DEVICE):
        self.device = device
        self.tokenizer = tokenizer
        self.class_map = class_map

        # Initialize Model
        self.model = TransformerNumNorm(
            vocab_size=tokenizer.vocab_size, num_classes=len(class_map)
        ).to(device)

        # Load Checkpoint
        checkpoint = load_checkpoint(model_path, self.model, device=device)
        if checkpoint:
            print(
                f"Model loaded from {model_path} (Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.4f})"
            )
        else:
            print(
                f"Warning: No checkpoint found at {model_path}. Using random weights."
            )

        self.model.eval()

        # Identify Copy Classes
        self.copy_indices = set()
        if "PLAIN" in class_map:
            self.copy_indices.add(class_map["PLAIN"])
        if "PUNCT" in class_map:
            self.copy_indices.add(class_map["PUNCT"])

    def predict_batch(self, batch, beam_width=Config.BEAM_WIDTH):
        """
        Runs prediction on a batch.
        """
        src = batch["src"].to(self.device)
        ids = batch["id"]

        batch_size = src.size(0)
        predictions = []

        with torch.no_grad():
            # 1. Encode
            # memory: (batch, seq_len, d_model)
            # src_mask: (batch, seq_len) - True where padding
            memory, src_mask = self.model.encode(src)

            # 2. Classify
            # We need to manually replicate the pooling logic from model.forward
            # Create mask for division (batch, seq, 1)
            mask_float = (~src_mask).float().unsqueeze(-1)
            memory_masked = memory * mask_float
            sum_pooled = torch.sum(memory_masked, dim=1)
            lengths = torch.sum(mask_float, dim=1)
            lengths = torch.clamp(lengths, min=1.0)
            mean_pooled = sum_pooled / lengths

            class_logits = self.model.class_head(mean_pooled)
            pred_classes = torch.argmax(class_logits, dim=1)  # (batch,)

            # 3. Hybrid Decoding
            for i in range(batch_size):
                pred_class_idx = pred_classes[i].item()
                current_id = ids[i]

                # Check if we should copy
                if pred_class_idx in self.copy_indices:
                    # Copy Input
                    # Get original indices (excluding padding)
                    src_indices = src[i].cpu().numpy().tolist()
                    decoded_text = self.tokenizer.decode(src_indices)
                    predictions.append((current_id, decoded_text))
                else:
                    # Run Beam Search
                    # Slice memory and mask for this sample: (1, seq_len, ...)
                    mem_i = memory[i : i + 1]
                    mask_i = src_mask[i : i + 1]

                    decoded_text = beam_search_decode(
                        self.model, mem_i, mask_i, self.tokenizer, beam_width=beam_width
                    )
                    predictions.append((current_id, decoded_text))

        return predictions


def generate_submission(
    test_file=Config.TEST_CSV,
    submission_file=Config.SUBMISSION_FILE,
    model_path=Config.MODEL_CHECKPOINT,
    batch_size=Config.BATCH_SIZE,
    subset_size=None,
):
    """
    Generates the submission file for the test set.

    Args:
        test_file: Path to test metadata CSV.
        submission_file: Path to save submission CSV.
        model_path: Path to trained model checkpoint.
        batch_size: Batch size for inference.
        subset_size: If set, only process this many samples (for debugging).
    """
    print(f"Generating submission from {test_file}...")

    # 1. Prepare Artifacts (Tokenizer, Class Map)
    # This assumes training has been done and artifacts exist
    tokenizer, class_map = prepare_artifacts(load_cached_data=True)

    # 2. Process Test Data
    # This uses the cached parquet if available, or processes from scratch
    df_test = process_data("test", tokenizer, class_map, load_cached_data=True)

    if subset_size:
        print(f"Debugging: Using subset of {subset_size} samples.")
        df_test = df_test.iloc[:subset_size]

    # 3. Create DataLoader
    test_dataset = TextNormalizationDataset(df_test)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Predictor
    predictor = Predictor(model_path, tokenizer, class_map)

    # 5. Run Inference
    all_predictions = []
    start_time = time.time()

    print(f"Starting inference on {len(test_dataset)} samples...")

    for batch_idx, batch in enumerate(test_loader):
        preds = predictor.predict_batch(batch)
        all_predictions.extend(preds)

        if (batch_idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {batch_idx + 1} batches ({len(all_predictions)} samples) - {elapsed:.2f}s"
            )

    # 6. Save Submission
    print("Saving submission...")
    df_submission = pd.DataFrame(all_predictions, columns=["id", "after"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(submission_file, index=False)

    print(f"Submission saved to {submission_file}")
    print(f"Total time: {time.time() - start_time:.2f}s")
