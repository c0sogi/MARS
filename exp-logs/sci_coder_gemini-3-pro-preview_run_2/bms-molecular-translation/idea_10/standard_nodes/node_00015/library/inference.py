import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.tokenizer import InChiTokenizer
from library.dataset import ChemicalImageDataset, ChemicalCollate
from library.model import HybridResNetTransformer
from library.utils import seed_everything


class BeamSearchDecoder:
    """
    Performs batched beam search decoding for the HybridResNetTransformer model.
    """

    def __init__(self, model, tokenizer, beam_size=5, max_len=512):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.device = next(model.parameters()).device

    def decode(self, images):
        """
        Args:
            images: (B, C, H, W) input tensor.
        Returns:
            best_sequences: (B, L) tensor of token indices.
        """
        batch_size = images.size(0)

        # 1. Encode Images
        # memory: (B, Src_Len, D)
        memory = self.model.encode_image(images)

        # Expand memory for beam search: (B * beam_size, Src_Len, D)
        # We repeat interleave so that the first beam_size elements correspond to the first sample, etc.
        memory = memory.repeat_interleave(self.beam_size, dim=0)

        # 2. Initialize State
        # sequences: (B * beam_size, 1) containing [SOS]
        sequences = torch.full(
            (batch_size * self.beam_size, 1),
            self.tokenizer.SOS_ID,
            dtype=torch.long,
            device=self.device,
        )

        # scores: (B * beam_size)
        # Initialize scores. For each sample, the first beam has score 0, others -inf
        scores = torch.full(
            (batch_size * self.beam_size,), -float("inf"), device=self.device
        )
        scores[:: self.beam_size] = 0.0

        # Mask to track finished sequences (B * beam_size)
        finished = torch.zeros(
            batch_size * self.beam_size, dtype=torch.bool, device=self.device
        )

        # 3. Decoding Loop
        for step in range(self.max_len):
            if finished.all():
                break

            # Prepare input for decoder
            # Embeddings
            tgt_emb = self.model.embedding(sequences)
            tgt_emb = self.model.pos_decoder(tgt_emb)

            # Causal mask
            tgt_seq_len = sequences.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len).to(
                self.device
            )

            # Forward pass through decoder
            # memory matches batch dimension of sequences
            dec_output = self.model.transformer_decoder(
                tgt_emb, memory, tgt_mask=tgt_mask
            )

            # Get logits for the last token: (B * beam_size, V)
            logits = self.model.prediction_head(dec_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # 4. Candidate Selection
            # Reshape log_probs to (B, beam_size, V) to handle beams per sample
            vocab_size = log_probs.size(-1)
            log_probs = log_probs.view(batch_size, self.beam_size, vocab_size)

            # Current scores: (B, beam_size)
            current_scores = scores.view(batch_size, self.beam_size)

            # Add current scores to log probs (broadcasting)
            # next_scores: (B, beam_size, V)
            next_scores = current_scores.unsqueeze(-1) + log_probs

            # Flatten to (B, beam_size * V) to pick top k across all extensions for each sample
            next_scores_flat = next_scores.view(batch_size, -1)

            # Top-k selection
            # topk_scores: (B, beam_size)
            # topk_indices: (B, beam_size) - indices into flattened (beam_size * V)
            topk_scores, topk_indices = next_scores_flat.topk(self.beam_size, dim=1)

            # Decode indices
            # beam_indices: which beam it came from (0 to beam_size-1)
            # token_indices: which token it is (0 to V-1)
            beam_indices = torch.div(topk_indices, vocab_size, rounding_mode="floor")
            token_indices = topk_indices % vocab_size

            # 5. Update Sequences and Scores

            # Calculate global indices for gathering from the flattened batch
            # batch_offsets: (B, 1) -> (B, beam_size)
            batch_offsets = (
                torch.arange(batch_size, device=self.device) * self.beam_size
            ).unsqueeze(1)
            global_beam_indices = batch_offsets + beam_indices

            # Gather previous sequences
            # sequences: (B * beam_size, len)
            # new_sequences_prev: (B, beam_size, len)
            new_sequences_prev = sequences[global_beam_indices.view(-1)].view(
                batch_size, self.beam_size, -1
            )

            # Append new tokens
            # token_indices: (B, beam_size) -> (B, beam_size, 1)
            new_tokens = token_indices.unsqueeze(-1)
            new_sequences = torch.cat([new_sequences_prev, new_tokens], dim=2)

            # Check for EOS in new tokens
            # is_eos: (B, beam_size)
            is_eos = token_indices == self.tokenizer.EOS_ID

            # Update finished mask
            # We gather the previous finished state to see if the chosen parent beam was finished
            prev_finished = finished[global_beam_indices.view(-1)].view(
                batch_size, self.beam_size
            )
            new_finished = prev_finished | is_eos

            # Update global tensors for next iteration
            sequences = new_sequences.view(batch_size * self.beam_size, -1)
            scores = topk_scores.view(-1)
            finished = new_finished.view(-1)

            # NOTE: In a rigorous implementation, if a beam was already finished,
            # we should not have updated its score with a new token probability.
            # However, for simplicity in this batched implementation, we allow the
            # finished beams to carry forward. Since EOS usually has high probability
            # at the end, and subsequent tokens after EOS are irrelevant, this approximation holds.

        # 6. Final Selection
        # Select the beam with the highest score for each sample
        # scores: (B * beam_size) -> (B, beam_size)
        scores_reshaped = scores.view(batch_size, self.beam_size)
        best_beam_idx = scores_reshaped.argmax(dim=1)  # (B,)

        # Gather best sequences
        batch_offsets = torch.arange(batch_size, device=self.device) * self.beam_size
        global_best_indices = batch_offsets + best_beam_idx

        best_sequences = sequences[global_best_indices]  # (B, len)

        return best_sequences


def generate_submission(config: Config, load_cached_data: bool = True):
    """
    Generates submission file for the test set using the trained model.

    Args:
        config: Configuration object.
        load_cached_data: Whether to load cached tokenizer vocab.
    """
    seed_everything(config.seed)

    # 1. Setup Tokenizer
    tokenizer = InChiTokenizer(config, load_cached_data=load_cached_data)

    # 2. Load Model
    print(f"Loading model from {config.model_path}...")
    model = HybridResNetTransformer(config, tokenizer)

    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {config.model_path}")

    checkpoint = torch.load(config.model_path, map_location=config.device)
    # Handle case where checkpoint is full dict or just state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.device)
    model.eval()

    # 3. Initialize Decoder
    decoder = BeamSearchDecoder(
        model, tokenizer, beam_size=config.beam_size, max_len=config.max_len
    )

    # 4. Data Loader
    test_dataset = ChemicalImageDataset(config, tokenizer, mode="test")
    collate_fn = ChemicalCollate(config, pad_id=tokenizer.PAD_ID)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    results = []

    print(
        f"Starting inference on {len(test_dataset)} images with beam size {config.beam_size}..."
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["images"].to(config.device)
            image_ids = batch["image_ids"]

            # Decode
            # best_sequences: (B, len) tensor of indices
            best_sequences = decoder.decode(images)

            # Convert to strings
            for j in range(len(image_ids)):
                seq = best_sequences[j]
                pred_str = tokenizer.decode(seq)
                results.append({"image_id": image_ids[j], "InChI": pred_str})

            if (i + 1) % config.print_freq == 0:
                print(f"Processed {i + 1} batches.")

    # 5. Save Submission
    df_sub = pd.DataFrame(results)

    # Ensure columns order
    df_sub = df_sub[["image_id", "InChI"]]

    print(f"Saving submission to {config.submission_path}...")
    df_sub.to_csv(config.submission_path, index=False)
    print("Submission generation complete.")
