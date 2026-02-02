import os
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HybridCTCAttentionModel
from library.tokenizer import InChiTokenizer


class BeamSearchDecoder:
    def __init__(self, model, tokenizer, beam_width=3, max_len=Config.MAX_SEQ_LEN):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_len = max_len
        self.device = next(model.parameters()).device
        self.sos_idx = tokenizer.sos_idx
        self.eos_idx = tokenizer.eos_idx

    def decode_batch(self, images):
        """
        Performs batched beam search decoding.
        Args:
            images: (Batch, C, H, W) tensor
        Returns:
            list of strings (decoded InChI)
        """
        batch_size = images.size(0)

        # 1. Encode images
        with torch.no_grad():
            # (B, T, Enc_Dim)
            features = self.model.backbone(images)
            features = self.model.feature_projection(features)
            features = self.model.pos_encoder(features)
            memory = self.model.transformer_encoder(features)

        # 2. Expand memory for beam search
        # Shape: (B * Beam, T, Dim)
        memory = memory.repeat_interleave(self.beam_width, dim=0)

        # 3. Initialize beams
        # sequences: (B * Beam, 1) containing SOS
        input_seq = torch.full(
            (batch_size * self.beam_width, 1),
            self.sos_idx,
            dtype=torch.long,
            device=self.device,
        )

        # scores: (B, Beam)
        # Initialize first beam with 0, others with -inf to force selection of the first beam path initially
        beam_scores = torch.full(
            (batch_size, self.beam_width), float("-inf"), device=self.device
        )
        beam_scores[:, 0] = 0.0

        # Flatten scores for easier manipulation: (B * Beam)
        beam_scores = beam_scores.view(-1)

        # Keep track of finished sequences
        # (B * Beam)
        is_finished = torch.zeros(
            batch_size * self.beam_width, dtype=torch.bool, device=self.device
        )

        # 4. Autoregressive Loop
        for step in range(self.max_len):
            # If all beams for all batch items are finished, stop
            if is_finished.all():
                break

            # Prepare Decoder Input
            # Embed current sequences
            tgt_emb = self.model.embedding(input_seq)
            tgt_emb = self.model.pos_decoder(tgt_emb)

            # Causal Mask
            seq_len = input_seq.size(1)
            causal_mask = self.model.make_causal_mask(seq_len).to(self.device)

            # Run Decoder
            # Note: We must pass the full sequence history because nn.TransformerDecoder doesn't expose KV-cache state easily
            decoder_output = self.model.transformer_decoder(
                tgt=tgt_emb, memory=memory, tgt_mask=causal_mask
            )

            # Get logits for the last token: (B * Beam, Vocab)
            logits = self.model.attention_head(decoder_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # Add previous scores
            # (B * Beam, Vocab)
            candidate_scores = beam_scores.unsqueeze(1) + log_probs

            # Mask out finished beams (force them to pick padding or similar, effectively stopping score updates)
            # However, standard practice is to just not update them.
            # Here we let them continue but we will finalize results later.
            # To simplify vectorized logic: we reshape to (B, Beam * Vocab) and pick top k

            # Reshape to (B, Beam * Vocab)
            vocab_size = log_probs.size(1)
            candidate_scores = candidate_scores.view(batch_size, -1)

            # Select Top-K for each batch item
            # topk_scores: (B, Beam)
            # topk_indices: (B, Beam) -> indices into the flattened (Beam * Vocab) array
            topk_scores, topk_indices = candidate_scores.topk(self.beam_width, dim=1)

            # Resolve indices
            # Which beam did this come from?
            beam_indices = topk_indices // vocab_size  # (B, Beam)
            # Which token is this?
            token_indices = topk_indices % vocab_size  # (B, Beam)

            # Calculate global indices to gather from the flattened batch dimension
            # We need to select the correct history from input_seq.
            # input_seq is (B*Beam, L). We need to permute it based on beam_indices.

            # Create batch offsets: [0, Beam, 2*Beam, ...]
            batch_offsets = (
                torch.arange(batch_size, device=self.device) * self.beam_width
            ).unsqueeze(1)

            # Absolute indices in the flattened batch
            gather_indices = batch_offsets + beam_indices  # (B, Beam)
            gather_indices = gather_indices.view(-1)  # (B * Beam)

            # Select history
            input_seq = input_seq[gather_indices]

            # Append new tokens
            new_tokens = token_indices.view(-1, 1)  # (B * Beam, 1)
            input_seq = torch.cat([input_seq, new_tokens], dim=1)

            # Update scores
            beam_scores = topk_scores.view(-1)

            # Update finished status
            # A sequence is finished if it was already finished OR if the new token is EOS
            previously_finished = is_finished[gather_indices]
            newly_finished = new_tokens.squeeze(1) == self.eos_idx
            is_finished = previously_finished | newly_finished

            # If a sequence was already finished, we technically shouldn't have updated its score/token
            # with log_probs, but usually we handle this by post-processing or masking.
            # For simplicity in this implementation, we allow the EOS to be generated again or
            # just take the top-1 at the end.

        # 5. Final Selection
        # Reshape to (B, Beam, L)
        final_sequences = input_seq.view(batch_size, self.beam_width, -1)
        final_scores = beam_scores.view(batch_size, self.beam_width)

        # Select best beam (index 0 is highest score because topk sorts)
        best_sequences = final_sequences[:, 0, :]

        # Decode to strings
        decoded_strings = []
        for seq in best_sequences:
            decoded_strings.append(self.tokenizer.sequence_to_text(seq))

        return decoded_strings


def generate_submission(
    checkpoint_path=Config.CHECKPOINT_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    beam_width=3,
    debug=False,
):
    """
    Runs inference on the test set and generates a submission file.
    """
    print(f"Starting inference with checkpoint: {checkpoint_path}")
    print(f"Batch size: {batch_size}, Beam width: {beam_width}, Debug: {debug}")

    # 1. Setup
    device = torch.device(Config.DEVICE)

    # Load Data
    # We use get_dataloaders to get the test loader.
    # Note: get_dataloaders returns (train, val, test, tokenizer)
    _, _, test_loader, tokenizer = get_dataloaders(debug=debug)

    # 2. Load Model
    model = HybridCTCAttentionModel().to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights (expect poor results)."
        )

    model.eval()

    # 3. Initialize Decoder
    decoder = BeamSearchDecoder(model, tokenizer, beam_width=beam_width)

    # 4. Inference Loop
    results = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # collate_fn returns (images, image_ids) for test set
            images, image_ids = batch
            images = images.to(device)

            # Run Beam Search
            predictions = decoder.decode_batch(images)

            # Store results
            for img_id, pred in zip(image_ids, predictions):
                results.append({"image_id": img_id, "InChI": pred})

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} batches...")

    # 5. Save Submission
    df_sub = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(df_sub)}")
    print(df_sub.head())
