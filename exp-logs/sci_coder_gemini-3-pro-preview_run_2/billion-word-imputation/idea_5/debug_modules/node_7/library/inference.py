import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from library.config import Config
from library.models import LocatorModel, InfillerModel
from library.utils import setup_logger


class BeamSearchPipeline:
    """
    Implements the Probabilistic Beam-Search Cascade for inference.

    Pipeline Steps:
    1. Locator (DeBERTa-v3): Predicts Top-K likely positions for the missing word.
    2. Hypothesis Expansion: Generates K candidate sentences with <mask> tokens.
    3. Infiller (RoBERTa-Large): Predicts the missing word for each candidate.
    4. Joint Ranking: Selects the best (Position, Word) pair based on P(loc) * P(word).
    """

    def __init__(self):
        self.logger = setup_logger(
            "Inference", os.path.join(Config.WORKING_DIR, "inference.log")
        )
        self.device = Config.DEVICE

        # Load Tokenizers
        self.logger.info("Loading tokenizers...")
        self.tokenizer_loc = AutoTokenizer.from_pretrained(
            Config.LOCATOR_MODEL_NAME, use_fast=True
        )
        self.tokenizer_inf = AutoTokenizer.from_pretrained(
            Config.INFILLER_MODEL_NAME, use_fast=True
        )

        # Load Models
        self.logger.info("Loading models...")
        self.locator = LocatorModel().to(self.device)
        self.infiller = InfillerModel().to(self.device)

        # Load Weights
        if os.path.exists(Config.BEST_LOCATOR_PATH):
            self.locator.load_state_dict(
                torch.load(Config.BEST_LOCATOR_PATH, map_location=self.device)
            )
            self.logger.info(f"Loaded Locator weights from {Config.BEST_LOCATOR_PATH}")
        else:
            self.logger.warning(
                f"Locator weights not found at {Config.BEST_LOCATOR_PATH}. Using random init (expect poor performance)."
            )

        if os.path.exists(Config.BEST_INFILLER_PATH):
            # The InfillerModel wraps AutoModelForMaskedLM. The state dict keys should match.
            self.infiller.load_state_dict(
                torch.load(Config.BEST_INFILLER_PATH, map_location=self.device)
            )
            self.logger.info(
                f"Loaded Infiller weights from {Config.BEST_INFILLER_PATH}"
            )
        else:
            self.logger.warning(
                f"Infiller weights not found at {Config.BEST_INFILLER_PATH}. Using random init."
            )

        self.locator.eval()
        self.infiller.eval()

    def predict(self, test_loader):
        """
        Runs the full inference pipeline on the test set.

        Args:
            test_loader (DataLoader): DataLoader providing test samples.

        Returns:
            pd.DataFrame: DataFrame containing 'id' and 'sentence' (predicted).
        """
        results = []

        self.logger.info(f"Starting inference with Beam Width = {Config.BEAM_WIDTH}...")

        # Disable gradients for inference
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # -------------------------------------------------------
                # Stage 1: Locator (Find Top-K Gap Positions)
                # -------------------------------------------------------
                raw_texts = batch["raw_text"]
                sample_ids = batch["id"]

                # Use pre-computed inputs and offsets from DataLoader
                loc_input_ids = batch["input_ids"].to(self.device)
                loc_attention_mask = batch["attention_mask"].to(self.device)
                loc_offsets = batch["offset_mapping"].cpu().numpy()

                # Get Locator Logits
                loc_logits = self.locator(
                    loc_input_ids, loc_attention_mask
                )  # (Batch, Seq)
                loc_probs = torch.sigmoid(loc_logits)  # (Batch, Seq)

                # Beam Search: Get Top-K indices per sentence
                # We mask out padding tokens and special tokens (CLS/SEP) to avoid selecting them
                # DeBERTa special tokens: [CLS] at 0, [SEP] at end.
                # We can just rely on the model learning this, or enforce it.
                # Enforcing mask on 0 and padding helps.
                seq_len = loc_input_ids.shape[1]
                valid_mask = loc_attention_mask.clone()
                valid_mask[:, 0] = 0  # Mask CLS

                # Apply mask to probs (set invalid to -1)
                masked_probs = loc_probs.clone()
                masked_probs[valid_mask == 0] = -1.0

                topk_probs, topk_indices = torch.topk(
                    masked_probs, k=Config.BEAM_WIDTH, dim=1
                )

                topk_probs = topk_probs.cpu().numpy()
                topk_indices = topk_indices.cpu().numpy()

                # -------------------------------------------------------
                # Stage 2: Hypothesis Expansion & Infiller Scoring
                # -------------------------------------------------------

                # Optimization: Use tensor splicing to avoid costly CPU string manipulation and re-tokenization.
                # This requires Locator and Infiller to share the same tokenizer (DeBERTa-v3).

                mask_token_id = self.tokenizer_inf.mask_token_id
                pad_token_id = self.tokenizer_inf.pad_token_id

                inf_input_ids_list = []
                inf_attention_mask_list = []
                metadata_map = []

                # We iterate to construct the spliced tensors
                # While still a loop, tensor operations are faster than string decoding/encoding
                for b_idx in range(len(raw_texts)):
                    # Get original token IDs and mask
                    ids = loc_input_ids[b_idx]
                    mask = loc_attention_mask[b_idx]
                    offsets = loc_offsets[b_idx]
                    text = raw_texts[b_idx]

                    for k in range(Config.BEAM_WIDTH):
                        token_idx = int(topk_indices[b_idx, k])
                        loc_score = topk_probs[b_idx, k]

                        # Calculate character insertion index for final reconstruction
                        if token_idx >= len(offsets):
                            t_idx_safe = len(offsets) - 1
                        else:
                            t_idx_safe = token_idx

                        start, end = offsets[t_idx_safe]
                        insertion_char_idx = end if end != 0 else len(text)

                        # Tensor Splicing: Insert [MASK] after token_idx
                        # The gap is *after* token_idx, so we insert at token_idx + 1
                        split_idx = token_idx + 1

                        # Handle bounds
                        if split_idx > len(ids):
                            split_idx = len(ids)

                        # Create new tensors
                        # [0...split_idx] + [MASK] + [split_idx...]
                        # We perform this on device to avoid transfers, though list append happens on CPU

                        # Slices
                        part1 = ids[:split_idx]
                        part2 = ids[split_idx:]

                        mask_tensor = torch.tensor([mask_token_id], device=self.device)
                        new_ids = torch.cat([part1, mask_tensor, part2])

                        # Attention mask: Insert 1
                        mask_part1 = mask[:split_idx]
                        mask_part2 = mask[split_idx:]
                        one_tensor = torch.tensor([1], device=self.device)
                        new_mask = torch.cat([mask_part1, one_tensor, mask_part2])

                        # Truncate or Pad to Config.MAX_LEN
                        curr_len = new_ids.size(0)
                        if curr_len > Config.MAX_LEN:
                            new_ids = new_ids[: Config.MAX_LEN]
                            new_mask = new_mask[: Config.MAX_LEN]
                        elif curr_len < Config.MAX_LEN:
                            pad_len = Config.MAX_LEN - curr_len
                            pad_ids = torch.full(
                                (pad_len,), pad_token_id, device=self.device
                            )
                            pad_zeros = torch.zeros((pad_len,), device=self.device)
                            new_ids = torch.cat([new_ids, pad_ids])
                            new_mask = torch.cat([new_mask, pad_zeros])

                        inf_input_ids_list.append(new_ids)
                        inf_attention_mask_list.append(new_mask)

                        metadata_map.append(
                            {
                                "batch_idx": b_idx,
                                "loc_score": loc_score,
                                "insertion_idx": insertion_char_idx,
                                "original_text": text,
                                "sample_id": sample_ids[b_idx].item(),
                            }
                        )

                # Stack tensors
                inf_input_ids = torch.stack(inf_input_ids_list)
                inf_attention_mask = torch.stack(inf_attention_mask_list)

                # Run Infiller
                inf_outputs = self.infiller(inf_input_ids, inf_attention_mask)
                inf_logits = inf_outputs.logits  # (Batch*Beam, Seq, Vocab)

                # Extract prediction at <mask> position
                mask_token_id = self.tokenizer_inf.mask_token_id

                # Find mask indices
                # torch.where returns (indices_dim0, indices_dim1)
                mask_locs = (inf_input_ids == mask_token_id).nonzero()

                # We need exactly one prediction per sample.
                # If truncation removed the mask (rare), we handle it.

                # Create a tensor to store best word prob and id for each beam candidate
                beam_word_probs = torch.zeros(len(infiller_texts), device=self.device)
                beam_word_ids = torch.zeros(
                    len(infiller_texts), dtype=torch.long, device=self.device
                )

                # Map mask locations to the batch
                # mask_locs[:, 0] is the batch index in the flattened infiller batch
                # mask_locs[:, 1] is the sequence index

                # We only take the first mask occurrence per sequence if multiple exist (shouldn't happen with our construction)
                seen_seqs = set()
                for i in range(mask_locs.shape[0]):
                    seq_idx = mask_locs[i, 0].item()
                    token_pos = mask_locs[i, 1].item()

                    if seq_idx in seen_seqs:
                        continue
                    seen_seqs.add(seq_idx)

                    # Get logits for this position
                    vocab_logits = inf_logits[seq_idx, token_pos, :]
                    probs = torch.softmax(vocab_logits, dim=0)

                    max_prob, max_id = torch.max(probs, dim=0)

                    beam_word_probs[seq_idx] = max_prob
                    beam_word_ids[seq_idx] = max_id

                # -------------------------------------------------------
                # Stage 3: Joint Ranking & Reconstruction
                # -------------------------------------------------------

                # Group by original batch index to select best candidate
                # metadata_map aligns 1:1 with infiller_texts and beam_word_* tensors

                batch_candidates = (
                    {}
                )  # Map batch_idx -> list of (score, word_str, insertion_idx)

                beam_word_probs_cpu = beam_word_probs.cpu().numpy()
                # Batch decode all predicted words at once for efficiency
                predicted_words = self.tokenizer_inf.batch_decode(
                    beam_word_ids, skip_special_tokens=True
                )

                for i, meta in enumerate(metadata_map):
                    b_idx = meta["batch_idx"]
                    loc_score = meta["loc_score"]
                    word_prob = beam_word_probs_cpu[i]

                    # Joint Score
                    # We can use product of probabilities
                    joint_score = loc_score * word_prob

                    # Get decoded word
                    predicted_word = predicted_words[i].strip()

                    if b_idx not in batch_candidates:
                        batch_candidates[b_idx] = []

                    batch_candidates[b_idx].append(
                        {
                            "score": joint_score,
                            "word": predicted_word,
                            "insertion_idx": meta["insertion_idx"],
                            "original_text": meta["original_text"],
                            "sample_id": meta["sample_id"],
                        }
                    )

                # Select best for each sample in batch
                for b_idx in range(len(raw_texts)):
                    cands = batch_candidates.get(b_idx, [])
                    if not cands:
                        # Fallback if something failed (e.g. mask truncated)
                        # Append original sentence
                        results.append((sample_ids[b_idx].item(), raw_texts[b_idx]))
                        continue

                    # Sort by score descending
                    best_cand = sorted(cands, key=lambda x: x["score"], reverse=True)[0]

                    # Reconstruct Sentence
                    # text[:idx] + " " + word + text[idx:]
                    # We ensure exactly one space padding
                    orig = best_cand["original_text"]
                    idx = best_cand["insertion_idx"]
                    word = best_cand["word"]

                    # Heuristic for clean spacing:
                    # If idx is at end, add space before word.
                    # If idx is middle, ensure spaces around.
                    # Our insertion logic was: text[:idx] + " " + mask + text[idx:]
                    # So we replicate:
                    final_sent = f"{orig[:idx]} {word}{orig[idx:]}"

                    # Clean up potential double spaces
                    final_sent = " ".join(final_sent.split())

                    results.append((best_cand["sample_id"], final_sent))

        # Convert to DataFrame
        df_results = pd.DataFrame(results, columns=["id", "sentence"])
        return df_results

    def generate_submission(self, test_loader):
        """
        Generates predictions and saves them to the submission file.
        """
        df_preds = self.predict(test_loader)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV with specific quoting to match requirements
        # Requirement: id,"sentence"
        # Pandas to_csv with quoting=1 (QUOTE_ALL) gives "id","sentence"
        # We need id,"sentence". We can achieve this by manually formatting or using csv module options.
        # However, standard CSV readers handle "id" or id fine. The prompt example shows id,"sentence".
        # We will use QUOTE_NONNUMERIC (2) which quotes non-numbers. ID is int, sentence is str.
        import csv

        df_preds.to_csv(
            Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC
        )

        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Generated {len(df_preds)} predictions.")
