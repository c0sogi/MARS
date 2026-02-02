import os
import torch
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything, load_model, save_model
from library.models import LocatorModel, InfillerModel, VerifierModel
from library.data import get_dataloaders, get_test_dataloader
from library.engine import Trainer


class InferencePipeline:
    """
    Orchestrates the 3-stage inference process:
    1. Locate gap (Locator)
    2. Fill gap (Infiller)
    3. Verify coherence (Verifier)
    """

    def __init__(self, load_models=True):
        self.device = Config.DEVICE

        # Load tokenizers
        self.tokenizer_loc = AutoTokenizer.from_pretrained(
            Config.LOCATOR_MODEL_NAME, use_fast=True
        )
        self.tokenizer_inf = AutoTokenizer.from_pretrained(
            Config.INFILLER_MODEL_NAME, use_fast=True
        )
        self.tokenizer_ver = AutoTokenizer.from_pretrained(
            Config.VERIFIER_MODEL_NAME, use_fast=True
        )

        # Initialize models
        # We use pretrained=False when loading from checkpoint to save time/bandwidth
        self.locator = LocatorModel(pretrained=not load_models)
        self.infiller = InfillerModel(pretrained=not load_models)
        self.verifier = VerifierModel(pretrained=not load_models)

        if load_models:
            print("Loading model checkpoints...")
            self.locator = load_model(
                self.locator, Config.LOCATOR_CKPT_PATH, self.device
            )
            self.infiller = load_model(
                self.infiller, Config.INFILLER_CKPT_PATH, self.device
            )
            self.verifier = load_model(
                self.verifier, Config.VERIFIER_CKPT_PATH, self.device
            )

        self.locator.to(self.device)
        self.infiller.to(self.device)
        self.verifier.to(self.device)

        self.locator.eval()
        self.infiller.eval()
        self.verifier.eval()

    def run_inference(self, test_loader):
        """
        Runs the full pipeline on the test loader.
        Returns a list of (id, predicted_sentence) tuples.
        """
        results = []

        # Iterate over batches
        for batch in tqdm(test_loader, desc="Inference"):
            batch_ids = batch["id"].numpy()
            sentences = batch["sentence"]
            input_ids_loc = batch["input_ids"].to(self.device)
            attention_mask_loc = batch["attention_mask"].to(self.device)
            offset_mapping = batch["offset_mapping"]  # [B, Seq, 2]

            # -------------------------------------------------------
            # Stage 1: Locator
            # -------------------------------------------------------
            with torch.no_grad():
                loc_logits = self.locator(input_ids_loc, attention_mask_loc).logits

            # Probability of gap (class 1)
            # Shape: [B, Seq]
            loc_probs = torch.softmax(loc_logits, dim=-1)[:, :, 1]

            # Mask out padding to avoid selecting pad tokens
            loc_probs = loc_probs * attention_mask_loc

            # Get Top-K locations
            # topk_indices: [B, K]
            topk_probs, topk_indices = torch.topk(
                loc_probs, k=Config.LOCATOR_TOP_K, dim=1
            )

            # -------------------------------------------------------
            # Stage 2: Candidate Generation & Infilling
            # -------------------------------------------------------
            candidate_sentences = []
            candidate_metadata = []  # Stores (batch_idx, locator_prob)

            # Construct masked strings for Infiller
            for b_idx in range(len(sentences)):
                sent = sentences[b_idx]
                offsets = offset_mapping[b_idx]

                for k in range(Config.LOCATOR_TOP_K):
                    token_idx = topk_indices[b_idx, k].item()
                    prob = topk_probs[b_idx, k].item()

                    # Safety check for bounds
                    if token_idx >= len(offsets):
                        continue

                    # offset_mapping[token_idx] = (start_char, end_char)
                    # The Locator labels the token PRECEDING the gap.
                    # So we insert AFTER this token.
                    end_char = offsets[token_idx][1].item()

                    # Construct sentence with mask
                    # We add spaces around mask to ensure tokenization separates it
                    part_a = sent[:end_char]
                    part_b = sent[end_char:]
                    masked_sent = f"{part_a} {self.tokenizer_inf.mask_token} {part_b}"

                    candidate_sentences.append(masked_sent)
                    candidate_metadata.append((b_idx, prob))

            if not candidate_sentences:
                # Fallback: return original sentences if something fails
                for b_idx in range(len(sentences)):
                    results.append((batch_ids[b_idx], sentences[b_idx]))
                continue

            # Run Infiller in chunks to manage VRAM
            infilled_candidates = []  # Stores (predicted_word, mlm_prob)
            chunk_size = 64

            for i in range(0, len(candidate_sentences), chunk_size):
                chunk_sents = candidate_sentences[i : i + chunk_size]

                enc = self.tokenizer_inf(
                    chunk_sents,
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_LENGTH,
                    return_tensors="pt",
                )
                inp = enc["input_ids"].to(self.device)
                msk = enc["attention_mask"].to(self.device)

                with torch.no_grad():
                    mlm_logits = self.infiller(inp, msk).logits

                # Find the mask token index for each sentence in chunk
                mask_token_id = self.tokenizer_inf.mask_token_id

                for j in range(len(chunk_sents)):
                    # Find indices where input_ids == mask_token_id
                    mask_pos = (inp[j] == mask_token_id).nonzero(as_tuple=True)[0]

                    if len(mask_pos) == 0:
                        # Fallback if mask token somehow lost (truncation?)
                        infilled_candidates.append(("", 0.0))
                        continue

                    # Take the first mask occurrence
                    mask_idx = mask_pos[0].item()

                    # Get probability distribution over vocab
                    token_logits = mlm_logits[j, mask_idx, :]
                    token_probs = torch.softmax(token_logits, dim=0)

                    # Get top 1 word
                    top_token_id = torch.argmax(token_logits).item()
                    top_token_prob = token_probs[top_token_id].item()

                    word = self.tokenizer_inf.decode(
                        [top_token_id], skip_special_tokens=True
                    ).strip()
                    infilled_candidates.append((word, top_token_prob))

            # -------------------------------------------------------
            # Stage 3: Verification
            # -------------------------------------------------------
            verifier_sentences = []

            for idx, (masked_sent, (word, _)) in enumerate(
                zip(candidate_sentences, infilled_candidates)
            ):
                # Replace <mask> with the predicted word
                # Note: masked_sent contains the specific mask token string
                filled_sent = masked_sent.replace(self.tokenizer_inf.mask_token, word)
                # Clean up potential double spaces introduced by insertion
                filled_sent = " ".join(filled_sent.split())
                verifier_sentences.append(filled_sent)

            verifier_scores = []

            for i in range(0, len(verifier_sentences), chunk_size):
                chunk_sents = verifier_sentences[i : i + chunk_size]

                enc = self.tokenizer_ver(
                    chunk_sents,
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_LENGTH,
                    return_tensors="pt",
                )
                inp = enc["input_ids"].to(self.device)
                msk = enc["attention_mask"].to(self.device)

                with torch.no_grad():
                    ver_logits = self.verifier(inp, msk).logits

                # Probability of class 1 (Real/Correct)
                ver_probs = torch.softmax(ver_logits, dim=-1)[:, 1]
                verifier_scores.extend(ver_probs.cpu().tolist())

            # -------------------------------------------------------
            # Stage 4: Ranking & Selection
            # -------------------------------------------------------
            # Re-group candidates by original batch index
            batch_candidates = [[] for _ in range(len(sentences))]

            for k in range(len(candidate_metadata)):
                b_idx, loc_prob = candidate_metadata[k]
                word, mlm_prob = infilled_candidates[k]
                ver_prob = verifier_scores[k]
                final_sent = verifier_sentences[k]

                # Joint Score
                # S = log(P_loc) + log(P_mlm) + lambda * log(P_ver)
                epsilon = 1e-9
                score = (
                    np.log(loc_prob + epsilon)
                    + np.log(mlm_prob + epsilon)
                    + Config.VERIFIER_LAMBDA * np.log(ver_prob + epsilon)
                )

                batch_candidates[b_idx].append((score, final_sent))

            # Select best candidate for each sentence in batch
            for b_idx in range(len(sentences)):
                cands = batch_candidates[b_idx]
                if not cands:
                    results.append((batch_ids[b_idx], sentences[b_idx]))
                else:
                    # Sort by score descending
                    cands.sort(key=lambda x: x[0], reverse=True)
                    best_sent = cands[0][1]
                    results.append((batch_ids[b_idx], best_sent))

        return results


def train_pipeline():
    """
    Orchestrates the training of all three models.
    """
    seed_everything()
    print("Starting Training Pipeline...")

    # Get DataLoaders (cached or generated)
    dataloaders = get_dataloaders(load_cached_data=True)

    # 1. Train Locator
    print("\n=== Stage 1: Training Locator ===")
    if os.path.exists(Config.LOCATOR_CKPT_PATH):
        print(f"Checkpoint found at {Config.LOCATOR_CKPT_PATH}. Skipping training.")
    else:
        locator = LocatorModel(pretrained=True)
        train_loader, val_loader = dataloaders["locator"]
        Trainer.train_locator(locator, train_loader, val_loader)
        # Clear memory
        del locator, train_loader, val_loader
        torch.cuda.empty_cache()

    # 2. Train Infiller
    print("\n=== Stage 2: Training Infiller ===")
    if os.path.exists(Config.INFILLER_CKPT_PATH):
        print(f"Checkpoint found at {Config.INFILLER_CKPT_PATH}. Skipping training.")
    else:
        infiller = InfillerModel(pretrained=True)
        train_loader, val_loader = dataloaders["infiller"]
        Trainer.train_infiller(infiller, train_loader, val_loader)
        del infiller, train_loader, val_loader
        torch.cuda.empty_cache()

    # 3. Train Verifier
    print("\n=== Stage 3: Training Verifier ===")
    if os.path.exists(Config.VERIFIER_CKPT_PATH):
        print(f"Checkpoint found at {Config.VERIFIER_CKPT_PATH}. Skipping training.")
    else:
        verifier = VerifierModel(pretrained=True)
        train_loader, val_loader = dataloaders["verifier"]
        Trainer.train_verifier(verifier, train_loader, val_loader)
        del verifier, train_loader, val_loader
        torch.cuda.empty_cache()

    print("\nAll models trained successfully.")


def generate_submission():
    """
    Main entry point for generating the submission file.
    Checks for models, trains if necessary, runs inference, and saves CSV.
    """
    seed_everything()

    # Ensure models exist
    if not (
        os.path.exists(Config.LOCATOR_CKPT_PATH)
        and os.path.exists(Config.INFILLER_CKPT_PATH)
        and os.path.exists(Config.VERIFIER_CKPT_PATH)
    ):
        print("One or more model checkpoints missing. Initiating training...")
        train_pipeline()

    print("\n=== Generating Submission ===")

    # Initialize Pipeline
    pipeline = InferencePipeline(load_models=True)

    # Get Test Data
    test_loader = get_test_dataloader()

    # Run Inference
    predictions = pipeline.run_inference(test_loader)

    # Save to CSV
    # Format: id,"sentence"
    # We use csv.QUOTE_NONNUMERIC to quote non-numeric fields (sentence) but not numeric (id).
    df_sub = pd.DataFrame(predictions, columns=["id", "sentence"])

    print(f"Saving {len(df_sub)} predictions to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print("Submission generation complete.")
