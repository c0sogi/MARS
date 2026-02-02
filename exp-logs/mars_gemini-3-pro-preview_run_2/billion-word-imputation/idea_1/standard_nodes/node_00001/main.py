import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from nltk.metrics import edit_distance
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.train_engine import run_training
from library.inference_engine import predict_submission, InferenceDataset
from library.model_factory import load_model_and_tokenizer


def setup_baseline_config():
    """
    Overrides Config parameters for a fast baseline run within time limits.
    """
    # Limit training steps to ensure completion within ~1 hour
    Config.MAX_STEPS = 3000
    Config.EVAL_STEPS = 1000
    Config.SAVE_STEPS = 1000
    Config.LOGGING_STEPS = 100

    # Adjust learning rate and batch size if needed (defaults in Config are generally fine)
    # Using the defaults: BS=128, LR=5e-5

    print("Configuration updated for baseline run:")
    print(f"  MAX_STEPS: {Config.MAX_STEPS}")
    print(f"  EVAL_STEPS: {Config.EVAL_STEPS}")


def perform_validation_logic(model, tokenizer, sentences, device):
    """
    Performs the sliding window inference on a list of 'corrupted' sentences
    (where one word is missing). Returns the list of predicted sentences.
    """
    mask_token_id = tokenizer.mask_token_id
    mask_token = tokenizer.mask_token

    predicted_sentences = []

    # Process one by one (or could batch, but for 1000 val samples, simple loop is fine/safer)
    # To speed up, we will batch the *candidates* for a single sentence.

    model.eval()

    for sent in sentences:
        words = sent.split()
        n_words = len(words)

        # If sentence is too short or empty, just return as is
        if n_words < 2:
            predicted_sentences.append(sent)
            continue

        # Generate candidates: insert <mask> at every index from 1 to n_words
        candidates = []
        candidate_indices = []

        # Valid insertion indices: 1 to n_words (similar to inference engine logic)
        valid_indices = range(1, n_words)

        for ins_idx in valid_indices:
            new_words = words[:ins_idx] + [mask_token] + words[ins_idx:]
            candidates.append(" ".join(new_words))
            candidate_indices.append(ins_idx)

        if not candidates:
            predicted_sentences.append(sent)
            continue

        # Create DataLoader for candidates
        dataset = InferenceDataset(candidates, tokenizer, Config.MAX_SEQ_LEN)
        loader = DataLoader(
            dataset,
            batch_size=Config.INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Avoid overhead for small batches
            pin_memory=True,
        )

        best_score = -float("inf")
        best_word = ""
        best_ins_idx = -1

        candidate_ptr = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(enabled=Config.USE_FP16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                # Move to CPU
                batch_input_ids = input_ids.detach().cpu()
                batch_logits = logits.detach().cpu()

                for b in range(input_ids.size(0)):
                    ins_idx = candidate_indices[candidate_ptr]
                    candidate_ptr += 1

                    # Find mask position
                    m_indices = (batch_input_ids[b] == mask_token_id).nonzero(
                        as_tuple=True
                    )[0]
                    if len(m_indices) == 0:
                        continue
                    m_idx = m_indices[0].item()

                    # Get prediction score
                    token_logits = batch_logits[b, m_idx, :]
                    score, pred_id = torch.max(token_logits, dim=0)

                    if score.item() > best_score:
                        best_score = score.item()
                        best_word = tokenizer.decode(
                            [pred_id.item()], skip_special_tokens=True
                        ).strip()
                        best_ins_idx = ins_idx

        # Reconstruct sentence
        if best_ins_idx != -1:
            final_words = words[:best_ins_idx] + [best_word] + words[best_ins_idx:]
            predicted_sentences.append(" ".join(final_words))
        else:
            predicted_sentences.append(sent)

    return predicted_sentences


def validate_and_analyze():
    """
    Loads validation data, corrupts it, predicts, computes Levenshtein distance,
    and performs failure analysis.
    """
    print("\nStarting Validation and Failure Analysis...")

    # 1. Load Validation Subset
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    # Sample 1000 rows for quick but representative validation
    val_sample = val_df.sample(n=1000, random_state=Config.SEED).reset_index(drop=True)

    original_sentences = val_sample["sentence"].tolist()
    corrupted_sentences = []
    dropped_info = []  # Store (word, index) for debugging if needed

    # 2. Corrupt Sentences (Simulate Test Data)
    print("Generating synthetic test cases from validation data...")
    valid_indices_mask = []  # To keep track of valid samples

    for idx, sent in enumerate(original_sentences):
        words = sent.split()
        # Task rule: removed word is never first or last.
        # Need at least 3 words to remove a middle one.
        if len(words) >= 3:
            # Pick index from 1 to len-2 (inclusive)
            remove_idx = random.randint(1, len(words) - 2)
            removed_word = words[remove_idx]

            # Create corrupted sentence
            new_words = words[:remove_idx] + words[remove_idx + 1 :]
            corrupted_sentences.append(" ".join(new_words))
            dropped_info.append((removed_word, remove_idx))
            valid_indices_mask.append(idx)
        else:
            # Cannot corrupt according to rules, skip this sample for metric calculation
            pass

    # Filter original sentences to match corrupted ones
    target_sentences = [original_sentences[i] for i in valid_indices_mask]

    print(f"Valid validation samples created: {len(corrupted_sentences)}")

    # 3. Run Inference
    device = Config.get_device()
    # Load best model
    model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(
            "Best model not found, using base model (this implies training didn't save)."
        )
        model_path = Config.MODEL_NAME

    print(f"Loading model for validation from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path)

    print("Running inference on validation set...")
    predicted_sentences = perform_validation_logic(
        model, tokenizer, corrupted_sentences, device
    )

    # 4. Compute Metric
    distances = []
    for pred, target in zip(predicted_sentences, target_sentences):
        d = edit_distance(pred, target)
        distances.append(d)

    mean_levenshtein = np.mean(distances)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mean_levenshtein}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate features
    lengths = [len(s.split()) for s in target_sentences]

    # Correlation: Error vs Sentence Length
    corr_len, _ = pearsonr(distances, lengths)
    print(
        f"Correlation between Error (Levenshtein) and Sentence Length (Words): {corr_len:.4f}"
    )

    # Check error distribution
    print(f"Mean Error: {np.mean(distances):.4f}")
    print(f"Max Error: {np.max(distances)}")
    print(
        f"Zero Error Count (Perfect Predictions): {sum(d == 0 for d in distances)} / {len(distances)}"
    )

    # Spot check
    print("\nSample Predictions:")
    for i in range(min(5, len(distances))):
        print(f"Target:    {target_sentences[i]}")
        print(f"Predicted: {predicted_sentences[i]}")
        print(f"Distance:  {distances[i]}")
        print("-" * 20)


def main():
    # 1. Setup
    setup_baseline_config()

    # 2. Train
    # We pass load_cached_data=True. The library handles checking/creating cache.
    try:
        run_training(
            epochs=Config.EPOCHS, max_steps=Config.MAX_STEPS, load_cached_data=True
        )
    except Exception as e:
        print(f"Training encountered an issue: {e}")
        # Continue to submission generation even if training fails/stops early

    # 3. Validate & Analyze
    try:
        validate_and_analyze()
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()

    # 4. Generate Submission
    try:
        predict_submission()
    except Exception as e:
        print(f"Submission generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
