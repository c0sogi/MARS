import os
import torch
import pandas as pd
import csv
from torch.utils.data import DataLoader

from library.config import Config
from library.vocabulary import WordVocabulary
from library.dataset import InfillingDataset, collate_fn
from library.model import DualHeadTransformer
from library.utils import set_seed, load_checkpoint, insert_word_in_sentence


def generate_submission(debug=Config.DEBUG, batch_size=Config.VAL_BATCH_SIZE):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug (bool): If True, runs on a small subset of data.
        batch_size (int): Batch size for inference.
    """
    # 1. Setup
    # Update Config debug state to ensure Dataset respects it
    Config.DEBUG = debug
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting inference on device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Load Vocabulary
    # We assume the vocabulary was already built during training.
    # If not found, this will raise an error (as expected for inference).
    vocab = WordVocabulary()
    vocab.load(Config.TARGET_VOCAB_PATH)
    vocab_size = len(vocab)
    print(f"Vocabulary loaded. Size: {vocab_size}")

    # 3. Load Test Data
    print("Initializing Test Dataset...")
    test_dataset = InfillingDataset(split="test", vocabulary=vocab)

    # We need the tokenizer for reconstruction
    tokenizer = test_dataset.tokenizer

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Load Model
    print("Initializing Model...")
    model = DualHeadTransformer(vocab_size=vocab_size)
    model.to(device)

    # Load weights
    print(f"Loading checkpoint from {Config.MODEL_SAVE_PATH}...")
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model.eval()

    # 5. Inference Loop
    results_ids = []
    results_sentences = []

    print("Running Inference...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Metadata (kept on CPU)
            batch_ids = batch["id"]
            original_texts = batch["original_text"]

            # Forward pass
            # loc_logits: (Batch, Seq_Len)
            # word_logits: (Batch, Seq_Len, Vocab_Size)
            loc_logits, word_logits = model(input_ids, attention_mask)

            # --- Decoding ---

            # 1. Find best location (argmax over sequence length)
            # Shape: (Batch,)
            pred_loc_indices = torch.argmax(loc_logits, dim=1)

            # 2. Find best word at that location
            # Gather word logits for the predicted positions
            # We need to index [batch_i, loc_i, :]
            batch_indices = torch.arange(input_ids.size(0), device=device)

            # Shape: (Batch, Vocab_Size)
            target_word_logits = word_logits[batch_indices, pred_loc_indices, :]

            # Shape: (Batch,)
            pred_word_ids = torch.argmax(target_word_logits, dim=1)

            # Move to CPU for reconstruction
            pred_loc_indices = pred_loc_indices.cpu().numpy()
            pred_word_ids = pred_word_ids.cpu().numpy()

            # --- Reconstruction ---
            for i in range(len(batch_ids)):
                sent_id = batch_ids[i]
                original_text = original_texts[i]
                loc_idx = pred_loc_indices[i]
                word_id = pred_word_ids[i]

                # Convert word ID to string
                predicted_word = vocab.id_to_token(int(word_id))

                # Handle special tokens if predicted (unlikely but possible)
                if predicted_word in [vocab.pad_token, vocab.unk_token]:
                    # Fallback strategy: if model predicts UNK/PAD,
                    # we might want to insert a generic word or just skip.
                    # Given the task, we must insert *something*.
                    # 'the' is a safe statistical bet if UNK is predicted.
                    if predicted_word == vocab.unk_token:
                        predicted_word = "the"
                    else:
                        predicted_word = ""  # Should not happen for PAD usually

                # Reconstruct sentence
                # Note: insert_word_in_sentence expects the token index relative to the tokenizer's encoding
                final_sentence = insert_word_in_sentence(
                    original_text, predicted_word, loc_idx, tokenizer
                )

                results_ids.append(sent_id)
                results_sentences.append(final_sentence)

    # 6. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")

    df_submission = pd.DataFrame({"id": results_ids, "sentence": results_sentences})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    # Requirement: id,"sentence"
    # quoting=csv.QUOTE_NONNUMERIC will quote non-numeric fields (strings) and leave numbers unquoted.
    # This matches the format: 1,"Sentence text..."
    df_submission.to_csv(
        Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC
    )

    print("Submission generation complete.")
