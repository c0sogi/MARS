import os
import sys
import torch
import numpy as np
import pandas as pd
import gc
import importlib
from torch.utils.data import DataLoader
from nltk.metrics import edit_distance

# Import library modules
import library.config
from library.config import Config
from library.vocabulary import get_vocab
from library.dataset import TextNormalizationDataset
from library.model import Encoder, Decoder, Attention, Seq2Seq
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    # Initialize directories
    Config.setup()

    # Set seeds for reproducibility
    Config.set_seed()

    # Override Config for Fast Baseline
    # We reduce epochs to ensure execution within ~2 hours while maintaining performance
    Config.N_EPOCHS = 3
    # Cite debug_lesson_1: Explicitly update BATCH_SIZE in memory to handle persistent environment caching
    Config.BATCH_SIZE = 64

    device = Config.get_device()
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n--- Loading Data ---")

    # Load Vocabulary
    vocab = get_vocab(load_cached_data=True)

    # Load Training Data
    train_dataset = TextNormalizationDataset(
        split="train", vocab=vocab, load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=TextNormalizationDataset.collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Load Validation Data
    val_dataset = TextNormalizationDataset(
        split="val", vocab=vocab, load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=TextNormalizationDataset.collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Initializing Model ---")

    input_dim = len(vocab)
    output_dim = len(vocab)

    attn = Attention(Config.HIDDEN_DIM, Config.HIDDEN_DIM)

    enc = Encoder(
        input_dim=input_dim,
        emb_dim=Config.ENC_EMB_DIM,
        enc_hid_dim=Config.HIDDEN_DIM,
        dec_hid_dim=Config.HIDDEN_DIM,
        dropout=Config.ENC_DROPOUT,
    )

    dec = Decoder(
        output_dim=output_dim,
        emb_dim=Config.DEC_EMB_DIM,
        enc_hid_dim=Config.HIDDEN_DIM,
        dec_hid_dim=Config.HIDDEN_DIM,
        dropout=Config.DEC_DROPOUT,
        attention=attn,
    )

    model = Seq2Seq(enc, dec, device).to(device)

    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    # ==========================================
    # 4. Training
    # ==========================================
    print("\n--- Starting Training ---")

    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # ==========================================
    # 5. Validation Assessment & Failure Analysis
    # ==========================================
    print("\n--- Validation Assessment & Failure Analysis ---")

    # Load the best model saved by the trainer
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    model.eval()

    correct_count = 0
    total_count = 0

    error_magnitudes = []
    input_lengths = []

    print("Running inference on validation set...")

    with torch.no_grad():
        for batch in val_loader:
            src = batch["src"].to(device)
            src_len = batch["src_len"].to("cpu")
            tgt = batch["tgt"].to(device)
            raw_before = batch["raw_before"]

            # Predict using greedy decoding
            predictions = model.predict(src, src_len, max_len=Config.MAX_SEQ_LEN)

            # Iterate through batch
            for i in range(len(predictions)):
                # Decode prediction
                pred_indices = predictions[i].cpu().tolist()
                pred_str = vocab.decode(pred_indices, remove_special=True)

                # Decode target
                tgt_indices = tgt[i].cpu().tolist()
                tgt_str = vocab.decode(tgt_indices, remove_special=True)

                # Check accuracy
                if pred_str == tgt_str:
                    correct_count += 1
                    error_mag = 0
                else:
                    # Calculate Levenshtein distance as error magnitude
                    error_mag = edit_distance(pred_str, tgt_str)

                total_count += 1

                # Store for failure analysis
                error_magnitudes.append(error_mag)
                input_lengths.append(len(str(raw_before[i])))

    # Calculate Metric
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation
    if len(error_magnitudes) > 1:
        # Correlation between Input Length and Error Magnitude
        corr = np.corrcoef(input_lengths, error_magnitudes)[0, 1]
        print(f"Correlation between Input Length and Error Magnitude: {corr}")

        avg_error = np.mean(error_magnitudes)
        print(f"Average Edit Distance Error: {avg_error}")
    else:
        print("Insufficient data for correlation analysis.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")

    # Clear memory
    del model
    del train_loader
    del val_loader
    del trainer
    torch.cuda.empty_cache()

    # Generate submission using the library function
    # It reloads the model from disk and processes the test set
    generate_submission(load_cached_data=True)


if __name__ == "__main__":
    main()
