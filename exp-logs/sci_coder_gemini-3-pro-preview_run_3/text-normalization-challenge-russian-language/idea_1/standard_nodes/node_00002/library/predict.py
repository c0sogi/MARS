import os
import torch
import pandas as pd
import numpy as np
from library.config import Config, set_seed
from library.vocabulary import CharVocab
from library.model import Encoder, Decoder, Seq2Seq, Attention
from library.dataset import get_dataloader
from library.utils import load_checkpoint


def generate_submission(load_cached_data=True, batch_size=Config.batch_size):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached data for vocabulary and dataset.
        batch_size (int): Batch size for inference.
    """
    # 1. Setup
    set_seed(42)
    device = Config.device
    print(f"Inference using device: {device}")

    # 2. Load Vocabulary
    # We assume the vocabulary was already built during training.
    # If not, we try to build it from train data.
    vocab = CharVocab()
    vocab.build_vocab(Config.TRAIN_DATA_PATH, load_cached_data=load_cached_data)
    print(f"Vocabulary loaded. Size: {len(vocab)}")

    # 3. Initialize Model
    # Must match training configuration
    attn = Attention(Config.hidden_dim, Config.hidden_dim)

    enc = Encoder(
        input_dim=len(vocab),
        emb_dim=Config.embed_dim,
        hid_dim=Config.hidden_dim,
        n_layers=Config.n_layers,
        dropout=Config.dropout,
    )
    dec = Decoder(
        output_dim=len(vocab),
        emb_dim=Config.embed_dim,
        enc_hid_dim=Config.hidden_dim,
        dec_hid_dim=Config.hidden_dim,
        n_layers=Config.n_layers,
        dropout=Config.dropout,
        attention=attn,
    )
    model = Seq2Seq(enc, dec, device).to(device)

    # 4. Load Checkpoint
    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print(
            f"Warning: No checkpoint found at {checkpoint_path}. Using random weights (expect poor performance)."
        )

    model.eval()

    # 5. Load Test Data
    print("Loading test dataset...")
    # We use the metadata/test.csv path defined in Config
    test_loader = get_dataloader(
        data_path=Config.TEST_DATA_PATH,
        vocab=vocab,
        batch_size=batch_size,
        is_test=True,
        shuffle=False,  # Order is preserved implicitly by ID mapping, but False is safer/cleaner
        load_cached_data=load_cached_data,
    )

    # 6. Inference Loop
    results_id = []
    results_text = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            ids = batch["id"]
            src = batch["src"].to(device)

            # Predict using greedy decoding
            # Returns tensor [batch_size, max_len]
            preds = model.predict(
                src,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                max_len=Config.max_len,
            )

            # Decode predictions
            # Iterate over the batch to decode each sequence individually
            for i in range(len(ids)):
                pred_indices = preds[i]
                decoded_str = vocab.decode(pred_indices, remove_special_tokens=True)

                results_id.append(ids[i])
                results_text.append(decoded_str)

    # 7. Create Submission DataFrame
    df_submission = pd.DataFrame({"id": results_id, "after": results_text})

    # 8. Save Submission
    save_path = Config.SUBMISSION_PATH
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving submission to {save_path}...")
    df_submission.to_csv(save_path, index=False)

    print("Submission generation complete.")
