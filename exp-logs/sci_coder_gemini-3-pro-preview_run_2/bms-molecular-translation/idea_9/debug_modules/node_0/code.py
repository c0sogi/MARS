import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.tokenizer import InChiTokenizer
from library.dataset import get_dataloaders, collate_fn
from library.model import HybridCTCAttentionModel
from library.loss import HybridLoss
from library.trainer import Trainer
from library.inference import BeamSearchDecoder


def main():
    print("=" * 60)
    print("InChI Recognition Library Demonstration")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Configuration Setup (Optimized for Speed)
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config for a quick demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Very small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.VOCAB_CACHE_PATH = os.path.join(Config.WORKING_DIR, "vocab.npy")

    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)
    print("Configuration configured for debug mode.")

    # -------------------------------------------------------------------------
    # 2. Tokenizer Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Tokenizer...")

    # Initialize tokenizer (will build/load vocab from metadata)
    tokenizer = InChiTokenizer(load_cached_data=True)

    # Test string
    sample_inchi = "InChI=1S/H2O/h1H2"
    print(f"Original: {sample_inchi}")

    # Encode
    seq = tokenizer.text_to_sequence(sample_inchi, add_sos=True, add_eos=True)
    print(f"Encoded Sequence: {seq}")

    # Decode
    decoded = tokenizer.sequence_to_text(seq, remove_special=True)
    print(f"Decoded: {decoded}")

    # Assertions
    assert decoded == sample_inchi, "Decoded text does not match original!"
    assert seq[0] == tokenizer.sos_idx, "Sequence must start with SOS"
    assert seq[-1] == tokenizer.eos_idx, "Sequence must end with EOS"
    print("Tokenizer logic verified.")

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoader...")

    # Get loaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(debug=True)

    # Fetch a single batch
    images, sequences, lengths = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")  # (B, 3, H, W)
    print(f"Batch Sequences Shape: {sequences.shape}")  # (B, Max_Seq_Len_In_Batch)
    print(f"Lengths: {lengths}")

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == Config.IMAGE_HEIGHT
    ), f"Image height should be {Config.IMAGE_HEIGHT}"
    assert sequences.dim() == 2, "Sequences should be 2D tensor"
    assert len(lengths) == images.shape[0], "Lengths should match batch size"
    print("DataLoader logic verified.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = HybridCTCAttentionModel().to(device)

    # Move batch to device
    images = images.to(device)
    sequences = sequences.to(device)

    # Forward pass
    ctc_logits, attn_logits = model(images, sequences)

    print(f"CTC Logits Shape: {ctc_logits.shape}")  # (B, T_enc, Vocab)
    print(f"Attn Logits Shape: {attn_logits.shape}")  # (B, Seq_Len, Vocab)

    # Assertions
    # Encoder output length depends on width. ResNet34 downsamples H by 32, W by 2 (Anisotropic).
    # Input width W -> Feature width W/2.
    expected_enc_len = images.shape[3] // 2
    assert (
        ctc_logits.shape[1] == expected_enc_len
    ), f"Expected CTC len {expected_enc_len}, got {ctc_logits.shape[1]}"
    assert ctc_logits.shape[2] == Config.VOCAB_SIZE
    assert attn_logits.shape[1] == sequences.shape[1]
    assert attn_logits.shape[2] == Config.VOCAB_SIZE
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Loss Function...")

    criterion = HybridLoss().to(device)
    lengths = lengths.to(device)

    loss, metrics = criterion(ctc_logits, attn_logits, sequences, lengths)

    print(f"Total Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss > 0, "Loss should be positive"
    assert "loss_ctc" in metrics and "loss_attn" in metrics
    print("Loss calculation verified.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop Demonstration...")

    # Initialize Trainer
    trainer = Trainer(debug=True)

    # We force the trainer to use our already initialized (and small) config
    # The trainer internally calls get_dataloaders(debug=True) which uses Config.DEBUG_SAMPLE_SIZE

    print("Running 1 epoch of training...")
    avg_loss = trainer.train_epoch(epoch=0)

    print(f"Epoch finished. Average Loss: {avg_loss:.4f}")

    # Save a dummy checkpoint for inference testing
    torch.save(trainer.model.state_dict(), Config.CHECKPOINT_PATH)
    print(f"Saved checkpoint to {Config.CHECKPOINT_PATH}")

    # -------------------------------------------------------------------------
    # 7. Inference / Beam Search Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Inference (Beam Search)...")

    # Initialize Decoder
    decoder = BeamSearchDecoder(model, tokenizer, beam_width=2, max_len=50)

    # Run decoding on the batch we loaded earlier
    print("Decoding batch...")
    decoded_strings = decoder.decode_batch(images)

    print("Sample predictions:")
    for i, s in enumerate(decoded_strings[:2]):  # Print first 2
        print(f"  [{i}] {s}")

    assert isinstance(decoded_strings, list), "Output should be a list"
    assert isinstance(decoded_strings[0], str), "Elements should be strings"
    assert len(decoded_strings) == images.size(0), "Output length mismatch"
    print("Inference logic verified.")

    # -------------------------------------------------------------------------
    # 8. Full Submission Generation Verification
    # -------------------------------------------------------------------------
    print("\n[8] Verifying Submission Generation...")

    # We verify the Trainer's predict_test_set method which uses greedy decoding
    # This is faster than beam search for the demo
    trainer.predict_test_set()

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print(f"Rows: {len(df_sub)}")
        assert "image_id" in df_sub.columns
        assert "InChI" in df_sub.columns
        assert len(df_sub) > 0
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n" + "=" * 60)
    print("All demonstrations and verifications passed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
