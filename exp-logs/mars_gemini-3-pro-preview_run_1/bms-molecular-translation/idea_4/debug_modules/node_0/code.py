import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import parse_inchi_attributes, compute_levenshtein
from library.data import Tokenizer, get_dataloaders, set_seed
from library.model import AttributeConditionedModel
from library.train import train_one_epoch, validate, generate_submission


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Patch Configuration for Speed
    # -------------------------------------------------------------------------
    print("--- 1. Patching Configuration ---")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.EMBED_DIM = 64  # Reduce model size for speed
    Config.HIDDEN_DIM = 128
    # Note: Config.ENCODER_OUT_DIM remains 576 (MobileNetV3-Small fixed output)

    print(
        f"Config patched: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Embed Dim={Config.EMBED_DIM}"
    )

    # -------------------------------------------------------------------------
    # 2. Test Utilities
    # -------------------------------------------------------------------------
    print("\n--- 2. Testing Utilities ---")
    # Test Levenshtein
    s1, s2 = "InChI=1S/H2O", "InChI=1S/H2O2"
    dist = compute_levenshtein(s1, s2)
    print(f"Levenshtein distance between '{s1}' and '{s2}': {dist}")
    assert dist == 1, "Levenshtein distance calculation incorrect"

    # Test Attribute Parsing
    # Example: InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3
    # Atoms: C=2, H=6, O=1. Length=33.
    test_inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    attrs = parse_inchi_attributes(test_inchi)
    print(f"Parsed attributes for '{test_inchi}': {attrs}")

    # Config.ATOM_KEYS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]
    # Indices: C=0, H=1, O=2.
    assert attrs[0] == 2.0, "Carbon count incorrect"
    assert attrs[1] == 6.0, "Hydrogen count incorrect"
    assert attrs[2] == 1.0, "Oxygen count incorrect"
    assert attrs[-1] == len(test_inchi), "Length attribute incorrect"
    print("Utility tests passed.")

    # -------------------------------------------------------------------------
    # 3. Test Tokenizer
    # -------------------------------------------------------------------------
    print("\n--- 3. Testing Tokenizer ---")
    tokenizer = Tokenizer()
    seq = tokenizer.text_to_sequence(test_inchi)
    print(f"Tokenized sequence shape: {seq.shape}")
    decoded_text = tokenizer.sequence_to_text(seq)
    print(f"Decoded text: {decoded_text}")
    assert decoded_text == test_inchi, "Tokenizer round-trip failed"
    print("Tokenizer tests passed.")

    # -------------------------------------------------------------------------
    # 4. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- 4. Loading Data (Debug Mode) ---")
    # Use a small debug size to create loaders quickly
    debug_size = 50
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        load_cached_data=False, debug_size=debug_size
    )

    # Fetch a batch
    images, target_seqs, target_attrs = next(iter(train_loader))
    print(
        f"Batch shapes - Images: {images.shape}, Seqs: {target_seqs.shape}, Attrs: {target_attrs.shape}"
    )

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        256,
        256,
    ), "Image batch shape incorrect"
    assert target_seqs.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
    ), "Sequence batch shape incorrect"
    assert target_attrs.shape == (
        Config.BATCH_SIZE,
        Config.ATTRIBUTE_DIM,
    ), "Attribute batch shape incorrect"
    print("Data loading passed.")

    # -------------------------------------------------------------------------
    # 5. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- 5. Model Instantiation & Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AttributeConditionedModel().to(device)
    images = images.to(device)
    target_seqs = target_seqs.to(device)

    # Forward pass (Teacher Forcing)
    seq_logits, pred_attrs = model(images, target_seqs)

    print(f"Logits shape: {seq_logits.shape}")
    print(f"Predicted attributes shape: {pred_attrs.shape}")

    # Expected logits shape: (Batch, Seq_Len - 1, Vocab) because of teacher forcing input shifting
    # Note: The model implementation takes target_seqs[:, :-1] as input.
    expected_seq_len = Config.MAX_SEQ_LEN - 1
    assert seq_logits.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
        Config.VOCAB_SIZE,
    ), "Logits shape incorrect"
    assert pred_attrs.shape == (
        Config.BATCH_SIZE,
        Config.ATTRIBUTE_DIM,
    ), "Predicted attributes shape incorrect"
    print("Forward pass passed.")

    # -------------------------------------------------------------------------
    # 6. Inference (Greedy Decoding)
    # -------------------------------------------------------------------------
    print("\n--- 6. Testing Inference (Predict) ---")
    pred_seqs = model.predict(images, device=device)
    print(f"Predicted sequences shape: {pred_seqs.shape}")

    assert pred_seqs.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
    ), "Prediction shape incorrect"

    # Decode one prediction
    pred_text = tokenizer.sequence_to_text(pred_seqs[0].cpu().numpy())
    print(f"Sample prediction (untrained): {pred_text}")
    print("Inference passed.")

    # -------------------------------------------------------------------------
    # 7. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 7. Training One Epoch ---")
    criterion_seq = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    criterion_attr = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss, train_seq, train_attr = train_one_epoch(
        train_loader, model, criterion_seq, criterion_attr, optimizer, device, epoch=0
    )
    print(
        f"Epoch finished. Total Loss: {train_loss:.4f}, Seq Loss: {train_seq:.4f}, Attr Loss: {train_attr:.4f}"
    )
    assert not np.isnan(train_loss), "Training loss is NaN"
    print("Training step passed.")

    # -------------------------------------------------------------------------
    # 8. Validation Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 8. Validation Step ---")
    val_loss, val_lev = validate(
        val_loader, model, criterion_seq, criterion_attr, tokenizer, device
    )
    print(f"Validation finished. Loss: {val_loss:.4f}, Levenshtein: {val_lev:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    print("Validation step passed.")

    # -------------------------------------------------------------------------
    # 9. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- 9. Generating Submission ---")
    # We use the trained model (even if trained for just 1 epoch on 50 items)
    generate_submission(test_loader, model, tokenizer, device)

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created. Shape: {df_sub.shape}")
        print(f"First few rows:\n{df_sub.head()}")
        assert df_sub.shape[0] == len(
            test_loader.dataset
        ), "Submission row count mismatch"
        # Verify columns
        assert (
            "image_id" in df_sub.columns and "InChI" in df_sub.columns
        ), "Submission columns incorrect"
    else:
        raise FileNotFoundError("Submission file was not created.")
    print("Submission generation passed.")

    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    run_demo()
