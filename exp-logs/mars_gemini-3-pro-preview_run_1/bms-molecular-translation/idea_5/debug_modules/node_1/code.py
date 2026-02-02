import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import (
    Tokenizer,
    parse_inchi_attributes,
    AttributeNormalizer,
    compute_levenshtein,
)
from library.dataset import prepare_datasets, collate_fn, get_transforms
from library.model import AttributeAugmentedAttnNet
from library.engine import run_training, generate_predictions, set_seed


def main():
    print("Initializing Demonstration Script...")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # --------------------------------------------------------------------------
    config = Config()

    # Override paths for the demo to keep things isolated
    config.WORKING_DIR = "./working/demo_run"
    config.SUBMISSION_DIR = "./working/demo_submission"
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Update file paths in config based on new directories
    config.TOKENIZER_PATH = os.path.join(config.WORKING_DIR, "tokenizer.json")
    config.MODEL_PATH = os.path.join(config.WORKING_DIR, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    config.TRAIN_ATTR_CACHE = os.path.join(config.WORKING_DIR, "train_attributes.npy")
    config.VAL_ATTR_CACHE = os.path.join(config.WORKING_DIR, "val_attributes.npy")
    config.ATTR_STATS_CACHE = os.path.join(config.WORKING_DIR, "attr_stats.npy")

    # Set optimization flags for speed
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 32  # Small subset for quick verification
    config.BATCH_SIZE = 8
    config.EPOCHS = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    config.ENCODER_NAME = "resnet18"  # Use a lighter backbone for demo speed if timm supports it (efficientnet_b0 is default, but resnet18 is often faster to load/run on cpu)
    # Revert to default encoder if resnet18 causes issues, but efficientnet_b0 is fine.
    # Let's keep efficientnet_b0 as defined in Config to ensure we test the actual code provided.
    config.ENCODER_NAME = "efficientnet_b0"

    set_seed(config.SEED)
    print("Configuration configured for fast execution.")

    # --------------------------------------------------------------------------
    # 2. Verify Utilities
    # --------------------------------------------------------------------------
    print("\n--- Verifying Utilities ---")

    # Test parse_inchi_attributes
    test_inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    # TRACKED_ATOMS = ["C", "H", "O", "N", "S", "F", "Cl", "Br", "I"]
    # C=2, H=6, O=1. Length = len(test_inchi)
    attrs = parse_inchi_attributes(test_inchi, config.TRACKED_ATOMS)

    # Indices: C=0, H=1, O=2.
    assert attrs[0] == 2.0, f"Expected 2 Carbons, got {attrs[0]}"
    assert attrs[1] == 6.0, f"Expected 6 Hydrogens, got {attrs[1]}"
    assert attrs[2] == 1.0, f"Expected 1 Oxygen, got {attrs[2]}"
    assert attrs[-1] == len(test_inchi), "Sequence length attribute mismatch"
    print("parse_inchi_attributes: OK")

    # Test Tokenizer
    tokenizer = Tokenizer(config)
    dummy_texts = ["InChI=1S/H2O", "InChI=1S/CH4"]
    tokenizer.fit_on_texts(dummy_texts)

    seq = tokenizer.text_to_sequence("H2O")
    decoded = tokenizer.sequence_to_text(seq)
    # Note: text_to_sequence adds SOS/EOS. sequence_to_text removes them.
    assert decoded == "H2O", f"Tokenizer decode failed. Got {decoded}"
    print("Tokenizer: OK")

    # Test Levenshtein
    dist = compute_levenshtein(["kitten"], ["sitting"])
    assert dist == 3, f"Levenshtein distance incorrect. Expected 3, got {dist}"
    print("compute_levenshtein: OK")

    # --------------------------------------------------------------------------
    # 3. Verify Dataset Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Verifying Dataset Pipeline ---")

    # This will load metadata, fit tokenizer/normalizer on the debug subset (implicitly via prepare_datasets logic)
    # Note: prepare_datasets loads full metadata then passes to Dataset class.
    # Dataset class handles DEBUG slicing.
    train_ds, val_ds, test_ds, tokenizer = prepare_datasets(
        config, load_cached_data=False
    )

    print(f"Train Dataset Length (Debug): {len(train_ds)}")
    assert (
        len(train_ds) == config.DEBUG_SAMPLE_SIZE
    ), "Dataset did not respect DEBUG_SAMPLE_SIZE"

    # Fetch one sample
    sample = train_ds[0]
    assert "image" in sample
    assert "token_ids" in sample
    assert "attributes" in sample
    assert sample["image"].shape == (
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    ), "Image shape mismatch"
    print("Single sample fetch: OK")

    # Test DataLoader and Collate
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0
    )

    batch = next(iter(train_loader))
    images = batch["image"]
    token_ids = batch["token_ids"]
    attributes = batch["attributes"]

    assert images.shape[0] == config.BATCH_SIZE
    assert images.shape[1] == 3
    assert attributes.shape == (config.BATCH_SIZE, config.NUM_ATTRIBUTES)
    # token_ids shape is (B, max_seq_len_in_batch)
    assert token_ids.dim() == 2
    print("DataLoader batch generation: OK")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    model = AttributeAugmentedAttnNet(config, tokenizer.vocab_size)
    model.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)
    token_ids = token_ids.to(config.DEVICE)

    # Forward pass
    seq_logits, attr_pred = model(images, targets=token_ids, teacher_forcing_ratio=0.0)

    # Check shapes
    # seq_logits: (B, seq_len, vocab_size)
    # attr_pred: (B, num_attributes)
    assert seq_logits.shape[0] == config.BATCH_SIZE
    assert seq_logits.shape[2] == tokenizer.vocab_size
    assert attr_pred.shape == (config.BATCH_SIZE, config.NUM_ATTRIBUTES)

    print("Model forward pass: OK")

    # --------------------------------------------------------------------------
    # 5. Verify Training Loop (Engine)
    # --------------------------------------------------------------------------
    print("\n--- Verifying Training Loop ---")

    # run_training handles the loop. Since we set EPOCHS=1 and DEBUG=True, this should be fast.
    run_training(config)

    if os.path.exists(config.MODEL_PATH):
        print("Training completed and model checkpoint saved.")
    else:
        raise AssertionError("Model checkpoint was not created after training.")

    # --------------------------------------------------------------------------
    # 6. Verify Inference (Engine)
    # --------------------------------------------------------------------------
    print("\n--- Verifying Inference ---")

    generate_predictions(config)

    if os.path.exists(config.SUBMISSION_PATH):
        print("Inference completed and submission file created.")

        # Validate submission format
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        assert "image_id" in df_sub.columns
        assert "InChI" in df_sub.columns
        assert len(df_sub) > 0
        print(f"Submission file format verified. Rows: {len(df_sub)}")
    else:
        raise AssertionError("Submission file was not created.")

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
