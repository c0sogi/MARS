import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader

# Import from the provided library files
from library.tokenizer import Tokenizer
from library.dataset import ChemicalDataset, get_transforms
from library.model import FormulaConditionedModel
from library.trainer import Trainer

# Constants for demonstration
METADATA_DIR = "./metadata"
INPUT_ROOT = "./input"
WORKING_DIR = "./working/demo_execution"
TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

# Hyperparameters optimized for speed in this demo
BATCH_SIZE = 8
DEBUG_SIZE = 32  # Very small subset to ensure quick execution
EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_demo():
    print(f"--- Starting Demo Execution on {DEVICE} ---")
    set_seed(42)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Tokenizer Demonstration
    # ------------------------------------------------------------------------
    print("\n[1] Initializing Tokenizer...")
    # We use the training metadata to build the vocabulary.
    # The cache_dir is set to our working directory to avoid modifying input/metadata folders.
    tokenizer = Tokenizer(
        metadata_path=TRAIN_METADATA,
        load_cached_data=False,  # Force build for demo purposes
        cache_dir=WORKING_DIR,
    )
    print(f"Vocabulary size: {len(tokenizer)}")

    # Verify Tokenizer Logic
    sample_inchi = "InChI=1S/H2O/h1H2"
    encoded = tokenizer.text_to_sequence(sample_inchi)
    decoded = tokenizer.sequence_to_text(encoded)

    print(f"Sample Text: {sample_inchi}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Assertion to ensure tokenizer consistency
    # Note: text_to_sequence adds <SOS> and <EOS>, sequence_to_text stops at <EOS>
    assert decoded == sample_inchi, "Tokenizer encoding/decoding cycle failed!"
    print("Tokenizer verification passed.")

    # ------------------------------------------------------------------------
    # 2. Dataset & DataLoader Demonstration
    # ------------------------------------------------------------------------
    print("\n[2] Initializing Datasets (Debug Mode)...")

    # Transforms
    transforms = get_transforms(img_size=256)

    # Train Dataset
    train_dataset = ChemicalDataset(
        metadata_path=TRAIN_METADATA,
        tokenizer=tokenizer,
        transform=transforms,
        mode="train",
        load_cached_data=False,  # Force re-calculation for demo
        cache_dir=WORKING_DIR,
        debug_size=DEBUG_SIZE,
        input_root=INPUT_ROOT,
    )

    # Validation Dataset
    val_dataset = ChemicalDataset(
        metadata_path=VAL_METADATA,
        tokenizer=tokenizer,
        transform=transforms,
        mode="val",
        load_cached_data=False,
        cache_dir=WORKING_DIR,
        debug_size=DEBUG_SIZE,  # Keep it small
        input_root=INPUT_ROOT,
    )

    # Test Dataset
    test_dataset = ChemicalDataset(
        metadata_path=TEST_METADATA,
        tokenizer=tokenizer,
        transform=transforms,
        mode="test",
        load_cached_data=False,
        cache_dir=WORKING_DIR,
        debug_size=DEBUG_SIZE,
        input_root=INPUT_ROOT,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Verify Data Loading
    sample_batch = next(iter(train_loader))
    print(f"Batch keys: {sample_batch.keys()}")
    print(f"Image shape: {sample_batch['image'].shape}")  # Should be (B, 3, 256, 256)
    print(f"Sequence shape: {sample_batch['sequence'].shape}")  # Should be (B, SeqLen)
    print(
        f"Atom counts shape: {sample_batch['atom_counts'].shape}"
    )  # Should be (B, 12)

    assert sample_batch["image"].shape == (BATCH_SIZE, 3, 256, 256)
    assert (
        sample_batch["atom_counts"].shape[1] == 12
    )  # 12 atom types defined in utils.py
    print("Dataset verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = FormulaConditionedModel(
        vocab_size=len(tokenizer),
        embed_dim=256,
        hidden_dim=512,
        pretrained_encoder=False,  # False for speed/offline demo, True in production
    )
    model = model.to(DEVICE)
    print("Model initialized successfully.")

    # Verify Forward Pass
    images = sample_batch["image"].to(DEVICE)
    sequences = sample_batch["sequence"].to(DEVICE)

    # Forward pass returns logits and predicted atom counts
    logits, pred_atoms = model(images, sequences)

    # Logits shape: (B, SeqLen - 1, VocabSize) because of teacher forcing input shift
    expected_seq_len = sequences.shape[1] - 1
    print(f"Logits shape: {logits.shape}")
    print(f"Pred atoms shape: {pred_atoms.shape}")

    assert logits.shape == (BATCH_SIZE, expected_seq_len, len(tokenizer))
    assert pred_atoms.shape == (BATCH_SIZE, 12)
    print("Model forward pass verification passed.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Starting Training Loop (1 Epoch)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        tokenizer=tokenizer,
        checkpoint_dir=WORKING_DIR,
    )

    # Run fit
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    print("Training loop completed.")

    # ------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # ------------------------------------------------------------------------
    print("\n[5] Generating Submission on Test Set...")

    # Load best model (in this demo, it's the one we just trained/saved)
    best_model_path = os.path.join(WORKING_DIR, "model_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded best model checkpoint.")
    else:
        print(
            "Checkpoint not found (maybe validation didn't improve), using current model state."
        )

    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            image_ids = batch["image_id"]

            # Predict using greedy decoding
            predicted_inchis = model.predict(
                images, tokenizer, max_len=100, device=DEVICE
            )

            for img_id, pred in zip(image_ids, predicted_inchis):
                results.append({"image_id": img_id, "InChI": pred})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Save submission
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(f"Generated {len(submission_df)} predictions.")
    print("Sample predictions:")
    print(submission_df.head())

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
