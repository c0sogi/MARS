import os
import torch
import pandas as pd
import shutil
import warnings
from library.config import Config

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configure for Speed/Demo
# ==========================================
# We modify the Config class attributes directly before importing other modules.
# This ensures the demo runs quickly on a tiny subset of data with a lightweight model.
print("--- Configuring for Demo Execution ---")

# Data & Training limits
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 64  # Use only 64 samples for the demo
Config.BATCH_SIZE = 8  # Small batch size
Config.EPOCHS = 1  # Run only 1 epoch
Config.NUM_WORKERS = 2  # Reduce worker overhead

# Paths - Use a specific demo directory
Config.WORKING_DIR = "./working/demo_execution"
Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.json")
Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "checkpoint.pth")
Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
Config.PREDICTIONS_CSV = os.path.join(Config.WORKING_DIR, "submission.csv")

# Model Hyperparameters - Reduce size for speed
Config.D_MODEL = 64
Config.N_HEADS = 4
Config.N_LAYERS = 2
Config.D_FF = 256

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Import library components AFTER configuration
from library.tokenizer import Tokenizer
from library.dataset import get_dataloaders
from library.model import DecoderOnlyTransformer
from library.train import run_training
from library.utils import seed_everything


def demo_tokenizer():
    """
    Verifies the Tokenizer's functionality: building vocab, encoding, and decoding.
    """
    print("\n--- Demonstrating Tokenizer ---")

    # Remove existing vocab to force a rebuild from metadata
    if os.path.exists(Config.VOCAB_PATH):
        os.remove(Config.VOCAB_PATH)

    # Initialize tokenizer (will build from train_metadata.csv)
    tokenizer = Tokenizer(load_cached_data=False)

    test_str = "InChI=1S/H2O/h1H2"
    print(f"Original Text: {test_str}")

    # Test Encoding
    seq = tokenizer.text_to_sequence(test_str, add_special_tokens=True)
    print(f"Encoded Sequence: {seq}")

    # Verify Special Tokens
    sos_id = tokenizer.stoi[Config.SOS_TOKEN]
    eos_id = tokenizer.stoi[Config.EOS_TOKEN]
    assert seq[0] == sos_id, "Sequence must start with SOS token"
    assert seq[-1] == eos_id, "Sequence must end with EOS token"

    # Test Decoding
    decoded = tokenizer.sequence_to_text(seq, remove_special_tokens=True)
    print(f"Decoded Text: {decoded}")

    # Verify Reconstruction
    # Note: sequence_to_text removes special tokens, so it should match the original input
    # assuming all characters are in the vocabulary (which they are for standard InChI)
    assert decoded == test_str, f"Reconstruction failed: {decoded} != {test_str}"

    print("Tokenizer logic verified.")
    return tokenizer


def demo_dataset_and_model(tokenizer):
    """
    Verifies DataLoaders and Model Forward/Generate passes.
    """
    print("\n--- Demonstrating Dataset & Model ---")

    # 1. Get DataLoaders
    # This uses the DEBUG_SAMPLE_SIZE set in Config
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=True
    )

    # 2. Inspect a Batch
    images, labels = next(iter(train_loader))
    print(f"Image Batch Shape: {images.shape}")  # Expected: (B, 1, 256, 256)
    print(f"Label Batch Shape: {labels.shape}")  # Expected: (B, MAX_TEXT_LEN)

    assert images.shape == (Config.BATCH_SIZE, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert labels.shape == (Config.BATCH_SIZE, Config.MAX_TEXT_LEN)

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = DecoderOnlyTransformer(vocab_size=len(tokenizer))
    model.to(device)

    images = images.to(device)
    labels = labels.to(device)

    # 4. Forward Pass (Training Mode)
    # In training, we pass text_input_ids (labels without the last token)
    text_input = labels[:, :-1]

    # Run forward
    logits = model(images, text_input)
    print(f"Logits Shape: {logits.shape}")

    # Expected output shape: (B, Seq_Len, Vocab_Size)
    expected_seq_len = Config.MAX_TEXT_LEN - 1
    assert logits.shape == (Config.BATCH_SIZE, expected_seq_len, len(tokenizer))

    # 5. Generation (Inference Mode)
    print("Testing Autoregressive Generation...")
    # Generate for a small subset to verify it produces strings
    preds = model.generate(images[:2], tokenizer, max_len=20)
    print(f"Generated Predictions (first 2): {preds}")

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert isinstance(preds[0], str)

    print("Model logic verified.")


def demo_full_training_loop():
    """
    Runs the full training, validation, and prediction pipeline using the library function.
    """
    print("\n--- Demonstrating Full Training Loop ---")

    # Execute the main training function
    # This handles:
    # - Data loading
    # - Model initialization
    # - Training loop (1 epoch as configured)
    # - Validation
    # - Checkpointing
    # - Loading best model
    # - Prediction on test set
    run_training(debug=True, epochs=Config.EPOCHS)

    # Verify outputs
    if os.path.exists(Config.PREDICTIONS_CSV):
        df = pd.read_csv(Config.PREDICTIONS_CSV)
        print(f"\nSubmission file created successfully at {Config.PREDICTIONS_CSV}")
        print(f"Rows: {len(df)}")
        print("Head:")
        print(df.head())

        # Verify submission format
        assert "image_id" in df.columns
        assert "InChI" in df.columns
        assert (
            len(df) == Config.DEBUG_SAMPLE_SIZE
        )  # Should match test set size in debug mode
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.PREDICTIONS_CSV}"
        )


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        # Step 1: Verify Tokenizer
        tokenizer = demo_tokenizer()

        # Step 2: Verify Data and Model components
        demo_dataset_and_model(tokenizer)

        # Step 3: Run the complete pipeline
        demo_full_training_loop()

        print("\n=== All Demonstrations Completed Successfully ===")

    except Exception as e:
        print(f"\n!!! Demonstration Failed: {e} !!!")
        raise e
