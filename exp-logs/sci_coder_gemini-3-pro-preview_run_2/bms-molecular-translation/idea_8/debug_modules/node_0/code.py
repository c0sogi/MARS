import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2

# Ensure the current directory is in the path to import the library modules
sys.path.append(".")

from library.config import Config
from library.tokenizer import InChiTokenizer
from library.dataset import InChiDataset, CollateFn
from library.model import CNNTransformerCTC
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import calc_levenshtein


def run_demonstration():
    print("=" * 60)
    print("InChI Prediction Pipeline Demonstration")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config parameters to create a lightweight model and short training run
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Model Complexity for Demo
    Config.D_MODEL = 64
    Config.NHEAD = 2
    Config.NUM_ENCODER_LAYERS = 1
    Config.DIM_FEEDFORWARD = 128

    # Reduce Training params
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.IMG_HEIGHT = 128

    # Setup directories
    Config.setup()

    # -------------------------------------------------------------------------
    # 2. Tokenizer Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Tokenizer Logic...")
    tokenizer = InChiTokenizer()

    sample_text = "InChI=1S/C"
    print(f"Sample Text: {sample_text}")

    # Encode
    encoded = tokenizer.text_to_sequence(sample_text)
    print(f"Encoded Sequence: {encoded.tolist()}")

    # Verify indices match Config.CHAR2IDX
    expected_indices = [Config.CHAR2IDX[c] for c in sample_text]
    assert encoded.tolist() == expected_indices, "Tokenizer encoding mismatch!"

    # Decode (Simulate perfect logits)
    # Create one-hot logits for the sequence
    seq_len = len(encoded)
    vocab_size = len(Config.VOCAB)
    logits = torch.zeros(1, seq_len, vocab_size)
    for t, idx in enumerate(encoded):
        logits[0, t, idx] = 10.0  # High confidence

    decoded = tokenizer.decode_ctc_greedy(logits, batch_first=True)
    print(f"Decoded Text: {decoded[0]}")

    # Note: CTC decoding collapses repeats. "InChI=1S/C" has no repeats, so it should match exactly.
    assert decoded[0] == sample_text, "Tokenizer decoding mismatch!"
    print("Tokenizer logic verified.")

    # -------------------------------------------------------------------------
    # 3. Dataset and Collate Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and Collate Function...")

    # Load a tiny subset of training data
    train_dataset = InChiDataset(Config.TRAIN_CSV, mode="train", sample_size=16)
    print(f"Dataset size: {len(train_dataset)}")

    # Fetch one sample
    sample = train_dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample target text: {sample['target_text']}")

    assert sample["image"].shape[0] == 1, "Image should be 1-channel (grayscale)"
    assert sample["image"].shape[1] == Config.IMG_HEIGHT, "Image height mismatch"

    # Test Collate Function
    collate_fn = CollateFn()
    batch_list = [train_dataset[i] for i in range(4)]
    batch = collate_fn(batch_list)

    images = batch["images"]
    targets = batch["targets"]
    input_lengths = batch["input_lengths"]

    print(f"Batch images shape: {images.shape}")
    print(f"Batch targets shape: {targets.shape}")

    assert images.shape[0] == 4, "Batch size mismatch"
    assert images.shape[1] == 1, "Channel dim mismatch"
    assert images.shape[2] == Config.IMG_HEIGHT, "Height dim mismatch"
    # Width should be max of the sample widths
    expected_width = max([s["width"] for s in batch_list])
    assert images.shape[3] == expected_width, "Padded width mismatch"
    assert len(input_lengths) == 4, "Input lengths mismatch"

    print("Dataset and Collate logic verified.")

    # -------------------------------------------------------------------------
    # 4. Model Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = CNNTransformerCTC().to(device)

    # Move batch to device
    images = images.to(device)

    # Forward
    with torch.no_grad():
        logits = model(images)

    print(f"Logits shape: {logits.shape}")  # (N, Seq_Len, Vocab)

    # Expected sequence length is roughly width / 4 due to ResNet strides
    # The exact calculation in model.py is: x = x.mean(dim=2) after layer4
    # ResNet18 layer4 output stride is 32 total? No, model.py modifies strides.
    # Modified ResNet:
    # conv1 (s=2) -> pool (s=2) -> layer1 (s=1) -> layer2 (s=(2,1)) -> layer3 (s=(2,1)) -> layer4 (s=(2,1))
    # Height downsample: 2*2*2*2*2 = 32. 128 / 32 = 4.
    # Final mean(dim=2) collapses height.
    # Width downsample: 2*2*1*1*1 = 4.
    expected_seq_len = images.shape[3] // 4

    assert logits.shape[0] == 4, "Logits batch size mismatch"
    assert (
        logits.shape[1] == expected_seq_len
    ), f"Logits seq len mismatch. Got {logits.shape[1]}, expected {expected_seq_len}"
    assert logits.shape[2] == Config.VOCAB_SIZE, "Logits vocab size mismatch"

    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch, Debug Mode)...")

    # Initialize Trainer with debug=True to limit dataset size to 1000 internally
    trainer = Trainer(debug=True)

    # Run fit
    trainer.fit()

    # Verify artifact creation
    if os.path.exists(Config.MODEL_PATH):
        print(f"Training complete. Model saved to {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model file was not created after training.")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set (Sample)...")

    # Generate submission for a small subset of test data
    generate_submission(sample_size=20)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")

        # Validate submission format
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")
        print(f"Columns: {df_sub.columns.tolist()}")

        assert "image_id" in df_sub.columns
        assert "InChI" in df_sub.columns
        assert len(df_sub) > 0

        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n" + "=" * 60)
    print("Demonstration Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        run_demonstration()
    except Exception as e:
        print(f"\nCRITICAL ERROR DURING DEMONSTRATION: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
