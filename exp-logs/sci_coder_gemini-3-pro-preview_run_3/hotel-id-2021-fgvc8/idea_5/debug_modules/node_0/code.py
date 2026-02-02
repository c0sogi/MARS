import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import EfficientNetArcFace
from library.trainer import train_fn, eval_fn
from library.inference import inference_fn


def run_demo():
    print("=== Starting Hotel ID Task Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Set paths for demo outputs
    DEMO_DIR = "./working/demo"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.WORKING_DIR = DEMO_DIR
    Config.OUTPUT_DIR = DEMO_DIR
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "model_demo.pth")

    # Reduce computational load
    Config.IMAGE_SIZE = 128
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.EPOCHS = 1

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = str(device)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")

    # Seed
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Generate DataLoaders
    # This will also create the label_encoder.parquet in the working dir
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        load_cached_data=False, debug=True  # Force regeneration for demo
    )

    print(f"    Num Classes: {num_classes}")
    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")
    print(f"    Test Batches: {len(test_loader)}")

    # Validate Train Batch
    images, labels = next(iter(train_loader))
    print(f"    Sample Batch Shape: {images.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert labels.max() < num_classes, "Label index out of bounds"

    # Validate Label Encoder Cache
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.parquet")
    assert os.path.exists(encoder_path), "Label encoder parquet file was not created"

    # ---------------------------------------------------------
    # 3. Model Logic Validation
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Instantiate model
    # Force pretrained=False to avoid downloading weights during demo
    model = EfficientNetArcFace(n_classes=num_classes, pretrained=False).to(device)

    # Create dummy input
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    dummy_labels = torch.randint(0, num_classes, (Config.BATCH_SIZE,)).to(device)

    # Test Forward Pass (Training Mode - with labels)
    # Should return ArcFace logits
    model.train()
    logits = model(dummy_input, dummy_labels)
    assert logits.shape == (
        Config.BATCH_SIZE,
        num_classes,
    ), f"Expected logits shape {(Config.BATCH_SIZE, num_classes)}, got {logits.shape}"

    # Test Forward Pass (Inference Mode - no labels)
    # Should return Embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(dummy_input)
    assert embeddings.shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_SIZE,
    ), f"Expected embedding shape {(Config.BATCH_SIZE, Config.EMBEDDING_SIZE)}, got {embeddings.shape}"

    print("    Model forward passes successful.")

    # ---------------------------------------------------------
    # 4. Training & Evaluation Simulation
    # ---------------------------------------------------------
    print("\n[4] Simulating Training and Evaluation...")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run one training step (using the full loader for 1 epoch on debug subset)
    avg_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch=0)
    print(f"    Training Loss: {avg_loss:.4f}")

    assert avg_loss > 0, "Training loss should be positive"

    # Run evaluation
    map_score = eval_fn(val_loader, model, device)
    print(f"    Validation MAP@5: {map_score:.4f}")

    assert 0.0 <= map_score <= 1.0, "MAP score must be between 0 and 1"

    # ---------------------------------------------------------
    # 5. Inference Pipeline Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Inference Pipeline...")

    # Load label map for decoding
    label_df = pd.read_parquet(encoder_path)
    label_map = dict(zip(label_df["hotel_id"], label_df["label_idx"]))

    # Run inference
    # This saves the submission file to Config.SUBMISSION_FILE
    inference_fn(test_loader, model, device, label_map)

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission shape: {sub_df.shape}")
    print("    First few rows:")
    print(sub_df.head(2))

    # Validate format
    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission missing required columns"
    assert len(sub_df) == len(test_loader.dataset), "Submission row count mismatch"

    # Check prediction format (space delimited)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction must be a string"
    assert len(sample_pred.split()) <= 5, "Should predict at most 5 hotel IDs"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
