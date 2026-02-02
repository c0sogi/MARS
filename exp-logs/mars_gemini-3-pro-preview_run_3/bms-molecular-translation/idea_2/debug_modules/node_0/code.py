import os
import shutil
import torch
import pandas as pd
import numpy as np
import cv2
import albumentations as A

# Import from library
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_dataloaders, InChiDataset, get_transforms
from library.model import InChiModel
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import (
    compute_levenshtein,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
)


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. Configure for Demo/Speed
    print("\n[1] Configuring Hyperparameters for Demo...")
    Config.WORKING_DIR = "./working/demo"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.json")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DECODER_DIM = 64  # Smaller model
    Config.FF_DIM = 128
    Config.NUM_HEADS = 2
    Config.NUM_LAYERS = 1
    Config.IMAGE_SIZE = 64  # Smaller image

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Size: {Config.IMAGE_SIZE}")

    # 2. Test Utils
    print("\n[2] Testing Utils...")
    lev_score = compute_levenshtein(["InChI=1S/H2O"], ["InChI=1S/H2O"])
    assert lev_score == 0.0, "Levenshtein distance for identical strings should be 0"
    lev_score = compute_levenshtein(["ABC"], ["ABD"])
    assert lev_score == 1.0, "Levenshtein distance for ABC vs ABD should be 1"
    print("Levenshtein check passed.")

    meter = AverageMeter()
    meter.update(10)
    meter.update(20)
    assert meter.avg == 15.0, "AverageMeter calculation incorrect"
    print("AverageMeter check passed.")

    # 3. Test Tokenizer
    print("\n[3] Testing Tokenizer...")
    # Force rebuild to ensure it works with current metadata
    if os.path.exists(Config.VOCAB_PATH):
        os.remove(Config.VOCAB_PATH)

    tokenizer = Tokenizer(load_cached_data=False)
    print(f"Vocabulary size: {len(tokenizer)}")

    test_str = "InChI=1S/C"
    seq = tokenizer.text_to_sequence(test_str)
    decoded_str = tokenizer.sequence_to_text(seq)

    print(f"Original: {test_str}")
    print(f"Encoded: {seq.tolist()}")
    print(f"Decoded: {decoded_str}")

    assert test_str == decoded_str, "Tokenizer encode-decode cycle failed"
    assert seq[0] == tokenizer.sos_token_id, "Sequence must start with SOS"
    # Sequence might end with EOS or PAD depending on length, usually EOS first
    assert tokenizer.eos_token_id in seq, "Sequence must contain EOS"
    print("Tokenizer check passed.")

    # 4. Test Dataset and DataLoader
    print("\n[4] Testing Dataset and DataLoader...")
    # Use a very small subset
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_subset_size=20,
    )

    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Incorrect label batch shape"
    print("DataLoader check passed.")

    # 5. Test Model
    print("\n[5] Testing Model Architecture...")
    model = InChiModel(vocab_size=len(tokenizer))
    model.to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)

    # Forward pass (Teacher Forcing input)
    decoder_input = labels[:, :-1]

    with torch.cuda.amp.autocast():
        logits = model(images, decoder_input, pad_token_id=tokenizer.pad_token_id)

    print(f"Logits Shape: {logits.shape}")
    expected_seq_len = Config.MAX_LEN - 1
    assert logits.shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
        len(tokenizer),
    ), "Incorrect logits shape"
    print("Model forward pass passed.")

    # Test Prediction (Greedy Decode)
    print("Testing Model Prediction...")
    # Predict on a few images
    pred_seqs = model.predict(images[:2], tokenizer, max_len=10)
    print(f"Prediction Shape: {pred_seqs.shape}")
    assert pred_seqs.shape[0] == 2, "Incorrect prediction batch size"
    # The length depends on when EOS is generated, but it should be <= max_len + 1 (start token)
    print("Model prediction passed.")

    # 6. Test Trainer
    print("\n[6] Testing Trainer (Fit Loop)...")
    # We use debug=True which sets subset_size=1000 in Trainer init
    # To make it even faster for this demo script, we override the loaders with tiny ones

    trainer = Trainer(load_cached_data=True, debug=True)

    # Override loaders for extreme speed in demonstration
    tiny_loader, _, _ = get_dataloaders(
        tokenizer=trainer.tokenizer,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        debug_subset_size=16,  # 2 batches
    )
    trainer.train_loader = tiny_loader
    trainer.val_loader = tiny_loader

    # Adjust scheduler steps because we changed the loader length
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(trainer.train_loader),
        epochs=Config.NUM_EPOCHS,
        pct_start=0.1,
    )

    trainer.fit()

    # Check if checkpoint exists
    checkpoint_path = os.path.join(
        Config.WORKING_DIR, "checkpoints", "checkpoint.pth.tar"
    )
    assert os.path.exists(checkpoint_path), "Checkpoint file not created"
    print("Trainer fit passed.")

    # 7. Test Inference
    print("\n[7] Testing Inference Pipeline...")
    # Create a dummy best model checkpoint to ensure inference script finds it
    # The trainer loop creates 'model_best.pth.tar' if valid loss improves.
    # Since we ran 1 epoch starting from high loss, it should have saved it.
    best_model_path = os.path.join(
        Config.WORKING_DIR, "checkpoints", "model_best.pth.tar"
    )
    if not os.path.exists(best_model_path):
        print("Best model not found, copying last checkpoint for demo purposes.")
        shutil.copy(checkpoint_path, best_model_path)

    generate_submission(load_cached_data=True, debug=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{sub_df.head()}")
    assert (
        "image_id" in sub_df.columns and "InChI" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission dataframe is empty"
    print("Inference pipeline passed.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
