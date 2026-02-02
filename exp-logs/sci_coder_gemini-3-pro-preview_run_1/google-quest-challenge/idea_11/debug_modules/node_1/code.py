import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.data import load_data, QADataset, Collate
from library.model import DistilRoBERTaDualEncoder
from library.engine import get_optimizer, get_scheduler, train_fn, eval_fn, inference_fn
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("Starting Demo Script...")

    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print("\n[1] Configuring environment...")

    # Override Config for the demo to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cached.parquet")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cached.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cached.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Speed optimizations
    Config.MAX_LEN = 64  # Short sequence length for demo
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seed
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading and Processing
    # ==========================================
    print("\n[2] Loading and processing data...")

    # Load raw dataframes (ignoring cache to force loading from metadata for this demo)
    train_df, val_df, test_df = load_data(load_cached_data=False)

    # SUBSET DATA for speed
    print("    Subsetting data for rapid demonstration...")
    train_subset = train_df.head(20).reset_index(drop=True)
    val_subset = val_df.head(10).reset_index(drop=True)
    test_subset = test_df.head(10).reset_index(drop=True)

    print(f"    Train shape: {train_subset.shape}")
    print(f"    Val shape:   {val_subset.shape}")
    print(f"    Test shape:  {test_subset.shape}")

    # Initialize Tokenizer
    print("    Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.BACKBONE)

    # Create Datasets
    print("    Creating Datasets...")
    train_ds = QADataset(train_subset, tokenizer, Config.MAX_LEN, is_test=False)
    val_ds = QADataset(val_subset, tokenizer, Config.MAX_LEN, is_test=False)
    test_ds = QADataset(test_subset, tokenizer, Config.MAX_LEN, is_test=True)

    # Verify Dataset Logic
    sample_item = train_ds[0]
    assert "q_input_ids" in sample_item
    assert "a_input_ids" in sample_item
    assert "labels" in sample_item
    assert sample_item["labels"].shape[0] == 30, "Target dimension mismatch"
    print("    Dataset verification passed.")

    # Create DataLoaders
    print("    Creating DataLoaders...")
    collate_fn = Collate(pad_token_id=tokenizer.pad_token_id)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[3] Initializing Model...")
    model = DistilRoBERTaDualEncoder()
    model.to(Config.DEVICE)

    # Verify Model Output Shape with a dummy batch
    print("    Verifying model forward pass...")
    dummy_batch = next(iter(train_loader))
    with torch.no_grad():
        logits = model(
            q_input_ids=dummy_batch["q_input_ids"].to(Config.DEVICE),
            q_attention_mask=dummy_batch["q_attention_mask"].to(Config.DEVICE),
            a_input_ids=dummy_batch["a_input_ids"].to(Config.DEVICE),
            a_attention_mask=dummy_batch["a_attention_mask"].to(Config.DEVICE),
        )
    assert logits.shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Logits shape mismatch: {logits.shape}"
    print("    Model forward pass verification passed.")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n[4] Running Training Loop (1 Epoch)...")

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_train_steps=len(train_loader))

    # Train
    train_loss = train_fn(
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        device=Config.DEVICE,
        scheduler=scheduler,
        epoch=0,
    )
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Evaluate
    print("    Running Validation...")
    val_loss, val_corr = eval_fn(val_loader, model, Config.DEVICE)
    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val Spearman Correlation: {val_corr:.4f}")

    # Save Model (Demo)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"    Model saved to {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 5. Inference
    # ==========================================
    print("\n[5] Running Inference on Test Set...")

    test_preds = inference_fn(test_loader, model, Config.DEVICE)

    # Verify Predictions
    assert test_preds.shape == (len(test_subset), 30), "Prediction shape mismatch"
    assert (test_preds >= 0).all() and (
        test_preds <= 1
    ).all(), "Predictions out of range [0, 1]"
    print("    Inference verification passed.")

    # ==========================================
    # 6. Submission File Generation
    # ==========================================
    print("\n[6] Generating Submission File...")

    submission_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "qa_id", test_subset["qa_id"])

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Verify file content
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert saved_df.shape == (10, 31), "Saved submission shape incorrect"
    print("    Submission file verification passed.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
