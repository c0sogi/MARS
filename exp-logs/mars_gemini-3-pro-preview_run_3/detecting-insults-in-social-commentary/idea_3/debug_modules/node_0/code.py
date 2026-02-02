import os
import sys
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import transformers
import logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, decode_text, get_score
from library.data import get_dataloaders
from library.model import InsultModel
from library.trainer import get_optimizer_params, train_fn, valid_fn, inference_fn

# Suppress warnings and verbose logs
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Library Demonstration ===")

    # 1. Setup and Configuration
    # We override Config values to ensure the demo runs quickly (Debug Mode)
    print("\n[1] Configuring environment...")
    Config.debug = True
    Config.debug_sample_size = 20  # Small sample for speed
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.epochs = 1
    Config.output_dir = "./working/demo_run/"

    # Ensure clean state
    if os.path.exists(Config.output_dir):
        shutil.rmtree(Config.output_dir)
    os.makedirs(Config.output_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.seed)
    print("    Configuration set: Debug=True, Batch Size=4, Sample Size=20")

    # 2. Utility Verification
    print("\n[2] Verifying Utilities...")
    # Test decode_text
    raw_text = "Hello\\nWorld"
    decoded = decode_text(raw_text)
    assert decoded == "Hello\nWorld", f"decode_text failed: {decoded}"

    # Test get_score (AUC)
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    score = get_score(y_true, y_pred)
    assert 0.0 <= score <= 1.0, "get_score returned invalid range"
    print("    Utilities verified successfully.")

    # 3. Data Loading
    print("\n[3] Loading Data (Debug Mode)...")
    # This uses library.data.get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing to test logic
        debug=Config.debug,
        debug_sample_size=Config.debug_sample_size,
    )

    # Verify DataLoaders
    assert len(train_loader) > 0, "Train loader is empty"
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = {"input_ids", "attention_mask", "target"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Check shapes
    input_ids = batch["input_ids"]
    targets = batch["target"]
    print(f"    Batch shapes - Input: {input_ids.shape}, Target: {targets.shape}")

    assert input_ids.shape[0] == Config.train_batch_size, "Incorrect batch size"
    assert input_ids.shape[1] == Config.max_len, "Incorrect sequence length"
    assert targets.shape[0] == Config.train_batch_size, "Incorrect target shape"
    print("    Data loading verified.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    device = Config.device
    model = InsultModel()
    model.to(device)

    # Verify Forward Pass
    input_ids = input_ids.to(device)
    attention_mask = batch["attention_mask"].to(device)

    # The model returns logits (before sigmoid)
    logits = model(input_ids, attention_mask)

    assert logits.shape == (
        Config.train_batch_size,
        1,
    ), f"Model output shape mismatch: {logits.shape}"
    print("    Model forward pass successful. Output shape verified.")

    # 5. Optimizer & LLRD (Layer-wise Learning Rate Decay)
    print("\n[5] Configuring Optimizer with LLRD...")
    encoder_lr = 1e-5
    decoder_lr = 1e-4
    weight_decay = 0.01

    optimizer_params = get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay)

    # Basic check: ensure we have parameter groups
    assert len(optimizer_params) > 0, "Optimizer parameters list is empty"

    # LLRD Logic Check:
    # The head (decoder) should have the highest LR.
    # The embeddings (bottom of encoder) should have the lowest LR (due to decay).

    # Find head LR (usually the first group added in the function logic for non-backbone)
    # Based on implementation: Head params are added first.
    head_lr = optimizer_params[0]["lr"]

    # Find embedding LR (usually the last groups added in the function logic)
    embed_lr = optimizer_params[-1]["lr"]

    print(f"    Head LR: {head_lr}, Embedding LR: {embed_lr}")
    assert head_lr == decoder_lr, "Head LR does not match requested decoder_lr"
    assert embed_lr < head_lr, "LLRD failed: Embedding LR should be lower than Head LR"

    optimizer = torch.optim.AdamW(optimizer_params, lr=encoder_lr)
    print("    Optimizer configuration verified.")

    # 6. Training Loop (Single Epoch/Step)
    print("\n[6] Running Training Step...")
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch (which is very short due to debug sample size)
    train_loss = train_fn(
        train_loader,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device=device,
        epoch=0,
    )

    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss >= 0, "Training loss is negative"

    # 7. Validation Loop
    print("\n[7] Running Validation Step...")
    val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

    print(f"    Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "Validation AUC out of range"

    # 8. Inference & Submission Generation
    print("\n[8] Running Inference on Test Set...")
    preds = inference_fn(test_loader, model, device)

    assert len(preds) == len(test_loader.dataset), "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0,1]"

    print(f"    Generated {len(preds)} predictions.")

    # Create a dummy submission file to prove end-to-end capability
    test_df = pd.read_csv(Config.test_path).head(Config.debug_sample_size)
    submission = pd.DataFrame(
        {"Insult": preds, "Date": test_df["Date"], "Comment": test_df["Comment"]}
    )

    sub_path = os.path.join(Config.output_dir, "submission_demo.csv")
    submission.to_csv(sub_path, index=False)
    print(f"    Submission saved to {sub_path}")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
