import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import process_data, TweetDataset, get_loaders
from library.model import TweetModel
from library.loss import compute_loss
from library.engine import get_optimizer_params, train_fn, eval_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Sentiment Extraction Library Demo ===\n")

    # 1. Setup & Configuration
    print("--- Step 1: Configuration & Setup ---")
    seed_everything(42)

    # Override Config for rapid demonstration
    Config.WORKING_DIR = "./working/demo_execution/"
    Config.TRAIN_FILE = "./working/demo_train_subset.csv"
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.LOAD_CACHED_DATA = False  # Force fresh processing

    # Create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Prepare Demo Data
    print("\n--- Step 2: Preparing Demo Data ---")
    # Load original train metadata and sample a small subset
    full_train_df = pd.read_csv("./metadata/train_metadata.csv")
    # Filter for non-neutral to ensure we have targets (as per library logic)
    demo_df = full_train_df[full_train_df["sentiment"] != "neutral"].head(32).copy()
    demo_df.to_csv(Config.TRAIN_FILE, index=False)
    print(
        f"Created temporary training subset: {Config.TRAIN_FILE} ({len(demo_df)} rows)"
    )

    # 3. Test Data Processing Logic
    print("\n--- Step 3: Verifying Data Processing ---")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    # Create a synthetic example to verify target generation
    example_data = {
        "textID": ["test001"],
        "text": ["The weather is absolutely fantastic today!"],
        "sentiment": ["positive"],
        "selected_text": ["fantastic"],
    }
    df_example = pd.DataFrame(example_data)

    ids, mask, st, et, offsets = process_data(
        df_example, tokenizer, Config.MAX_LEN, Config.LABEL_SMOOTHING_SIGMA
    )

    print(f"Input IDs Shape: {ids.shape}")
    print(f"Start Targets Shape: {st.shape}")

    # Assertions
    assert ids.shape == (1, Config.MAX_LEN), "Incorrect Input IDs shape"
    assert st.shape == (1, Config.MAX_LEN), "Incorrect Start Targets shape"
    # The target should be a probability distribution (sum close to 1 due to gaussian smoothing)
    assert np.isclose(st.sum(), 1.0, atol=0.1), "Start targets do not sum to ~1"
    assert np.isclose(et.sum(), 1.0, atol=0.1), "End targets do not sum to ~1"

    # Check if the target is focusing on 'fantastic' (should not be at index 0)
    argmax_start = np.argmax(st)
    assert argmax_start > 0, "Target index should not be 0 (CLS) for this example"
    print("Data processing logic verified.")

    # 4. Test Dataset and DataLoader
    print("\n--- Step 4: Verifying Dataset & DataLoader ---")
    # Instantiate Dataset manually for the subset
    # We process the demo_df we created earlier
    ids_train, mask_train, st_train, et_train, off_train = process_data(
        demo_df, tokenizer, Config.MAX_LEN, Config.LABEL_SMOOTHING_SIGMA
    )

    train_dataset = TweetDataset(
        ids_train,
        mask_train,
        st_train,
        et_train,
        off_train,
        texts=demo_df["text"].values,
        selected_texts=demo_df["selected_text"].values,
        sentiments=demo_df["sentiment"].values,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )

    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    assert "input_ids" in batch
    assert "start_targets" in batch
    assert batch["input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    print("DataLoader yields correct batch structure.")

    # 5. Test Model Initialization
    print("\n--- Step 5: Initializing Model ---")
    model = TweetModel()
    model.to(Config.DEVICE)
    print("Model initialized successfully.")

    # 6. Test Forward Pass and Loss
    print("\n--- Step 6: Forward Pass & Loss Computation ---")
    b_input_ids = batch["input_ids"].to(Config.DEVICE)
    b_mask = batch["attention_mask"].to(Config.DEVICE)
    b_start_targets = batch["start_targets"].to(Config.DEVICE)
    b_end_targets = batch["end_targets"].to(Config.DEVICE)

    # Forward
    start_logits, end_logits = model(b_input_ids, b_mask)

    print(f"Logits Shape: {start_logits.shape}")
    assert start_logits.shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)

    # Loss
    loss = compute_loss(
        start_logits, end_logits, b_start_targets, b_end_targets, b_mask
    )
    print(f"Calculated Loss: {loss.item():.4f}")

    assert loss.item() > 0, "Loss should be positive"

    # Backward
    loss.backward()
    print("Backward pass successful.")

    # 7. Test Training Loop (Single Epoch on Subset)
    print("\n--- Step 7: Testing Training Loop (1 Epoch on 32 samples) ---")
    optimizer_params = get_optimizer_params(model, encoder_lr=1e-5, decoder_lr=1e-5)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Using train_fn from engine
    avg_loss = train_fn(train_loader, model, optimizer, Config.DEVICE)
    print(f"Training Loop Completed. Average Loss: {avg_loss:.4f}")

    # 8. Test Evaluation Loop
    print("\n--- Step 8: Testing Evaluation Loop ---")
    # We use the same loader for eval just to verify the function works
    # In reality, this would be the validation set
    val_loss, val_jaccard = eval_fn(train_loader, model, Config.DEVICE)
    print(f"Eval Loss: {val_loss:.4f}")
    print(f"Eval Jaccard: {val_jaccard:.4f}")

    assert 0 <= val_jaccard <= 1, "Jaccard score out of range"
    print("Evaluation logic verified.")

    # 9. Verify library.data.get_loaders integration
    print("\n--- Step 9: Verifying 'get_loaders' Integration ---")
    # This function loads the full validation set + our demo train set, then splits them.
    # It ensures that the high-level data pipeline is functional.
    try:
        print("Initializing standard loaders (Fold 0)...")
        gl_train, gl_val = get_loaders(fold=0)
        print(f"Standard Train Loader batches: {len(gl_train)}")
        print(f"Standard Val Loader batches: {len(gl_val)}")

        # Verify one batch from the generated loader
        gl_batch = next(iter(gl_train))
        assert gl_batch["input_ids"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
        print("get_loaders integration verified.")
    except Exception as e:
        print(f"Error during get_loaders execution: {e}")
        raise e

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
