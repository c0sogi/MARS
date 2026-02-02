import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_pearson, get_cpc_texts
from library.dataset import load_and_prepare_data, PearsonDataset
from library.model import CustomModel
from library.engine import train_fn, valid_fn
from library.train import run_training
from library.inference import predict


def demo_pipeline():
    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("--- Setting up Demo Configuration ---")

    # Override Config for a fast demonstration
    Config.seed = 42
    Config.debug = True  # Use subset of data (1000 samples)
    Config.epochs = 1  # Single epoch
    Config.train_batch_size = 8
    Config.valid_batch_size = 8
    Config.n_fold = 2  # Only 2 folds
    Config.working_dir = "./working/demo_exec"

    # Use a tiny model for speed in this demo
    # The original config uses 'microsoft/deberta-v3-large' which is resource intensive.
    # We switch to xsmall to ensure the demo runs quickly on the available hardware.
    Config.model_name = "microsoft/deberta-v3-xsmall"

    # Re-run setup to create the new working directory
    Config.setup()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Working Directory: {Config.working_dir}")
    print(f"Model: {Config.model_name}")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n--- Verifying Utilities ---")

    # Test Pearson Calculation
    y_true = [1.0, 0.5, 0.0]
    y_pred = [0.9, 0.6, 0.1]
    score = compute_pearson(y_true, y_pred)
    print(f"Pearson Score (Test): {score:.4f}")
    assert score > 0.9, "Pearson calculation seems incorrect"

    # Test CPC Text Loading
    # This reads from ./input/description.md
    cpc_texts = get_cpc_texts(load_cached_data=False)
    print(f"Loaded {len(cpc_texts)} CPC descriptions.")
    assert len(cpc_texts) > 0, "Failed to load CPC texts"

    # ------------------------------------------------------------------------
    # 3. Verify Dataset & Tokenization
    # ------------------------------------------------------------------------
    print("\n--- Verifying Dataset & Tokenization ---")

    # Load a small sample of training data
    # We use the metadata file provided in the environment
    df_train = load_and_prepare_data(Config.train_path, load_cached_data=False)
    df_sample = df_train.head(10).reset_index(drop=True)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Dataset
    dataset = PearsonDataset(
        df_sample, tokenizer, max_length=64
    )  # Short length for demo

    # Check item structure
    item = dataset[0]
    print("Dataset Item Keys:", item.keys())
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item

    input_ids = item["input_ids"]
    assert input_ids.dim() == 1, "Input IDs should be 1D tensor"
    print(f"Input IDs shape: {input_ids.shape}")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    # Initialize Model
    # We use pretrained=True to ensure we can load the config and weights correctly
    model = CustomModel(pretrained=True)
    model.to(Config.device)
    model.eval()

    # Create a dummy batch
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    b_input_ids = batch["input_ids"].to(Config.device)
    b_mask = batch["attention_mask"].to(Config.device)
    b_token_type = batch["token_type_ids"].to(Config.device)
    b_labels = batch["label"].to(Config.device)

    # Forward pass
    with torch.no_grad():
        output = model(b_input_ids, b_mask, b_token_type, b_labels)

    logits = output["logits"]
    loss = output["loss"]

    print(f"Logits shape: {logits.shape}")
    print(f"Loss value: {loss.item()}")

    assert logits.shape == (2, 1), "Logits shape mismatch (Batch Size, 1)"
    assert not torch.isnan(loss), "Loss is NaN"

    # Clean up memory
    del model, batch, dataloader
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 5. Execute Training Pipeline (Mini-Run)
    # ------------------------------------------------------------------------
    print("\n--- Executing Training Pipeline (Mini-Run) ---")
    # This function uses the Config settings we overrode earlier.
    # It will train for 1 epoch on 2 folds using a subset of data (Config.debug=True).
    run_training()

    # Check if models were saved
    for fold in range(Config.n_fold):
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")
        assert os.path.exists(model_path), f"Model for fold {fold} was not saved."
        print(f"Verified existence of {model_path}")

    # ------------------------------------------------------------------------
    # 6. Execute Inference Pipeline
    # ------------------------------------------------------------------------
    print("\n--- Executing Inference Pipeline ---")
    # Run prediction on the test set (debug mode samples subset)
    predict(debug=True)

    # Verify Submission
    submission_path = Config.submission_path
    assert os.path.exists(submission_path), "Submission file not found."

    df_sub = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df_sub.head())

    assert len(df_sub) > 0, "Submission file is empty."
    assert (
        "id" in df_sub.columns and "score" in df_sub.columns
    ), "Invalid submission format."
    assert df_sub["score"].between(0, 1).all(), "Scores out of range [0, 1]."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        demo_pipeline()
    except Exception as e:
        print(f"\n!!! Demo Failed: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
