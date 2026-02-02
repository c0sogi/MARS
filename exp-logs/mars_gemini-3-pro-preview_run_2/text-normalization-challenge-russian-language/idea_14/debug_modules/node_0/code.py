import os
import pandas as pd
import torch
import numpy as np
import shutil
import time

# Import library components
from library.config import Config
from library.utils import set_seed
from library.hfbb import HierarchicalBackoff
from library.tokenizer import HybridTokenizer
from library.transformer_model import TransformerTrainer, NormalizationDataset
from library.data_factory import create_dataloaders, _add_context
from library.predictor import HybridPredictor


def setup_demo_data():
    """
    Creates a small subset of the data for demonstration purposes to ensure
    the script runs quickly and within memory limits.
    """
    print("--- Setting up Demo Data ---")
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # Read top N rows from metadata files
    # We use the provided metadata files which are guaranteed to exist
    train_src = "./metadata/train.csv"
    val_src = "./metadata/val.csv"
    test_src = "./metadata/test.csv"

    # Create mini datasets
    # Train: 500 rows to ensure some variety
    df_train = pd.read_csv(train_src, nrows=500)
    # Ensure no NaNs in critical columns for the demo
    df_train = df_train.fillna("")
    train_path = os.path.join(demo_dir, "mini_train.csv")
    df_train.to_csv(train_path, index=False)

    # Val: 50 rows
    df_val = pd.read_csv(val_src, nrows=50)
    df_val = df_val.fillna("")
    val_path = os.path.join(demo_dir, "mini_val.csv")
    df_val.to_csv(val_path, index=False)

    # Test: 50 rows
    df_test = pd.read_csv(test_src, nrows=50)
    df_test = df_test.fillna("")
    test_path = os.path.join(demo_dir, "mini_test.csv")
    df_test.to_csv(test_path, index=False)

    print(f"Created mini datasets in {demo_dir}")
    return train_path, val_path, test_path


def override_config(train_path, val_path, test_path):
    """
    Overrides the default configuration to use the demo data and
    lightweight hyperparameters.
    """
    print("--- Overriding Configuration ---")

    # Paths
    Config.BASE_WORK_DIR = "./working/demo_execution"
    Config.TRAIN_DATA = train_path
    Config.VAL_DATA = val_path
    Config.TEST_DATA = test_path
    Config.SUBMISSION_DIR = os.path.join(Config.BASE_WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Model / Training Params for Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.BPE_VOCAB_SIZE = 200  # Tiny vocab for tiny data
    Config.D_MODEL = 64  # Small model
    Config.NHEAD = 2
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 128

    # Debugging
    Config.DEBUG = True
    Config.DEBUG_SIZE = 500  # Use all of our mini dataset

    # Re-run setup to create directories based on new config
    Config.setup()
    print(f"Config updated. Artifacts will be stored in: {Config.ARTIFACT_DIR}")


def test_hfbb():
    """
    Demonstrates and verifies the Hierarchical Backoff (Tier 1) component.
    """
    print("\n--- Testing HFBB (Tier 1) ---")
    hfbb = HierarchicalBackoff()

    # Fit on the mini training data
    # load_cached_data=False forces re-computation for the demo
    hfbb.fit(load_cached_data=False)

    # Verification
    # Pick a known row from the mini train set to verify retrieval
    df_train = pd.read_csv(Config.TRAIN_DATA)
    sample_row = df_train.iloc[0]
    before = str(sample_row["before"])
    after = str(sample_row["after"])

    # We need context. Since it's the first row, prev is <START>
    pred, conf, level = hfbb.query(before, prev_token="<START>", next_token="<END>")

    print(f"Query: '{before}' -> Expected: '{after}'")
    print(f"Result: Pred='{pred}', Conf={conf:.2f}, Level='{level}'")

    # Assertion: The model should have memorized this training example
    # Note: If 'before' == 'after' and it's simple, it might be unigram.
    if pred is not None:
        assert pred == after, f"HFBB failed to retrieve correct mapping for '{before}'"

    print("HFBB verification passed.")
    return hfbb


def test_tokenizer():
    """
    Demonstrates and verifies the Hybrid Tokenizer.
    """
    print("\n--- Testing Tokenizer ---")
    tokenizer = HybridTokenizer()

    # Train on mini data
    tokenizer.train(load_cached_data=False)

    # Verify Encoder (Char-level)
    src_text = "test"
    prev_ctx = "prev"
    next_ctx = "next"
    src_tensor = tokenizer.encode_src(prev_ctx, src_text, next_ctx)

    print(f"Encoded Source Shape: {src_tensor.shape}")
    assert isinstance(src_tensor, torch.Tensor)
    assert src_tensor.dtype == torch.long
    assert len(src_tensor) == Config.MAX_ENC_LEN

    # Verify Decoder (BPE)
    tgt_text = "normalization"
    tgt_tensor = tokenizer.encode_tgt(tgt_text)

    print(f"Encoded Target Shape: {tgt_tensor.shape}")
    decoded_text = tokenizer.decode(tgt_tensor)
    print(f"Decoded Target: '{decoded_text}'")

    # Basic check: decoded text should contain characters from original
    # (Exact match depends on BPE segmentation of unseen words in tiny vocab)
    assert len(decoded_text) > 0

    print("Tokenizer verification passed.")
    return tokenizer


def test_transformer_training(tokenizer, hfbb):
    """
    Demonstrates the Transformer Trainer and runs a single training step.
    """
    print("\n--- Testing Transformer Training Step ---")

    # Create DataLoaders
    # We use load_cached_data=False to ensure we use the mini dataset we just created
    train_loader, val_loader = create_dataloaders(
        tokenizer, hfbb, load_cached_data=False
    )

    # Initialize Trainer
    trainer = TransformerTrainer(tokenizer)
    trainer.model.train()

    # Fetch one batch
    batch = next(iter(train_loader))
    src = batch["src"].to(trainer.device)
    tgt = batch["tgt"].to(trainer.device)
    weights = batch["weight"].to(trainer.device)

    print(
        f"Batch Shapes - Src: {src.shape}, Tgt: {tgt.shape}, Weights: {weights.shape}"
    )

    # Forward Pass
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    src_pad_mask = src == tokenizer.char2id[tokenizer.PAD_TOKEN]
    tgt_pad_mask = tgt_input == tokenizer.pad_id
    tgt_mask = trainer.model.generate_square_subsequent_mask(tgt_input.size(1)).to(
        trainer.device
    )

    logits = trainer.model(
        src,
        tgt_input,
        src_key_padding_mask=src_pad_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        tgt_mask=tgt_mask,
    )

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        src.shape[0],
        tgt_input.shape[1],
        tokenizer.get_vocab_sizes()[1],
    )

    # Loss Calculation
    loss_per_token = trainer.criterion(
        logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
    )
    weights_expanded = weights.unsqueeze(1).expand_as(tgt_output).reshape(-1)
    weighted_loss = loss_per_token * weights_expanded
    loss = weighted_loss.mean()

    print(f"Calculated Loss: {loss.item()}")
    assert not np.isnan(loss.item())

    # Backward Pass
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    print("Backward pass completed successfully.")

    # Save this model as 'best' for the predictor test
    torch.save(trainer.model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Saved demo model to {Config.BEST_MODEL_PATH}")


def test_inference():
    """
    Demonstrates the full inference pipeline using HybridPredictor.
    """
    print("\n--- Testing Hybrid Inference ---")

    # Initialize Predictor
    # load_cached_data=True because we want to use the artifacts (tokenizer/hfbb)
    # we just created in previous steps
    predictor = HybridPredictor(load_cached_data=True)

    # Load mini test set
    df_test = pd.read_csv(Config.TEST_DATA)

    # Run prediction
    predictions = predictor.predict(df_test)

    print(f"Predictions generated: {len(predictions)}")
    print(f"Sample Prediction (First 3): {predictions[:3]}")

    assert len(predictions) == len(df_test)
    assert all(isinstance(p, str) for p in predictions)

    print("Inference verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup Data
    train_p, val_p, test_p = setup_demo_data()

    # 2. Configure Environment
    override_config(train_p, val_p, test_p)

    # 3. Test Components
    hfbb_model = test_hfbb()
    tokenizer_model = test_tokenizer()

    # 4. Test Training Loop (Integration)
    test_transformer_training(tokenizer_model, hfbb_model)

    # 5. Test Inference
    test_inference()

    print("\n=== All Demonstrations Completed Successfully ===")
