import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library modules
from library.config import Config, seed_everything
from library.utils import compute_score, write_submission
from library.data_processor import load_data
from library.dataset import ChatbotDataset
from library.model import SiameseDeberta
from library.trainer import Trainer
from library.inference import run_inference


def run_demo():
    print("Initializing Library Demo...")

    # --- 1. Configuration Overrides for Speed & Isolation ---
    # Create a specific directory for this demo execution
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir)

    print(f"Setting up demo configuration in {demo_working_dir}...")

    # Modify Config attributes globally for this process
    Config.WORKING_DIR = demo_working_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "demo_submission.csv")

    # Hyperparameters for rapid execution
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.MAX_LENGTH = 64  # Short sequence length for speed

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # --- 2. Test Data Loading and Processing ---
    print("\n[1/6] Testing Data Processor (load_data)...")
    # Load data in debug mode (triggers subsampling and processing)
    df_train, df_val, df_test = load_data(load_cached_data=False, debug=True)

    # Validation
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(df_train)}"
    assert len(df_val) == Config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(df_val)}"
    assert (
        len(df_test) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(df_test)}"

    expected_meta_cols = ["meta_prompt_len", "meta_a_len", "meta_b_len"]
    for col in expected_meta_cols:
        assert col in df_train.columns, f"Missing meta column {col} in train"
        assert col in df_val.columns, f"Missing meta column {col} in val"
        # Check scaling (roughly, since N is small, mean might not be exactly 0 but scaler should have run)
        assert df_train[col].dtype == float, f"Meta column {col} should be float"

    print("Data loaded and processed successfully.")

    # --- 3. Test Dataset Class ---
    print("\n[2/6] Testing Dataset Class (ChatbotDataset)...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create dataset instances
    train_dataset = ChatbotDataset(
        df_train, tokenizer, max_length=Config.MAX_LENGTH, is_test=False
    )

    # Fetch one item
    sample_item = train_dataset[0]

    # Validation
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "meta_features",
        "target",
    ]
    for key in required_keys:
        assert key in sample_item, f"Missing key {key} in dataset item"

    # Check tensor shapes
    assert sample_item["input_ids_a"].dim() == 1, "Input IDs should be 1D"
    assert (
        sample_item["input_ids_a"].size(0) == Config.MAX_LENGTH
    ), f"Input IDs length mismatch. Expected {Config.MAX_LENGTH}, got {sample_item['input_ids_a'].size(0)}"
    assert sample_item["meta_features"].size(0) == 3, "Meta features dimension mismatch"
    assert sample_item["target"].size(0) == 3, "Target dimension mismatch"

    print("Dataset initialized and verified.")

    # --- 4. Test Model Architecture ---
    print("\n[3/6] Testing Model Architecture (SiameseDeberta)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SiameseDeberta(model_name=Config.MODEL_NAME, num_classes=3, meta_dim=3)
    model.to(device)

    # Prepare a batch for forward pass
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample_item.items()}

    # Forward pass
    with torch.no_grad():
        logits = model(
            batch["input_ids_a"],
            batch["attention_mask_a"],
            batch["input_ids_b"],
            batch["attention_mask_b"],
            batch["meta_features"],
        )

    # Validation
    assert logits.shape == (1, 3), f"Logits shape mismatch: {logits.shape}"
    print("Model forward pass successful.")

    # --- 5. Test Training Loop ---
    print("\n[4/6] Testing Trainer (Training Loop)...")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    # Re-create val dataset to ensure consistency
    val_dataset = ChatbotDataset(
        df_val, tokenizer, max_length=Config.MAX_LENGTH, is_test=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False
    )

    trainer = Trainer(model, device=device, patience=1)

    # Run training
    trainer.train(train_loader, val_loader, epochs=Config.EPOCHS)

    # Validation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("Training loop completed and model saved.")

    # --- 6. Test Inference Pipeline ---
    print("\n[5/6] Testing Inference Pipeline (run_inference)...")

    # Run inference (this will reload data/model internally)
    # We use debug=True to ensure it runs on the small subset
    try:
        run_inference(load_cached_data=False, debug=True, batch_size=4, device=device)
    except Exception as e:
        raise AssertionError(f"Inference pipeline failed: {e}")

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission rows mismatch: {len(sub_df)}"
    assert list(sub_df.columns) == [
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ], "Submission columns mismatch"

    # Check probability constraints
    probs = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    assert np.allclose(probs, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Inference pipeline verified.")

    # --- 7. Test Metrics ---
    print("\n[6/6] Testing Metrics (compute_score)...")
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    score = compute_score(y_true, y_pred)

    # Validation
    assert isinstance(score, float), "Score should be a float"
    assert score > 0, "Score should be positive"
    print(f"Computed Log Loss: {score:.4f}")

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    run_demo()
