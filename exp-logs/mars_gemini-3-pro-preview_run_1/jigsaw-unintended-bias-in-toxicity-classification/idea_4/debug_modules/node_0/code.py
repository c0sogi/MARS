import os
import shutil
import numpy as np
import pandas as pd
import torch
import transformers
from library.config import Config
from library.metrics import calculate_final_score, compute_bias_metrics
from library.data_processing import create_dataloaders, calculate_sample_weights
from library.model import MultiTaskTransformer
from library.engine import run_training


# ==========================================
# Setup & Configuration
# ==========================================
def setup_environment():
    """Sets up the environment for a fast demonstration run."""
    print("Setting up environment...")

    # Suppress verbose logs from transformers
    transformers.logging.set_verbosity_error()

    # Modify Config for speed
    # We use a very small subset and 1 epoch to ensure completion within minutes
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16
    Config.WARMUP_STEPS = 0

    # Ensure clean working directory for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)


# ==========================================
# 1. Metric Logic Verification
# ==========================================
def test_metrics_logic():
    """Verifies that metric calculations handle various edge cases correctly."""
    print("\nTesting Metric Logic...")

    # Create synthetic data covering different bias scenarios
    # Columns: target, prediction, identity_columns...
    data = {
        "target": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],  # Mixed targets
        "prediction": [0.1, 0.8, 0.2, 0.9, 0.1, 0.9],  # Mixed predictions (some errors)
    }

    # Add identity columns (binary 0 or 1 for this test)
    for identity in Config.IDENTITY_COLUMNS:
        # Random binary assignment for test
        data[identity] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    df = pd.DataFrame(data)

    # Test compute_bias_metrics
    bias_metrics = compute_bias_metrics(
        df,
        pred_col="prediction",
        label_col="target",
        identity_cols=Config.IDENTITY_COLUMNS,
    )

    # Assertions
    assert isinstance(
        bias_metrics, pd.DataFrame
    ), "Bias metrics should return a DataFrame"
    expected_cols = ["subgroup", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    assert all(
        col in bias_metrics.columns for col in expected_cols
    ), f"Missing columns in bias metrics. Expected {expected_cols}"
    assert len(bias_metrics) == len(
        Config.IDENTITY_COLUMNS
    ), "Should have one row per identity"

    # Test calculate_final_score
    score_dict = calculate_final_score(df, pred_col="prediction", label_col="target")

    assert "score" in score_dict, "Final score dictionary missing 'score' key"
    assert (
        "overall_auc" in score_dict
    ), "Final score dictionary missing 'overall_auc' key"
    assert 0.0 <= score_dict["score"] <= 1.0, "Score should be between 0 and 1"

    print("Metric logic verified successfully.")


# ==========================================
# 2. Data Pipeline Verification
# ==========================================
def test_data_pipeline():
    """Verifies DataLoader construction and Sample Weighting logic."""
    print("\nTesting Data Pipeline...")

    # Use a small data limit to speed up loading
    data_limit = 100

    # 1. Test Sample Weight Calculation Logic directly
    # Create a dummy df with one identity mention and one without
    dummy_df = pd.DataFrame(
        {
            "comment_text": ["text1", "text2"],
            "male": [0.0, 0.6],  # 2nd row mentions 'male'
            "female": [0.0, 0.0],
            "homosexual_gay_or_lesbian": [0.0, 0.0],
            "christian": [0.0, 0.0],
            "jewish": [0.0, 0.0],
            "muslim": [0.0, 0.0],
            "black": [0.0, 0.0],
            "white": [0.0, 0.0],
            "psychiatric_or_mental_illness": [0.0, 0.0],
        }
    )

    weights = calculate_sample_weights(dummy_df)

    # Assertions for weights
    # Row 0: No identity -> Background weight
    assert np.isclose(
        weights[0], Config.BACKGROUND_WEIGHT_FACTOR
    ), "Row without identity should have background weight"
    # Row 1: Identity -> Identity weight
    assert np.isclose(
        weights[1], Config.IDENTITY_WEIGHT_FACTOR
    ), "Row with identity should have identity weight"

    # 2. Test DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        load_cached_data=False, data_limit=data_limit
    )

    # Fetch a batch
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = ["input_ids", "attention_mask", "target", "aux_target", "weight"]
    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Shapes
    batch_size = batch["input_ids"].size(0)
    assert batch["input_ids"].shape == (batch_size, Config.MAX_LEN)
    assert batch["target"].shape == (batch_size,)
    assert batch["aux_target"].shape == (batch_size, Config.NUM_IDENTITY_LABELS)
    assert batch["weight"].shape == (batch_size,)

    print("Data pipeline verified successfully.")
    return train_loader  # Return for model testing


# ==========================================
# 3. Model Architecture Verification
# ==========================================
def test_model_architecture(dataloader):
    """Verifies that the model forward pass works and produces correct shapes."""
    print("\nTesting Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = MultiTaskTransformer(model_name=Config.MODEL_NAME)
    model.to(device)
    model.eval()

    # Get a batch
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        tox_logits, ident_logits = model(input_ids, mask)

    # Verify Output Shapes
    batch_size = input_ids.size(0)

    # Toxicity head: [Batch, 1]
    assert tox_logits.shape == (
        batch_size,
        1,
    ), f"Expected toxicity logits shape {(batch_size, 1)}, got {tox_logits.shape}"

    # Identity head: [Batch, Num_Identities]
    expected_ident_shape = (batch_size, Config.NUM_IDENTITY_LABELS)
    assert (
        ident_logits.shape == expected_ident_shape
    ), f"Expected identity logits shape {expected_ident_shape}, got {ident_logits.shape}"

    print("Model architecture verified successfully.")


# ==========================================
# 4. Full Training Integration Test
# ==========================================
def test_full_training_loop():
    """Runs the full training engine on a small subset to verify end-to-end execution."""
    print("\nTesting Full Training Loop (Integration Test)...")

    # Limit data to a very small number for the integration test
    # 64 samples is enough to form a couple of batches
    subset_limit = 64

    # Run the engine
    # This handles loading, training, validation, saving, and submission generation
    run_training(load_cached_data=False, data_limit=subset_limit)

    # Verify Output Files
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify Submission Content
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(submission_df) == subset_limit
    ), f"Submission should have {subset_limit} rows (matching data_limit), got {len(submission_df)}"
    assert (
        "id" in submission_df.columns and "prediction" in submission_df.columns
    ), "Submission missing required columns"

    print("Full training loop verified successfully.")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    setup_environment()

    # Run Verification Steps
    test_metrics_logic()

    # We get the dataloader from the pipeline test to reuse it for the model test
    train_loader = test_data_pipeline()

    test_model_architecture(train_loader)

    test_full_training_loop()

    print("\nAll tests passed. The solution is valid and functional.")
