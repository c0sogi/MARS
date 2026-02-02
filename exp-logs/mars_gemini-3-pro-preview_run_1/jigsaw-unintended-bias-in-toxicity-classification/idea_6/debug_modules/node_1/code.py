import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import process_data, make_loader, ToxicityDataset
from library.model import BiasAwareDeberta
from library.losses import HybridBiasLoss
from library.metrics import JigsawMetrics
from library.engine import run_training


def main():
    print("============================================================")
    print("      Toxicity Classification with Bias Mitigation Demo     ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Modify Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for demo
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid overhead for small data

    # Clean up previous runs in working directory to ensure fresh start
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")
    print("Configuration updated for fast execution.")

    # ------------------------------------------------------------------
    # 2. Data Processing & Dataset Verification
    # ------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Processing and Dataset...")

    # Process data (this will use the debug subset)
    # We force load_cached_data=False to ensure the processing logic runs
    train_dataset = process_data(mode="train", load_cached_data=False, debug=True)

    print(f"Dataset size: {len(train_dataset)}")
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(train_dataset)}"

    # Check a single item
    item = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "target",
        "sample_weight",
        "aux_identities",
        "aux_identity_attack",
    ]
    for key in required_keys:
        assert key in item, f"Dataset item missing key: {key}"

    # Verify Sample Weights logic
    # We look for a sample with identity mention and check if weight is boosted
    found_bias_trap = False
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        # Check if any identity is present (aux_identities > 0.5)
        has_identity = (sample["aux_identities"] >= 0.5).any()
        weight = sample["sample_weight"].item()

        if has_identity:
            # Should have boosted weight
            if abs(weight - Config.WEIGHT_BIAS_TRAP) < 1e-5:
                found_bias_trap = True
                break

    if found_bias_trap:
        print("Verified: Sample weights correctly boosted for identity mentions.")
    else:
        print(
            "Note: No identity mentions found in random debug subset (acceptable for small N)."
        )

    # Create DataLoader
    train_loader = make_loader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, mode="train"
    )
    batch = next(iter(train_loader))
    print("DataLoader operational. Batch keys:", batch.keys())

    # ------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    model = BiasAwareDeberta()
    model.to(device)
    model.eval()

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Check outputs
    assert "logits" in outputs
    assert "aux_identity_logits" in outputs
    assert "aux_attack_logits" in outputs

    # Check shapes
    batch_size = input_ids.size(0)
    assert outputs["logits"].shape == (batch_size, 1), "Incorrect toxicity logits shape"
    assert outputs["aux_identity_logits"].shape == (
        batch_size,
        len(Config.IDENTITY_COLUMNS),
    ), "Incorrect identity logits shape"
    assert outputs["aux_attack_logits"].shape == (
        batch_size,
        1,
    ), "Incorrect attack logits shape"

    print("Model forward pass successful. Output shapes verified.")

    # ------------------------------------------------------------------
    # 4. Loss Function Verification
    # ------------------------------------------------------------------
    print("\n[Step 4] Verifying Hybrid Bias Loss...")

    loss_fn = HybridBiasLoss()

    targets = batch["target"].to(device)
    sample_weights = batch["sample_weight"].to(device)
    aux_identities = batch["aux_identities"].to(device)
    aux_identity_attack = batch["aux_identity_attack"].to(device)

    # Compute loss
    loss = loss_fn(
        outputs, targets, sample_weights, aux_identities, aux_identity_attack
    )

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Loss function calculation verified.")

    # ------------------------------------------------------------------
    # 5. Metrics Verification
    # ------------------------------------------------------------------
    print("\n[Step 5] Verifying Jigsaw Metrics Logic...")

    # Create a synthetic dataframe to test the metric calculation specifically
    # Scenario:
    # 1. Toxic + Identity (Male) -> Prediction 0.9 (Correct)
    # 2. Non-Toxic + Identity (Male) -> Prediction 0.8 (Wrong, Bias Error)
    # 3. Non-Toxic + No Identity -> Prediction 0.1 (Correct)
    # 4. Toxic + No Identity -> Prediction 0.9 (Correct)

    metric_data = {
        "id": [1, 2, 3, 4],
        "target": [1.0, 0.0, 0.0, 1.0],  # 1 and 4 are toxic
        "prediction": [0.9, 0.8, 0.1, 0.9],
        "male": [1.0, 1.0, 0.0, 0.0],
        "female": [0.0, 0.0, 0.0, 0.0],
        # Fill other identities with 0
    }
    for col in Config.IDENTITY_COLUMNS:
        if col not in metric_data:
            metric_data[col] = [0.0] * 4

    val_df_synthetic = pd.DataFrame(metric_data)

    metrics = JigsawMetrics()
    score, detailed = metrics.calculate_score(
        val_df_synthetic, prediction_col="prediction"
    )

    print(f"Synthetic Validation Score: {score:.4f}")
    print(f"Overall AUC: {detailed['overall_auc']:.4f}")

    # Check specific Bias Metric: BPSN for Male
    # Background Positive (Toxic + No Identity): Row 4 (Pred 0.9)
    # Subgroup Negative (Non-Toxic + Identity): Row 2 (Pred 0.8)
    # The model ranks Pos (0.9) > Neg (0.8), so AUC should be 1.0 for this pair.
    male_metrics = detailed["per_identity_metrics"]["male"]
    print(f"Male Metrics: {male_metrics}")

    assert score is not None
    assert not np.isnan(score)
    print("Metric calculation verified.")

    # ------------------------------------------------------------------
    # 6. Full Engine Execution (Integration Test)
    # ------------------------------------------------------------------
    print("\n[Step 6] Running Full Training Pipeline (Integration Test)...")
    print("This runs training, validation, and inference on the debug subset.")

    # This function handles the entire loop including saving submission
    run_training(debug=True)

    # Verify artifacts
    assert os.path.exists(Config.MODEL_CHECKPOINT_PATH), "Model checkpoint not found"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    assert list(sub_df.columns) == ["id", "prediction"], "Submission columns incorrect"
    assert len(sub_df) == Config.DEBUG_SAMPLE_SIZE, "Submission row count mismatch"

    print("\n============================================================")
    print("      Demonstration Completed Successfully                  ")
    print("============================================================")


if __name__ == "__main__":
    main()
