import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.data_processing import get_dataloaders, get_test_dataloader
from library.model import ToxicityModel
from library.trainer import Trainer
from library.metrics import compute_bias_metrics
from library.utils import seed_everything


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set paths to a clean working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update output paths based on new working dir
    Config.OUTPUT_MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.OUTPUT_SWA_MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_swa.pth")

    # Enable Debug mode to use a tiny subset of data (e.g., 50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50

    # Use a smaller model for speed (DistilRoBERTa) instead of RoBERTa-Large
    # We must also update hidden size to match the smaller model (768 vs 1024)
    Config.MODEL_NAME = "distilroberta-base"
    Config.HIDDEN_SIZE = 768

    # Training hyperparameters for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VAL_BATCH_SIZE = 8
    Config.ACCUMULATE_GRAD_STEPS = 1
    Config.USE_SWA = False  # Disable SWA to save time
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, Model=distilroberta-base, Epochs=1")

    # --------------------------------------------------------------------------
    # 2. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load dataloaders (force processing to ensure logic runs)
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    print(f"Train Loader length: {len(train_loader)}")
    print(f"Val Loader length: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "target",
        "aux_target",
        "sample_weight",
        "id",
    ]

    for key in required_keys:
        assert key in batch, f"Missing key {key} in batch"

    print("Batch keys verified.")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Target shape: {batch['target'].shape}")

    # Assert shapes
    # Batch size should be Config.TRAIN_BATCH_SIZE (or less if last batch, but here we have 50 samples / 4 = 12 batches)
    assert batch["input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    assert batch["input_ids"].shape[1] == Config.MAX_LEN

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Initialization...")

    model = ToxicityModel()
    # Move to CPU for this quick check (Trainer will move to GPU)
    model.to("cpu")
    model.eval()

    with torch.no_grad():
        # Use the batch from previous step
        outputs = model(batch["input_ids"], batch["attention_mask"])

    assert "logits" in outputs
    assert "aux_logits" in outputs

    # Logits should be (Batch,)
    assert outputs["logits"].shape == (Config.TRAIN_BATCH_SIZE,)
    # Aux logits should be (Batch, NumIdentities)
    assert outputs["aux_logits"].shape == (
        Config.TRAIN_BATCH_SIZE,
        len(Config.IDENTITY_COLUMNS),
    )

    print("Model forward pass successful. Output shapes correct.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    trainer = Trainer(train_loader, val_loader)
    trainer.train()

    # Verify model was saved
    assert os.path.exists(
        Config.OUTPUT_MODEL_PATH
    ), "Model file was not saved after training."
    print("Training complete. Best model saved.")

    # --------------------------------------------------------------------------
    # 5. Metric Calculation Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Bias Metrics Logic...")

    # Create synthetic data
    # 10 samples: 5 toxic, 5 non-toxic
    # Identity 'male': present in first 5
    n_samples = 10
    ids = np.arange(n_samples)
    targets = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])  # Mixed toxicity
    preds_perfect = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1])

    df_metric = pd.DataFrame(
        {"id": ids, "target": targets, "prediction": preds_perfect}
    )

    # Add identity columns
    for col in Config.IDENTITY_COLUMNS:
        df_metric[col] = 0.0

    # Set 'male' identity for first 5 rows
    df_metric["male"] = np.concatenate([np.ones(5), np.zeros(5)])

    metrics = compute_bias_metrics(
        df_metric,
        target_col="target",
        pred_col="prediction",
        identity_columns=Config.IDENTITY_COLUMNS,
    )

    print(f"Calculated Score (Perfect Preds): {metrics['score']:.4f}")

    # With perfect predictions, AUCs should be 1.0
    # Note: If a subgroup has only one class, AUC is NaN -> handled in code (returns nan or 0.5 depending on logic)
    # In our synthetic data:
    # Male subgroup (first 5): Targets [1, 1, 1, 0, 0] -> Both classes present -> AUC 1.0
    assert (
        metrics["overall_auc"] > 0.99
    ), "Overall AUC should be ~1.0 for perfect predictions"
    assert (
        metrics["per_identity_metrics"]["male"]["subgroup_auc"] > 0.99
    ), "Subgroup AUC should be ~1.0"

    # --------------------------------------------------------------------------
    # 6. Inference on Test Set
    # --------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    test_loader = get_test_dataloader(load_cached_data=False)

    # Load best model
    model.load_state_dict(torch.load(Config.OUTPUT_MODEL_PATH, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_ids = []
    test_preds = []

    print("Predicting...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["id"]

            # Simple trimming for speed
            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()

            test_ids.extend(ids.numpy())
            test_preds.extend(probs)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "prediction": test_preds})

    print(f"Inference complete. Generated {len(submission_df)} predictions.")
    print(submission_df.head())

    # Save submission
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    assert os.path.exists(sub_path)

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
