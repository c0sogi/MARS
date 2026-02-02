import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.feature_engineering import generate_svd_features
from library.dataset import InsultDataset
from library.models import HybridModel
from library.awp import AWP
from library import pipeline
from library import stacking


def run_demo():
    # =========================================================================
    # 1. Setup and Configuration
    # =========================================================================
    print(">>> [1/6] Setting up Configuration for Demo...")

    # Enable debug mode and override settings for speed
    Config.setup(debug=True)
    Config.device = torch.device(
        "cpu"
    )  # Force CPU for simple demo stability/simplicity
    if torch.cuda.is_available():
        Config.device = torch.device("cuda")

    # Use a tiny model for rapid execution
    tiny_model = "prajjwal1/bert-tiny"
    Config.model_deberta = tiny_model
    Config.models = [tiny_model]

    # Reduce hyperparameters
    Config.epochs = 1
    Config.n_folds = 2
    Config.trn_folds = [0]  # Only run fold 0
    Config.batch_size = 4
    Config.svd_components = 16  # Reduce SVD dim for tiny dataset
    Config.working_dir = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.working_dir, "cache")
    Config.MODEL_DIR = os.path.join(Config.working_dir, "models")
    Config.OUTPUT_DIR = os.path.join(Config.working_dir, "outputs")

    # Create directories
    for d in [
        Config.working_dir,
        Config.CACHE_DIR,
        Config.MODEL_DIR,
        Config.OUTPUT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    seed_everything(Config.seed)

    # =========================================================================
    # 2. Prepare Demo Data (Subset)
    # =========================================================================
    print(">>> [2/6] Preparing Demo Data Subset...")

    # Load original metadata
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/validation.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Take a tiny subset (e.g., 20 samples)
    train_subset = train_full.head(20).copy()
    val_subset = val_full.head(10).copy()
    test_subset = test_full.head(10).copy()

    # Save to working directory
    demo_train_path = os.path.join(Config.working_dir, "demo_train.csv")
    demo_val_path = os.path.join(Config.working_dir, "demo_val.csv")
    demo_test_path = os.path.join(Config.working_dir, "demo_test.csv")

    train_subset.to_csv(demo_train_path, index=False)
    val_subset.to_csv(demo_val_path, index=False)
    test_subset.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_PATH = demo_train_path
    Config.VAL_PATH = demo_val_path
    Config.TEST_PATH = demo_test_path

    print(f"    Train subset shape: {train_subset.shape}")
    print(f"    Test subset shape: {test_subset.shape}")

    # =========================================================================
    # 3. Verify Feature Engineering
    # =========================================================================
    print(">>> [3/6] Verifying Feature Engineering (SVD)...")

    train_texts = train_subset["Comment"].fillna("").astype(str).values
    val_texts = val_subset["Comment"].fillna("").astype(str).values
    test_texts = test_subset["Comment"].fillna("").astype(str).values

    # Generate SVD features
    # Note: We use a unique fold_idx to avoid reading existing cache if any
    train_svd, val_svd, test_svd = generate_svd_features(
        train_texts, val_texts, test_texts, fold_idx=999, load_cached_data=False
    )

    # Assertions
    assert train_svd.shape == (
        len(train_subset),
        Config.svd_components,
    ), f"Train SVD shape mismatch. Expected {(len(train_subset), Config.svd_components)}, got {train_svd.shape}"
    assert test_svd.shape == (
        len(test_subset),
        Config.svd_components,
    ), f"Test SVD shape mismatch. Expected {(len(test_subset), Config.svd_components)}, got {test_svd.shape}"

    print("    SVD Feature generation successful.")

    # =========================================================================
    # 4. Verify Dataset and Model Logic
    # =========================================================================
    print(">>> [4/6] Verifying Dataset and HybridModel...")

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)

    # Instantiate Dataset
    ds = InsultDataset(
        texts=train_texts,
        svd_features=train_svd,
        tokenizer=tokenizer,
        labels=train_subset["Insult"].values,
        max_len=32,  # Short max_len for demo
    )

    # Check __getitem__
    sample = ds[0]
    required_keys = ["input_ids", "attention_mask", "svd_features", "labels"]
    for key in required_keys:
        assert key in sample, f"Dataset sample missing key: {key}"

    assert sample["input_ids"].shape == (32,), "Incorrect input_ids shape"
    assert sample["svd_features"].shape == (
        Config.svd_components,
    ), "Incorrect svd_features shape"

    # Instantiate Model
    model = HybridModel(tiny_model, svd_dim=Config.svd_components, pretrained=True)
    model.to(Config.device)
    model.eval()

    # Create a batch
    batch_input_ids = sample["input_ids"].unsqueeze(0).to(Config.device)
    batch_mask = sample["attention_mask"].unsqueeze(0).to(Config.device)
    batch_svd = sample["svd_features"].unsqueeze(0).to(Config.device)

    # Forward pass
    with torch.no_grad():
        output = model(batch_input_ids, batch_mask, batch_svd)

    assert output.shape == (
        1,
        1,
    ), f"Model output shape mismatch. Expected (1, 1), got {output.shape}"
    print("    Model instantiation and forward pass successful.")

    # =========================================================================
    # 5. Verify Adversarial Weight Perturbation (AWP)
    # =========================================================================
    print(">>> [5/6] Verifying AWP Logic...")

    # We need an optimizer and a model with gradients
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a dummy backward pass to populate .grad attributes
    output = model(batch_input_ids, batch_mask, batch_svd)
    loss = nn.BCEWithLogitsLoss()(output, torch.tensor([[1.0]], device=Config.device))
    loss.backward()

    # Capture original weight of a specific parameter (e.g., classifier weight)
    # We pick the classifier layer: model.fc.weight
    orig_weight = model.fc.weight.data.clone()

    # Initialize AWP
    awp = AWP(
        model, optimizer, adv_lr=0.1, adv_eps=1.0
    )  # High LR to ensure visible change

    # Attack
    awp.attack()
    perturbed_weight = model.fc.weight.data.clone()

    # Check that weights changed
    diff = torch.norm(perturbed_weight - orig_weight).item()
    assert diff > 0, "AWP Attack did not perturb the weights!"

    # Restore
    awp.restore()
    restored_weight = model.fc.weight.data.clone()

    # Check that weights are restored
    restore_diff = torch.norm(restored_weight - orig_weight).item()
    assert restore_diff < 1e-6, "AWP Restore failed to recover original weights!"

    print("    AWP logic verified: Weights perturbed and restored correctly.")

    # =========================================================================
    # 6. Run Pipeline and Stacking
    # =========================================================================
    print(">>> [6/6] Running Pipeline (K-Fold CV) and Stacking...")

    # Run the pipeline wrapper
    # This will run training on the tiny subset for 1 epoch (Config.epochs=1, Config.trn_folds=[0])
    oof_preds, test_preds, y_labels = pipeline.run_kfold_cv(tiny_model)

    # Verify outputs
    # Note: oof_preds length corresponds to the concatenated train+val set used in pipeline
    # In pipeline.py, it concatenates Config.TRAIN_PATH and Config.VAL_PATH.
    total_train_len = len(train_subset) + len(val_subset)
    assert len(oof_preds) == total_train_len, "OOF preds length mismatch"
    assert len(test_preds) == len(test_subset), "Test preds length mismatch"

    print(
        f"    Pipeline complete. OOF Mean: {oof_preds.mean():.4f}, Test Mean: {test_preds.mean():.4f}"
    )

    # Stacking Demo
    # We simulate having two models by reusing the predictions
    oof_dict = {"model_a": oof_preds, "model_b": oof_preds * 0.9}  # Slight variation
    test_dict = {"model_a": test_preds, "model_b": test_preds * 0.9}

    # Train Meta Learner
    meta_model = stacking.train_meta_learner(oof_dict, y_labels)

    # Predict
    final_preds = stacking.predict_meta_learner(meta_model, test_dict)

    assert len(final_preds) == len(test_subset), "Final predictions length mismatch"
    assert (final_preds >= 0).all() and (
        final_preds <= 1
    ).all(), "Predictions out of range [0,1]"

    print("    Stacking verified.")
    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
