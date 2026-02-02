import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_map5
from library.dataset import get_dataloaders
from library.model import WhaleModel
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(42)

    print("Initializing Demo Configuration...")

    # Modify Config for a fast demo run
    Config.debug = True
    Config.epochs = 1
    Config.batch_size = 4  # Small batch size for demo
    Config.working_dir = "./working/demo_run"

    # Update dependent paths in Config since they were initialized at import time
    os.makedirs(Config.working_dir, exist_ok=True)
    Config.model_save_path = os.path.join(Config.working_dir, "model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")
    Config.train_embeddings_path = os.path.join(Config.working_dir, "train_emb.npy")
    Config.train_labels_path = os.path.join(Config.working_dir, "train_lbl.npy")
    Config.val_embeddings_path = os.path.join(Config.working_dir, "val_emb.npy")
    Config.val_labels_path = os.path.join(Config.working_dir, "val_lbl.npy")
    Config.test_embeddings_path = os.path.join(Config.working_dir, "test_emb.npy")
    Config.test_names_path = os.path.join(Config.working_dir, "test_names.npy")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading DataLoaders (Debug Mode)...")
    train_loader, gallery_loader, val_loader, test_loader, encoder = get_dataloaders(
        debug=Config.debug,
        load_cached_data=False,  # Force re-creation for demo
        batch_size=Config.batch_size,
        num_workers=2,  # Reduce workers for small demo
    )

    # Verify DataLoaders
    try:
        images, labels = next(iter(train_loader))
        print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")
        assert images.shape[0] == Config.batch_size
        assert images.shape[1] == 3  # Channels
        assert images.shape[2] == Config.input_size
        assert images.shape[3] == Config.input_size
    except StopIteration:
        print("Error: Train loader is empty.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("Verifying Model Architecture...")
    num_classes = len(encoder.classes_)
    model = WhaleModel(num_classes=num_classes).to(Config.device)

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.input_size, Config.input_size).to(
        Config.device
    )
    dummy_targets = torch.tensor([0, 1]).to(Config.device)

    # Test Training Forward Pass (returns logits)
    model.train()
    logits = model(dummy_input, targets=dummy_targets)
    assert logits.shape == (
        2,
        num_classes,
    ), f"Expected logits shape (2, {num_classes}), got {logits.shape}"

    # Test Inference Forward Pass (returns embeddings)
    model.eval()
    with torch.no_grad():
        embeddings = model(dummy_input)
    assert embeddings.shape == (
        2,
        Config.embedding_dim,
    ), f"Expected embeddings shape (2, {Config.embedding_dim}), got {embeddings.shape}"

    # Verify Normalization (L2 norm should be approx 1.0)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), "Embeddings are not normalized."

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric Verification
    # -------------------------------------------------------------------------
    print("Verifying Metric (MAP@5)...")
    # Ground truth: A, B
    # Preds for A: [A, C, D, E, F] -> Rank 0 -> Score 1.0
    # Preds for B: [C, A, B, D, E] -> Rank 2 -> Score 1/3
    # Mean = (1.0 + 0.333) / 2 = 0.666...
    gt = ["w_A", "w_B"]
    preds = [["w_A", "w_C", "w_D", "w_E", "w_F"], ["w_C", "w_A", "w_B", "w_D", "w_E"]]
    score = compute_map5(gt, preds)
    expected_score = (1.0 + (1.0 / 3.0)) / 2.0
    assert (
        abs(score - expected_score) < 1e-6
    ), f"MAP@5 calculation incorrect. Got {score}, expected {expected_score}"
    print(f"Metric verification passed. Score: {score:.4f}")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("Starting Training Loop...")
    trainer = Trainer(train_loader, gallery_loader, val_loader, encoder)

    # Run fit (1 epoch as per modified Config)
    trainer.fit()

    # Verify model file creation
    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.model_save_path}"
        )
    print("Training complete and model saved.")

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline
    # -------------------------------------------------------------------------
    print("Starting Inference Pipeline...")
    inference = InferenceEngine(checkpoint_path=Config.model_save_path)

    # Run prediction
    # Note: We use load_cached_data=False to ensure the pipeline runs fully
    inference.predict_with_qe(
        test_loader, gallery_loader, encoder, load_cached_data=False
    )

    # -------------------------------------------------------------------------
    # 7. Output Validation
    # -------------------------------------------------------------------------
    print("Validating Submission File...")
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    df_sub = pd.read_csv(Config.submission_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    assert "Image" in df_sub.columns
    assert "Id" in df_sub.columns

    # Check content format (space separated strings)
    sample_pred = df_sub.iloc[0]["Id"]
    assert isinstance(sample_pred, str)
    assert len(sample_pred.split()) <= 5

    print("Submission validation passed.")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
