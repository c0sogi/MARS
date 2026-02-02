import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, StandardScaler, RBFExpansion, compute_metrics
from library.data import get_loaders, collate_graphs, AtomGraphDataset
from library.model import RAGLUNet
from library.train import Trainer


def run_demo():
    print("Initializing Demo Run...")

    # 1. Setup Configuration for Demo
    # We override paths and hyperparameters to run a fast, small-scale test.
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    Config.WORKING_DIR = demo_working_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Use separate cache files for demo to avoid conflicts/overwriting real cache
    Config.TRAIN_GRAPHS_CACHE = os.path.join(
        demo_working_dir, "cache", "train_graphs.npz"
    )
    Config.VAL_GRAPHS_CACHE = os.path.join(demo_working_dir, "cache", "val_graphs.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(
        demo_working_dir, "cache", "test_graphs.npz"
    )
    Config.TARGET_SCALER_CACHE = os.path.join(
        demo_working_dir, "cache", "target_scaler.npz"
    )

    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce model size and training duration for speed
    Config.HIDDEN_DIM = 32
    Config.NUM_LAYERS = 2
    Config.NUM_RBF = 20
    Config.BATCH_SIZE = 4
    Config.MAX_EPOCHS = 2
    Config.PATIENCE = 2

    Config.setup()
    set_seed(Config.SEED)

    # 2. Create Tiny Metadata Subsets
    # We read the original metadata and sample a few rows to create a tiny dataset.
    print("\nCreating tiny metadata subsets...")
    metadata_demo_dir = os.path.join(demo_working_dir, "metadata")
    os.makedirs(metadata_demo_dir, exist_ok=True)

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Sample small subsets (ensure we pick IDs that exist, which these do)
    demo_train = orig_train.head(20)
    demo_val = orig_val.head(10)
    demo_test = orig_test.head(10)

    # Save to demo location
    Config.TRAIN_METADATA_PATH = os.path.join(metadata_demo_dir, "train_metadata.csv")
    Config.VAL_METADATA_PATH = os.path.join(metadata_demo_dir, "val_metadata.csv")
    Config.TEST_METADATA_PATH = os.path.join(metadata_demo_dir, "test_metadata.csv")

    demo_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    demo_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    demo_test.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(f"Demo Train size: {len(demo_train)}")
    print(f"Demo Val size: {len(demo_val)}")
    print(f"Demo Test size: {len(demo_test)}")

    # 3. Test Data Loading and Processing
    print("\nTesting Data Loading...")
    # This will trigger process_graphs, which reads geometry files and builds graphs
    train_loader, val_loader, test_loader, scaler = get_loaders(
        load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    # Verify Scaler
    print("Verifying Scaler...")
    assert scaler.mean is not None
    assert scaler.std is not None
    assert scaler.mean.shape[0] == 2

    # Verify Batch Structure
    print("Verifying Batch Structure...")
    batch = next(iter(train_loader))
    assert hasattr(batch, "x")
    assert hasattr(batch, "edge_index")
    assert hasattr(batch, "edge_attr")
    assert hasattr(batch, "y")
    assert hasattr(batch, "batch")

    # Check shapes
    # x should be [num_nodes]
    # edge_index should be [2, num_edges]
    # edge_attr should be [num_edges, 1]
    # y should be [batch_size, 2]
    # batch should be [num_nodes]
    print(f"Batch x shape: {batch.x.shape}")
    print(f"Batch edge_index shape: {batch.edge_index.shape}")
    print(f"Batch y shape: {batch.y.shape}")

    assert batch.y.shape == (Config.BATCH_SIZE, 2)
    assert batch.x.dim() == 1
    assert batch.edge_index.shape[0] == 2

    # 4. Test Model Initialization and Forward Pass
    print("\nTesting Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAGLUNet(config=Config).to(device)

    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Expected (4, 2), got {output.shape}"

    # 5. Test Utils
    print("\nTesting Utils...")
    # RBF Expansion
    rbf = RBFExpansion(vmin=0.0, vmax=5.0, bins=10)
    dist = torch.tensor([0.0, 2.5, 5.0])
    rbf_out = rbf(dist)
    assert rbf_out.shape == (3, 10)

    # Metrics
    dummy_preds = np.array([[1.0, 2.0], [0.5, 0.5]])
    dummy_targets = np.array([[1.1, 1.9], [0.5, 0.5]])
    metric = compute_metrics(dummy_preds, dummy_targets)
    print(f"Computed RMSLE: {metric}")
    assert metric >= 0

    # 6. Test Training Loop
    print("\nTesting Trainer...")
    trainer = Trainer(model, scaler, device)

    # Run fit (short run)
    trainer.fit(
        train_loader, val_loader, max_epochs=Config.MAX_EPOCHS, patience=Config.PATIENCE
    )

    # Check if checkpoint exists
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."

    # 7. Test Prediction
    print("\nTesting Prediction...")
    predictions = trainer.predict(test_loader)
    print(f"Predictions shape: {predictions.shape}")

    assert predictions.shape == (
        len(demo_test),
        2,
    ), f"Expected ({len(demo_test)}, 2), got {predictions.shape}"

    # 8. Generate Submission
    print("\nGenerating Demo Submission...")
    submission_df = pd.DataFrame(
        {
            "id": demo_test["id"],
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\nDemo Run Completed Successfully!")


if __name__ == "__main__":
    run_demo()
