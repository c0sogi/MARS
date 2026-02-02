import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
from functools import partial

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Monkey-patch tqdm to disable progress bars as requested ---
import tqdm


def no_op_tqdm(*args, **kwargs):
    # Return the iterable if provided, else an empty list
    if args:
        return args[0]
    if "iterable" in kwargs:
        return kwargs["iterable"]
    return []


# Replace tqdm in modules that use it
tqdm.tqdm = no_op_tqdm
sys.modules["tqdm"].tqdm = no_op_tqdm

# --- Import Library Components ---
from library.config import Config
from library.utils import (
    calculate_shortest_arc,
    calculate_euclidean_distance,
    calculate_closing_speed,
    calculate_log_distance,
)
from library.features import generate_features, POSITIONS_VOCAB
from library.dataset import NFLContactDataset
from library.model import EGRVNet
from library.loss import FocalLoss
from library.trainer import Trainer


def run_demo():
    print("--- Starting NFL Contact Detection Demo ---")

    # 1. Setup & Configuration Override
    # We use a specific working directory for this demo to avoid conflicts
    # and set parameters for a quick run.
    Config.WORKING_DIR = "./working/demo_run"
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"Configuration set. Working dir: {Config.WORKING_DIR}")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utility Functions ---")

    # Test Shortest Arc (0 vs 350 should be 10, not 350)
    angle_a = np.array([0, 10, 350])
    angle_b = np.array([350, 20, 10])
    arc = calculate_shortest_arc(angle_a, angle_b)
    expected_arc = np.array([10, 10, 20])
    assert np.allclose(arc, expected_arc), f"Shortest arc failed: {arc}"
    print("utils.calculate_shortest_arc: OK")

    # Test Euclidean Distance
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    dist = calculate_euclidean_distance(p1[0], p1[1], p2[0], p2[1])
    assert np.isclose(dist, 5.0), f"Euclidean distance failed: {dist}"
    print("utils.calculate_euclidean_distance: OK")

    # Test Closing Speed
    # Objects moving towards each other along X axis
    # Obj1 at 0, vel=10; Obj2 at 100, vel=-10. Relative vel = 20 closing.
    vx1, vy1 = 10, 0
    vx2, vy2 = -10, 0
    x1, y1 = 0, 0
    x2, y2 = 100, 0
    cs = calculate_closing_speed(vx1, vy1, vx2, vy2, x1, y1, x2, y2)
    # Closing speed should be positive when closing distance
    assert cs > 0, f"Closing speed logic error: {cs}"
    print("utils.calculate_closing_speed: OK")

    # 3. Verify Feature Generation
    print("\n--- Verifying Feature Generation (Debug Mode) ---")
    # This will load a small subset of metadata (5000 rows) and process it
    X_kin, X_vis, X_cat, y, ids = generate_features(split="train", debug=True)

    print(f"Generated Feature Shapes:")
    print(f"  Kinematic: {X_kin.shape}")
    print(f"  Visual:    {X_vis.shape}")
    print(f"  Categorical: {X_cat.shape}")
    print(f"  Targets:   {y.shape}")

    assert len(X_kin) == len(y), "Mismatch between features and targets"
    assert (
        X_cat.shape[1] == 4
    ), "Categorical features should have 4 columns (P1_Pos, P1_Team, P2_Pos, P2_Team)"

    # 4. Verify Dataset Class
    print("\n--- Verifying Dataset Class ---")
    # Initialize train dataset (fits scaler)
    train_ds = NFLContactDataset(split="train", debug=True)
    assert len(train_ds) > 0, "Train dataset is empty"

    # Fetch one sample
    sample = train_ds[0]
    required_keys = ["kinematic", "visual", "categorical", "target", "contact_id"]
    for k in required_keys:
        assert k in sample, f"Dataset sample missing key: {k}"

    # Check tensor types
    assert isinstance(sample["kinematic"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)
    print(f"Dataset sample retrieval: OK (ID: {sample['contact_id']})")

    # Initialize validation dataset (loads scaler)
    # Note: We must ensure validation data exists/can be processed.
    # debug=True samples from metadata, so it should work if metadata exists.
    val_ds = NFLContactDataset(split="validation", debug=True)
    assert len(val_ds) > 0, "Validation dataset is empty"
    print("Dataset scaler persistence (Train -> Val): OK")

    # 5. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = EGRVNet()
    model.eval()

    # Create dummy batch
    # Dimensions must match Config
    # Kinematic: (Features * (2*Window + 1))
    # Visual: (Features * 2 * (2*Window + 1))
    # Cat: 4

    # Model expects raw kinematic input (embeddings are handled internally)
    dummy_kin = torch.randn(2, Config.KINEMATIC_INPUT_DIM)
    dummy_vis = torch.randn(2, Config.VISUAL_INPUT_DIM)

    # Construct categorical inputs with valid ranges: [P1_Pos, P1_Team, P2_Pos, P2_Team]
    p1_pos = torch.randint(0, len(POSITIONS_VOCAB), (2, 1))
    p1_team = torch.randint(0, 3, (2, 1))  # Team size is 3
    p2_pos = torch.randint(0, len(POSITIONS_VOCAB), (2, 1))
    p2_team = torch.randint(0, 3, (2, 1))
    dummy_cat = torch.cat([p1_pos, p1_team, p2_pos, p2_team], dim=1)

    try:
        output = model(dummy_kin, dummy_vis, dummy_cat)
        assert output.shape == (2, 1), f"Model output shape mismatch: {output.shape}"
        print("Model forward pass: OK")
    except Exception as e:
        print(f"Model forward pass failed: {e}")
        raise e

    # 6. Verify Loss Function
    print("\n--- Verifying Loss Function ---")
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    dummy_logits = torch.tensor(
        [[2.0], [-2.0]], requires_grad=True
    )  # One likely pos, one neg
    dummy_targets = torch.tensor([[1.0], [0.0]])

    loss = criterion(dummy_logits, dummy_targets)
    loss.backward()
    assert loss.item() > 0, "Loss should be positive"
    print(f"Focal Loss computation: OK ({loss.item():.4f})")

    # 7. Verify Trainer (Training Loop)
    print("\n--- Verifying Trainer (Fit & Predict) ---")
    trainer = Trainer(device="cpu")  # Use CPU for simple demo stability

    print("Running Trainer.fit() with debug=True (1 epoch)...")
    best_mcc = trainer.fit(epochs=1, debug=True)

    print(f"Training complete. Best MCC: {best_mcc}")
    assert os.path.exists(trainer.best_model_path), "Best model file was not saved"
    assert os.path.exists(trainer.best_thresh_path), "Best threshold file was not saved"

    print("Running Trainer.predict_and_submit()...")
    # This will use the test set defined in metadata/test.csv
    trainer.predict_and_submit()

    submission_file = "./submission/submission.csv"
    assert os.path.exists(submission_file), "Submission file not found"

    df_sub = pd.read_csv(submission_file)
    print(f"Submission generated with {len(df_sub)} rows.")
    assert "contact_id" in df_sub.columns and "contact" in df_sub.columns

    # Check if we have rows
    if len(df_sub) == 0:
        print(
            "Warning: Submission file is empty. This might be due to empty test metadata in demo."
        )
    else:
        print("Submission content check: OK")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
