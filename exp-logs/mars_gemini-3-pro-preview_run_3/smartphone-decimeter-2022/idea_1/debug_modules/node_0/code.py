import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Configuration Override
# -----------------------------------------------------------------------------
# We patch the config module before importing other library modules to ensure
# they use our modified settings for a fast demonstration.
import library.config as config

print("[1] Configuration Override")
# Limit data size for speed
config.DEBUG_SAMPLE_SIZE = 500
# Reduce epochs and batch size for demonstration
config.NUM_EPOCHS = 1
config.BATCH_SIZE = 32
# Ensure we use the working directory for outputs
config.WORK_DIR = "./working/demo_run"
config.MODEL_SAVE_PATH = os.path.join(config.WORK_DIR, "model_best.pth")
os.makedirs(config.WORK_DIR, exist_ok=True)

print(f"DEBUG_SAMPLE_SIZE: {config.DEBUG_SAMPLE_SIZE}")
print(f"NUM_EPOCHS: {config.NUM_EPOCHS}")
print(f"WORK_DIR: {config.WORK_DIR}")

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
from library.utils import geodetic_to_ecef, ecef_to_geodetic
from library.data import load_dataset
from library.model import ResidualMLP
from library.train import run_training
from library.inference import predict_and_submit

# -----------------------------------------------------------------------------
# 3. Verify Utility Functions
# -----------------------------------------------------------------------------
print("\n[2] Verifying Coordinate Transformations")
# Test point: Googleplex (approximate)
lat_orig, lon_orig, alt_orig = 37.422, -122.084, 10.0

# Convert to ECEF
x, y, z = geodetic_to_ecef(lat_orig, lon_orig, alt_orig)
print(
    f"Geodetic ({lat_orig}, {lon_orig}, {alt_orig}) -> ECEF ({x:.2f}, {y:.2f}, {z:.2f})"
)

# Convert back to Geodetic
lat_new, lon_new, alt_new = ecef_to_geodetic(x, y, z)
print(f"ECEF -> Geodetic ({lat_new:.6f}, {lon_new:.6f}, {alt_new:.6f})")

# Assertions
assert np.isclose(lat_orig, lat_new, atol=1e-5), "Latitude mismatch"
assert np.isclose(lon_orig, lon_new, atol=1e-5), "Longitude mismatch"
assert np.isclose(alt_orig, alt_new, atol=1e-3), "Altitude mismatch"
print("Coordinate transformation verification passed.")

# -----------------------------------------------------------------------------
# 4. Data Loading and Processing
# -----------------------------------------------------------------------------
print("\n[3] Loading and Processing Training Data")
# We set load_cached_data=False to force processing of the small debug sample
train_dataset = load_dataset(mode="train", load_cached_data=False)

print(f"Train Dataset Size: {len(train_dataset)}")
if len(train_dataset) > 0:
    features, targets = train_dataset[0]
    print(f"Feature Shape: {features.shape}")
    print(f"Target Shape: {targets.shape}")

    # Verify dimensions match config
    assert (
        features.shape[0] == config.INPUT_DIM
    ), f"Expected input dim {config.INPUT_DIM}, got {features.shape[0]}"
    assert (
        targets.shape[0] == config.OUTPUT_DIM
    ), f"Expected output dim {config.OUTPUT_DIM}, got {targets.shape[0]}"
else:
    print("Warning: Train dataset is empty (likely due to missing files in sample).")

# -----------------------------------------------------------------------------
# 5. Model Initialization and Forward Pass
# -----------------------------------------------------------------------------
print("\n[4] Model Initialization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualMLP(input_dim=config.INPUT_DIM).to(device)
print(f"Model created on {device}")

if len(train_dataset) > 0:
    # Create a dummy batch
    dummy_input = torch.stack(
        [train_dataset[i][0] for i in range(min(4, len(train_dataset)))]
    ).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    assert output.shape == (
        dummy_input.shape[0],
        config.OUTPUT_DIM,
    ), "Model output shape mismatch"
    print("Forward pass successful.")

# -----------------------------------------------------------------------------
# 6. Training Pipeline
# -----------------------------------------------------------------------------
print("\n[5] Running Training Pipeline")
# This will load train/val datasets (processing them if not cached/forced) and train the model
trained_model = run_training(
    load_cached_data=False,
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=1e-3,
)

# Verify model checkpoint exists
if os.path.exists(config.MODEL_SAVE_PATH):
    print(f"Model checkpoint found at {config.MODEL_SAVE_PATH}")
else:
    raise FileNotFoundError("Model checkpoint was not saved.")

# -----------------------------------------------------------------------------
# 7. Inference and Submission
# -----------------------------------------------------------------------------
print("\n[6] Running Inference and Submission")
# This loads test data, loads the saved model, predicts, and writes submission.csv
predict_and_submit(load_cached_data=False)

submission_file = os.path.join(config.SUBMISSION_DIR, "submission.csv")
if os.path.exists(submission_file):
    print(f"Submission file created at {submission_file}")

    # Verify submission format
    df_sub = pd.read_csv(submission_file)
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if all(col in df_sub.columns for col in required_cols):
        print("Submission format verified: All required columns present.")
        print(f"Submission rows: {len(df_sub)}")
    else:
        raise ValueError(f"Submission missing columns. Found: {df_sub.columns}")
else:
    raise FileNotFoundError("Submission file not generated.")

print("\n[7] Demonstration Complete")
