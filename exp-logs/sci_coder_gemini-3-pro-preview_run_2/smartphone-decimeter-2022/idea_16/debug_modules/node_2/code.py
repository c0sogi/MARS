import os
import sys
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.trainer as trainer_lib
import library.inference as inference_lib


def main():
    print("==================================================")
    print("      GNSS Positioning Pipeline Demonstration     ")
    print("==================================================")

    # 1. Setup
    print("\n[1] Setting up environment...")
    config.set_seed(42)

    # Ensure working directories exist
    if not os.path.exists(config.CACHE_DIR):
        os.makedirs(config.CACHE_DIR)
    if not os.path.exists(config.SUBMISSION_DIR):
        os.makedirs(config.SUBMISSION_DIR)

    print(f"    Cache Directory: {config.CACHE_DIR}")
    print(f"    Submission Directory: {config.SUBMISSION_DIR}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test Haversine
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 0.0, 1.0
    dist = utils.haversine_distance(lat1, lon1, lat2, lon2)
    # Approx meters for 1 deg lon at equator is ~111km
    print(f"    Haversine Distance (0,0) -> (0,1): {dist:.2f} m")
    assert 111000 < dist < 112000, "Haversine distance calculation is incorrect."

    # Test Coordinate Conversion
    dx, dy = utils.wgs84_to_meters_relative(lat1, lon1, lat2, lon2)
    print(f"    Relative Meters: dx={dx:.2f}, dy={dy:.2f}")

    rec_lat, rec_lon = utils.meters_to_wgs84_relative(lat1, lon1, dx, dy)
    print(f"    Reconstructed: lat={rec_lat:.6f}, lon={rec_lon:.6f}")

    assert np.isclose(rec_lat, lat2), "Latitude reconstruction failed."
    assert np.isclose(rec_lon, lon2), "Longitude reconstruction failed."
    print("    Utils verification passed.")

    # 3. Demonstrate Data Loading
    print("\n[3] Demonstrating Data Loading...")
    # Load a very small subset of training data (max 100 samples)
    # We disable caching to force data processing logic to run
    try:
        train_dataset, scaler = data_loader.load_dataset(
            mode="train", max_samples=100, load_cached_data=False
        )
        print(f"    Successfully loaded {len(train_dataset)} training samples.")

        # Inspect a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            traj_shape = sample["traj_feat"].shape
            sky_shape = sample["sky_feat"].shape
            target_shape = sample["target"].shape

            print(
                f"    Sample shapes - Trajectory: {traj_shape}, Sky: {sky_shape}, Target: {target_shape}"
            )

            assert (
                traj_shape[0] == config.INPUT_DIM_TRAJ
            ), f"Trajectory dim mismatch: {traj_shape[0]} != {config.INPUT_DIM_TRAJ}"
            assert (
                sky_shape[0] == config.INPUT_DIM_SKY
            ), f"Sky dim mismatch: {sky_shape[0]} != {config.INPUT_DIM_SKY}"
            assert (
                target_shape[0] == config.OUTPUT_DIM
            ), f"Target dim mismatch: {target_shape[0]} != {config.OUTPUT_DIM}"
        else:
            print(
                "    Warning: Dataset is empty (possibly due to small max_samples filtering)."
            )

    except Exception as e:
        print(f"    Data loading failed: {e}")
        raise e

    # 4. Demonstrate Model Initialization and Forward Pass
    print("\n[4] Demonstrating Model...")
    model = model_lib.RelativeWindowedMLP()
    model.to(config.DEVICE)
    print(f"    Model initialized on {config.DEVICE}.")

    # Create a dummy batch
    batch_size = 4
    dummy_traj = torch.randn(batch_size, config.INPUT_DIM_TRAJ).to(config.DEVICE)
    dummy_sky = torch.randn(batch_size, config.INPUT_DIM_SKY).to(config.DEVICE)

    with torch.no_grad():
        output = model(dummy_traj, dummy_sky)

    print(f"    Forward pass output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        config.OUTPUT_DIM,
    ), "Model output shape is incorrect."
    print("    Model verification passed.")

    # 5. Demonstrate Training Pipeline
    print("\n[5] Running Training Pipeline (Short Run)...")
    # Train for 1 epoch on a small subset
    # This will save the best model to cache
    trained_scaler = trainer_lib.run_training(
        max_epochs=1, max_samples=100, load_cached=False
    )

    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Training complete. Model saved to {best_model_path}")
    else:
        raise FileNotFoundError("Best model file was not generated.")

    # 6. Demonstrate Inference Pipeline
    print("\n[6] Running Inference Pipeline...")

    # We mock load_dataset inside inference.py to limit the number of test samples processed.
    # This ensures the demonstration runs quickly.

    original_load_dataset = data_loader.load_dataset

    def mocked_load_dataset_for_inference(
        mode="test", scaler=None, max_samples=None, load_cached_data=True
    ):
        if mode == "test":
            print(
                "    [Mock] Limiting test dataset to 50 samples for demonstration speed."
            )
            # We pass max_samples=50 to limit processing time
            return original_load_dataset(
                mode=mode,
                scaler=scaler,
                max_samples=50,
                load_cached_data=load_cached_data,
            )
        return original_load_dataset(
            mode=mode,
            scaler=scaler,
            max_samples=max_samples,
            load_cached_data=load_cached_data,
        )

    # Patch the load_dataset function used in library.inference
    try:
        with patch(
            "library.inference.load_dataset",
            side_effect=mocked_load_dataset_for_inference,
        ):
            inference_lib.generate_predictions(scaler=trained_scaler, load_cached=False)

        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        if os.path.exists(submission_path):
            df_sub = pd.read_csv(submission_path)
            print(f"    Inference complete. Submission generated at {submission_path}")
            print(f"    Submission rows: {len(df_sub)}")
            print("    Submission Head:")
            print(df_sub.head())
        else:
            raise FileNotFoundError("Submission file was not generated.")

    except Exception as e:
        print(f"    Inference failed: {e}")
        raise e

    print("\n==================================================")
    print("      Demonstration Completed Successfully        ")
    print("==================================================")


if __name__ == "__main__":
    main()
