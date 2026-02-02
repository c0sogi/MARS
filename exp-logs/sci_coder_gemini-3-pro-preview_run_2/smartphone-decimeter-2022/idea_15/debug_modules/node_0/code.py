import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import library modules
from library.config import Config
from library.data_loader import get_dataloaders, DataProcessor, GNSSDataset
from library.model import SARTransformer
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import wls_to_meters, meters_to_wls, haversine_loss


def main():
    print(">>> Starting Library Usage Demonstration...")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce training parameters for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.WINDOW_SIZE = 10  # Slightly smaller window

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}")

    # ---------------------------------------------------------
    # 2. Data Loading (Subsampled)
    # ---------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # We use a very small sample fraction to ensure this runs quickly
    # This will trigger the DataProcessor to load metadata, sample trips,
    # read GNSS/IMU files, aggregate, and window them.
    try:
        train_loader, val_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, sample_frac=0.02
        )
        print("    DataLoaders created successfully.")
    except Exception as e:
        print(f"    Error creating dataloaders: {e}")
        # If data loading fails (e.g. no input data), we can't proceed with training
        # But we can still demonstrate model instantiation.
        train_loader = None
        val_loader = None

    if train_loader and len(train_loader) > 0:
        # Fetch one batch to verify shapes
        x_kin, x_sky, y = next(iter(train_loader))
        print(
            f"    Batch Shapes -> Kinematic: {x_kin.shape}, Sky: {x_sky.shape}, Target: {y.shape}"
        )

        # Verify dimensions match Config
        assert (
            x_kin.shape[1] == Config.WINDOW_SIZE
        ), "Window size mismatch in dataloader"
        assert x_kin.shape[2] == len(
            Config.KINEMATIC_FEATURES
        ), "Kinematic feature dim mismatch"
        assert x_sky.shape[1] == len(Config.SKY_FEATURES), "Sky feature dim mismatch"
        assert y.shape[1] == len(Config.TARGET_COLS), "Target dim mismatch"
        print("    Data shapes verified.")
    else:
        print("    Warning: Train loader is empty. Skipping data shape verification.")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[3] Testing Model Initialization...")

    model = SARTransformer(
        kinematic_input_dim=len(Config.KINEMATIC_FEATURES),
        sky_input_dim=len(Config.SKY_FEATURES),
        output_dim=len(Config.TARGET_COLS),
    )

    # Test forward pass with dummy data if loader failed, or real data if available
    if train_loader and len(train_loader) > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x_kin = x_kin.to(device)
        x_sky = x_sky.to(device)

        with torch.no_grad():
            output = model(x_kin, x_sky)

        print(f"    Forward pass output shape: {output.shape}")
        assert output.shape == (x_kin.shape[0], 2), "Output shape mismatch"
        print("    Forward pass successful.")
    else:
        print("    Skipping forward pass due to lack of data.")

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[4] Testing Trainer...")

    if train_loader and val_loader:
        trainer = Trainer(model, train_loader, val_loader)
        print("    Starting training (1 epoch)...")
        trainer.fit()

        # Verify model checkpoint creation
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"    Checkpoint found at {best_model_path}")
        else:
            raise AssertionError("Model checkpoint not found after training!")
    else:
        print("    Skipping training due to lack of data.")

    # ---------------------------------------------------------
    # 5. Utility Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Utility Functions...")

    # Test coordinate conversion logic
    # Baseline: (0, 0), Target: (1 deg lat away, 1 deg lon away)
    # 1 deg lat ~ 111320m. 1 deg lon at equator ~ 111320m.
    wls_lat = np.array([0.0])
    wls_lon = np.array([0.0])
    gt_lat = np.array([1.0])
    gt_lon = np.array([1.0])

    d_lat, d_lon = wls_to_meters(wls_lat, wls_lon, gt_lat, gt_lon)

    # Check approximate values
    print(f"    1 deg Lat in meters: {d_lat[0]:.2f} (Expected ~111320)")
    print(f"    1 deg Lon in meters: {d_lon[0]:.2f} (Expected ~111320)")

    assert np.isclose(d_lat[0], 111320.0, atol=1000), "Lat conversion error"
    assert np.isclose(d_lon[0], 111320.0, atol=1000), "Lon conversion error"

    # Inverse check
    rec_lat, rec_lon = meters_to_wls(wls_lat, wls_lon, d_lat, d_lon)
    assert np.isclose(rec_lat[0], 1.0), "Inverse Lat conversion error"
    assert np.isclose(rec_lon[0], 1.0), "Inverse Lon conversion error"
    print("    Coordinate conversion utilities verified.")

    # ---------------------------------------------------------
    # 6. Inference Pipeline (Subset)
    # ---------------------------------------------------------
    print("\n[6] Testing Inference Pipeline...")

    # To test inference quickly, we create a subset of the test metadata
    # pointing to a few valid rows, save it, and point Config to it.
    original_test_meta_path = Config.TEST_METADATA_PATH
    subset_test_meta_path = os.path.join(Config.WORKING_DIR, "test_metadata_subset.csv")

    if os.path.exists(original_test_meta_path):
        df_test = pd.read_csv(original_test_meta_path)
        # Take top 50 rows
        df_subset = df_test.head(50)
        df_subset.to_csv(subset_test_meta_path, index=False)

        # Monkeypatch Config
        Config.TEST_METADATA_PATH = subset_test_meta_path
        print(f"    Created test subset with {len(df_subset)} rows.")

        try:
            # We must ensure a model exists. If training ran, it's there.
            # If training didn't run (no data), we save the initialized model.
            if not os.path.exists(os.path.join(Config.WORKING_DIR, "best_model.pth")):
                torch.save(
                    model.state_dict(),
                    os.path.join(Config.WORKING_DIR, "best_model.pth"),
                )
                # Also need scalers if training didn't run
                proc = DataProcessor(mode="train")
                # Create dummy scalers
                dummy_kin = np.random.randn(
                    10, Config.WINDOW_SIZE, len(Config.KINEMATIC_FEATURES)
                )
                dummy_sky = np.random.randn(10, len(Config.SKY_FEATURES))
                # Flatten
                proc.scaler_kin.fit(
                    dummy_kin.reshape(-1, len(Config.KINEMATIC_FEATURES))
                )
                proc.scaler_sky.fit(dummy_sky)
                proc.save_scalers(os.path.join(Config.WORKING_DIR, "scaler.joblib"))

            # Run inference
            # load_cached_data=False forces reprocessing of our new subset
            generate_submission(load_cached_data=False)

            if os.path.exists(Config.SUBMISSION_PATH):
                sub_df = pd.read_csv(Config.SUBMISSION_PATH)
                print(f"    Submission generated with {len(sub_df)} rows.")
                assert len(sub_df) == 50, "Submission row count mismatch"
                print("    Inference pipeline verified.")
            else:
                raise AssertionError("Submission file was not created.")

        except Exception as e:
            print(f"    Inference failed: {e}")
            # Print traceback for debugging
            import traceback

            traceback.print_exc()
        finally:
            # Restore config
            Config.TEST_METADATA_PATH = original_test_meta_path
    else:
        print("    Test metadata not found, skipping inference test.")

    print("\n>>> Demonstration Complete.")


if __name__ == "__main__":
    main()
