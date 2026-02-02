import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.utils import WGS84_to_ECEF, ECEF_to_WGS84, haversine_distance
from library.data_loader import get_dataloaders
from library.model import DSTResNet
from library.train import Trainer
from library.inference import generate_submission as run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_utils():
    print("\n=== 1. Demonstrating Library Utils ===")

    # Test Coordinate Conversion
    # Known point: Googleplex (approx)
    lat, lon, alt = 37.422, -122.084, 10.0
    print(f"Original LLA: {lat}, {lon}, {alt}")

    x, y, z = WGS84_to_ECEF(lat, lon, alt)
    print(f"Converted to ECEF: {x:.2f}, {y:.2f}, {z:.2f}")

    lat_rec, lon_rec, alt_rec = ECEF_to_WGS84(x, y, z)
    print(f"Recovered LLA: {lat_rec:.6f}, {lon_rec:.6f}, {alt_rec:.2f}")

    # Assert correctness
    assert np.isclose(lat, lat_rec, atol=1e-5)
    assert np.isclose(lon, lon_rec, atol=1e-5)
    assert np.isclose(alt, alt_rec, atol=1e-2)
    print("Coordinate conversion verified.")

    # Test Haversine
    # 1 degree of latitude is approximately 111km
    dist = haversine_distance(0, 0, 1, 0)
    print(f"Distance for 1 deg lat at equator: {dist:.2f} meters")
    assert 110000 < dist < 112000
    print("Haversine distance verified.")


def demo_data_loading_and_model():
    print("\n=== 2. Demonstrating Data Loading & Model Initialization ===")

    # Parameters
    batch_size = 64
    window_size = 11

    # Force reprocessing to demonstrate logic (set load_cached_data=False)
    # This will read from ./metadata/*.csv and ./input/*
    # And save cache to ./working/idea_3/
    print("Initializing DataLoaders (this involves processing raw CSVs)...")
    # Note: This might take a minute or two as it processes the dataset
    train_loader, val_loader, test_loader, meta_test = get_dataloaders(
        batch_size=batch_size, window_size=window_size, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get a batch
    X_batch, y_batch = next(iter(train_loader))
    print(
        f"Sample Batch X shape: {X_batch.shape}"
    )  # Should be (Batch, Window, Features)
    print(f"Sample Batch y shape: {y_batch.shape}")  # Should be (Batch, 2)

    # Initialize Model
    # Features based on library.model.process_data:
    # Dynamic: dLat, dLon, dAlt, MeanCn0, MeanUnc, SatCount (6)
    # Static: WlsLat, WlsLon (2)
    model = DSTResNet(
        dynamic_features=6, static_features=2, window_size=window_size, hidden_dim=64
    )

    # Forward pass
    output = model(X_batch)
    print(f"Model Output shape: {output.shape}")
    assert output.shape == (X_batch.size(0), 2)
    print("Model forward pass successful.")

    return train_loader, val_loader, model


def demo_training(model, train_loader, val_loader):
    print("\n=== 3. Demonstrating Training Pipeline ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Setup components
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Use the directory expected by inference.py defaults
    checkpoint_dir = "./working/idea_3"

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=1,
        checkpoint_dir=checkpoint_dir,
    )

    # Train for 1 epoch for demonstration speed
    print("Starting training for 1 epoch...")
    trainer.fit(train_loader, val_loader, epochs=1)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint saved successfully at {checkpoint_path}")
    else:
        raise FileNotFoundError("Checkpoint was not saved!")

    return checkpoint_path


def demo_inference(checkpoint_path):
    print("\n=== 4. Demonstrating Inference ===")

    output_path = "./working/submission_demo.csv"

    # Run inference using the library function
    # We use load_cached_data=True because get_dataloaders already processed and cached the test data in Step 2
    run_inference(
        batch_size=128,
        window_size=11,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        load_cached_data=True,
    )

    if os.path.exists(output_path):
        df_sub = pd.read_csv(output_path)
        print(f"Submission generated at {output_path}")
        print(f"Submission shape: {df_sub.shape}")
        print("Head:")
        print(df_sub.head())

        # Basic validation
        assert "tripId" in df_sub.columns
        assert "UnixTimeMillis" in df_sub.columns
        assert "LatitudeDegrees" in df_sub.columns
        assert "LongitudeDegrees" in df_sub.columns
        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demos
    try:
        demo_utils()
        train_loader, val_loader, model = demo_data_loading_and_model()
        ckpt_path = demo_training(model, train_loader, val_loader)
        demo_inference(ckpt_path)
        print("\n=== All Demonstrations Completed Successfully ===")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise e
