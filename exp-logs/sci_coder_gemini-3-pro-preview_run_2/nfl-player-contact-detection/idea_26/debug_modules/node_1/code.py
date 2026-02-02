import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, shortest_arc_distance, clamp_values
from library.data_processing import get_dataset
from library.model import SEARVN
from library.training import train_model


def create_mini_dataset(
    source_meta_path, source_track_path, source_helm_path, dest_dir, prefix, nrows=500
):
    """
    Creates a consistent mini-dataset by slicing metadata and filtering
    tracking/helmet data to match the sampled game_plays.
    """
    print(f"Creating mini-dataset for {prefix}...")

    # 1. Load a small slice of metadata
    df_meta = pd.read_csv(source_meta_path, nrows=nrows)
    # Ensure we have at least one contact event for training stability if possible,
    # though strictly slicing head is fine for a demo.

    # Save mini metadata
    meta_path = os.path.join(dest_dir, f"{prefix}_meta.csv")
    df_meta.to_csv(meta_path, index=False)

    # 2. Get relevant game_plays
    valid_game_plays = df_meta["game_play"].unique()

    # 3. Filter Tracking Data
    # We read the full file but filter immediately.
    # Note: In a real constrained env, we might read in chunks, but here RAM is 220GB.
    print(f"  Filtering tracking data from {source_track_path}...")
    df_track = pd.read_csv(source_track_path)
    df_track_mini = df_track[df_track["game_play"].isin(valid_game_plays)].copy()
    track_path = os.path.join(dest_dir, f"{prefix}_tracking.csv")
    df_track_mini.to_csv(track_path, index=False)

    # 4. Filter Helmet Data
    print(f"  Filtering helmet data from {source_helm_path}...")
    df_helm = pd.read_csv(source_helm_path)
    df_helm_mini = df_helm[df_helm["game_play"].isin(valid_game_plays)].copy()
    helm_path = os.path.join(dest_dir, f"{prefix}_helmets.csv")
    df_helm_mini.to_csv(helm_path, index=False)

    return meta_path, track_path, helm_path


def run_demo():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    print("=== Step 1: Setup & Configuration ===")
    seed_everything(42)

    # Create a temporary directory for this demo run
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config to point to our demo directory and files
    Config.WORKING_DIR = DEMO_DIR
    Config.EPOCHS = 2  # Run only 2 epochs for speed
    Config.BATCH_SIZE = 32  # Smaller batch size for small dataset

    # =========================================================================
    # 2. Prepare Mini-Datasets
    # =========================================================================
    print("\n=== Step 2: Preparing Mini-Datasets ===")

    # Prepare Train (Split from original train)
    train_meta, train_track, train_helm = create_mini_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        Config.TRAIN_HELMETS_PATH,
        DEMO_DIR,
        "train",
        nrows=1000,
    )

    # Prepare Validation (Split from original validation)
    val_meta, _, _ = create_mini_dataset(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,  # Val uses train tracking source
        Config.TRAIN_HELMETS_PATH,
        DEMO_DIR,
        "val",
        nrows=200,
    )

    # Prepare Test (Split from original test)
    test_meta, test_track, test_helm = create_mini_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_TRACKING_PATH,
        Config.TEST_HELMETS_PATH,
        DEMO_DIR,
        "test",
        nrows=200,
    )

    # Update Config paths to point to these new mini files
    Config.TRAIN_METADATA_PATH = train_meta
    Config.TRAIN_TRACKING_PATH = train_track
    Config.TRAIN_HELMETS_PATH = train_helm

    Config.VAL_METADATA_PATH = val_meta
    # Note: Val uses the same tracking/helmet files as train in the library logic
    # if split="val", so we don't need to override tracking paths for val specifically
    # in Config, but get_dataset("val") uses Config.TRAIN_TRACKING_PATH.
    # Since we updated Config.TRAIN_TRACKING_PATH to the mini version which contains
    # only 'train' game_plays, we must ensure the 'val' game_plays are also in there
    # OR update the logic.
    # The library `get_dataset` for 'val' uses `Config.TRAIN_TRACKING_PATH`.
    # Our mini `train_tracking.csv` ONLY has game_plays from `train_meta`.
    # `val_meta` has different game_plays. This will cause empty merges for val.
    # FIX: We need to append val tracking data to the file pointed to by Config.TRAIN_TRACKING_PATH.

    print(
        "  Merging Val tracking/helmets into Train files for library compatibility..."
    )
    # Load the mini train files we just made
    df_tr_track = pd.read_csv(train_track)
    df_tr_helm = pd.read_csv(train_helm)

    # Load the raw data for val game_plays
    df_val_meta = pd.read_csv(val_meta)
    val_gps = df_val_meta["game_play"].unique()

    df_raw_track = pd.read_csv("./input/train_player_tracking.csv")
    df_raw_helm = pd.read_csv("./input/train_baseline_helmets.csv")

    df_val_track = df_raw_track[df_raw_track["game_play"].isin(val_gps)]
    df_val_helm = df_raw_helm[df_raw_helm["game_play"].isin(val_gps)]

    # Concat and overwrite
    pd.concat([df_tr_track, df_val_track]).to_csv(train_track, index=False)
    pd.concat([df_tr_helm, df_val_helm]).to_csv(train_helm, index=False)

    # Update Test Config
    Config.TEST_METADATA_PATH = test_meta
    Config.TEST_TRACKING_PATH = test_track
    Config.TEST_HELMETS_PATH = test_helm

    # =========================================================================
    # 3. Verify Utility Logic
    # =========================================================================
    print("\n=== Step 3: Verifying Utilities ===")

    # Test shortest_arc_distance
    angle1 = torch.tensor([10.0, 350.0])
    angle2 = torch.tensor([20.0, 10.0])
    dist = shortest_arc_distance(angle1, angle2)
    print(f"  Angles: {angle1} vs {angle2} -> Dist: {dist}")
    assert torch.allclose(
        dist, torch.tensor([10.0, 20.0])
    ), "Arc distance calculation failed!"

    # Test clamp_values
    vals = np.array([-150.0, 0.0, 150.0])
    clamped = clamp_values(vals, -100, 100)
    print(f"  Clamping: {vals} -> {clamped}")
    assert clamped[0] == -100.0 and clamped[2] == 100.0, "Clamping logic failed!"

    # =========================================================================
    # 4. Data Loading & Processing
    # =========================================================================
    print("\n=== Step 4: Data Loading ===")

    # Load Train Dataset
    # load_cached_data=False forces processing of our new mini files
    train_ds = get_dataset("train", load_cached_data=False)
    print(f"  Train Dataset Size: {len(train_ds)}")

    # Load Val Dataset
    val_ds = get_dataset("val", load_cached_data=False)
    print(f"  Val Dataset Size: {len(val_ds)}")

    # Verify Item Structure
    x_kin, x_vis, x_cat, y = train_ds[0]
    print(
        f"  Sample Shapes -> Kin: {x_kin.shape}, Vis: {x_vis.shape}, Cat: {x_cat.shape}, Label: {y.shape}"
    )

    assert x_kin.dim() == 1, "Kinematic features should be 1D per sample"
    assert x_vis.dim() == 1, "Visual features should be 1D per sample"
    assert (
        x_cat.dim() == 1 and x_cat.shape[0] == 4
    ), "Categorical features should be size 4"

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # =========================================================================
    # 5. Model Initialization
    # =========================================================================
    print("\n=== Step 5: Model Initialization ===")

    dims = train_ds.get_feature_dims()
    print(f"  Feature Dimensions: {dims}")

    model = SEARVN(kin_input_dim=dims["kinematic"], vis_input_dim=dims["visual"]).to(
        Config.DEVICE
    )

    # Switch to eval mode for single-sample inference check
    model.eval()

    # Dummy Forward Pass
    dummy_kin = x_kin.unsqueeze(0).to(Config.DEVICE)
    dummy_vis = x_vis.unsqueeze(0).to(Config.DEVICE)
    dummy_cat = x_cat.unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        out = model(dummy_kin, dummy_vis, dummy_cat)

    print(f"  Model Output Shape: {out.shape}")
    assert out.shape == (1, 1), "Model output shape mismatch!"

    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    print("\n=== Step 6: Training Loop ===")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    trained_model, best_thresh = train_model(
        train_loader,
        val_loader,
        model,
        optimizer,
        Config.DEVICE,
        epochs=Config.EPOCHS,
        patience=1,
    )

    print(f"  Best Threshold: {best_thresh}")
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "best_model.pth")
    ), "Model file not saved!"

    # =========================================================================
    # 7. Inference
    # =========================================================================
    print("\n=== Step 7: Inference ===")

    test_ds = get_dataset("test", load_cached_data=False)
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    trained_model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            x_kin, x_vis, x_cat, contact_ids = batch

            x_kin = x_kin.to(Config.DEVICE)
            x_vis = x_vis.to(Config.DEVICE)
            x_cat = x_cat.to(Config.DEVICE)

            logits = trained_model(x_kin, x_vis, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Apply threshold
            preds = (probs >= best_thresh).astype(int)

            predictions.extend(preds)
            ids.extend(contact_ids)

    # Create submission dataframe
    df_sub = pd.DataFrame({"contact_id": ids, "contact": predictions})
    print(f"  Generated predictions for {len(df_sub)} samples.")
    print(df_sub.head())

    # Verify uniqueness
    assert len(df_sub) == len(test_ds), "Output size mismatch"

    print("\n=== Demo Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
