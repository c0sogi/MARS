import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.utils import seed_everything, get_device, load_metadata
from library.train import run_training
from library.inference import predict_submission
from library.model import BraTSEfficientNet
from library.data_loader import (
    get_dataloaders,
    prepare_data,
    CACHE_DIR,
    NUM_SLICES,
    MODALITIES,
)


def main():
    # ==========================================
    # Configuration
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    PATIENCE = 5
    THRESHOLD = 0.6978181818181817

    # Paths
    WORKING_DIR = "./working/idea_11"
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Setup
    seed_everything(SEED)
    device = get_device()

    print("========================================")
    print(" STARTING PIPELINE")
    print("========================================")

    # ==========================================
    # 1. Training
    # ==========================================
    print("\n[Step 1] Training Model...")
    # run_training handles data loading (and caching) internally
    # It returns the best validation AUC observed during training
    _ = run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        load_cached_data=True,
    )

    # ==========================================
    # 2. Validation & Failure Analysis
    # ==========================================
    print("\n[Step 2] Validation & Failure Analysis...")

    # Load the best model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Calculate input channels dynamically (Cite debug_lesson_6)
    in_channels = len(MODALITIES) * NUM_SLICES

    # Explicitly inject dependency (Cite debug_lesson_7)
    model = BraTSEfficientNet(in_channels=in_channels)
    model.to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get Validation Data
    # We use get_dataloaders to ensure consistent preprocessing
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True)

    # We also need the IDs to correlate with metadata.
    # get_dataloaders doesn't return IDs for val set, so we load them from cache/prepare_data
    # Since load_cached_data=True, this is fast
    _, _, val_ids = prepare_data("val", load_cached_data=True)

    # Run Inference
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Metric
    final_auc = roc_auc_score(all_targets, all_probs)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # 1. Calculate Error
    errors = np.abs(all_targets - all_probs)

    # 2. Load Metadata for features
    val_meta = load_metadata("val")

    # 3. Create Analysis DataFrame
    # Ensure IDs match. val_ids from prepare_data are strings.
    df_analysis = pd.DataFrame({"BraTS21ID": val_ids, "error": errors})

    # 4. Extract Meta-Features (Slice Counts)
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Pre-process metadata for merging
    val_meta["BraTS21ID"] = val_meta["BraTS21ID"].astype(str)
    for mod in modalities:
        col = f"{mod}_paths"
        val_meta[f"{mod}_count"] = val_meta[col].apply(
            lambda x: len(x) if x is not None else 0
        )

    # Merge
    df_merged = df_analysis.merge(val_meta, on="BraTS21ID", how="left")

    # 5. Correlation
    print("\nCorrelation between Error Magnitude and Slice Counts:")
    feature_cols = [f"{mod}_count" for mod in modalities]
    corrs = df_merged[feature_cols].corrwith(df_merged["error"])
    print(corrs)

    # ==========================================
    # 3. Submission
    # ==========================================
    print("\n[Step 3] Submission Check...")

    if final_auc > THRESHOLD:
        print(
            f"Validation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Proceeding to submission."
        )

        # CLEAR TEST CACHE
        # This ensures that if the test set has changed (hidden set), we re-process it.
        # We leave train/val cache intact to speed up get_dataloaders.
        test_cache_X = os.path.join(CACHE_DIR, "X_test.npy")
        test_cache_ids = os.path.join(CACHE_DIR, "ids_test.npy")

        if os.path.exists(test_cache_X):
            print("Clearing test set cache to ensure full dataset processing...")
            os.remove(test_cache_X)
        if os.path.exists(test_cache_ids):
            os.remove(test_cache_ids)

        # Generate Submission
        predict_submission(
            model_path=MODEL_PATH,
            output_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
            load_cached_data=True,  # Will load Train/Val from cache, but regen Test
            device=device,
        )

    else:
        print(
            f"Validation AUC ({final_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
