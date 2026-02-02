import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.feature_extractor import process_split, DualBackbone
from library.trainer import train_ensemble
from library.inference import predict_ensemble
from library.dataset import FeatureDataset
from library.model import HierarchicalMLP
from library.utils import HierarchyMap


def main():
    # 1. Setup
    Config.setup()
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Define subset sizes for fast baseline execution
    # 200k samples is sufficient for a strong baseline while fitting in the time limit
    TRAIN_SUBSET_SIZE = 200000
    VAL_SUBSET_SIZE = 50000

    # Paths for temporary subset metadata
    subset_train_meta_path = os.path.join(Config.CACHE_DIR, "train_subset.csv")
    subset_val_meta_path = os.path.join(Config.CACHE_DIR, "val_subset.csv")

    # 2. Prepare Metadata Subsets
    # We always regenerate these to ensure we are using the correct subset size for this run
    print("Preparing metadata subsets...")

    # Train Subset
    df_train = pd.read_csv(Config.TRAIN_META)
    if len(df_train) > TRAIN_SUBSET_SIZE:
        df_train = df_train.sample(n=TRAIN_SUBSET_SIZE, random_state=Config.SEED)
    df_train.to_csv(subset_train_meta_path, index=False)
    print(f"Created train subset with {len(df_train)} samples.")

    # Val Subset
    df_val = pd.read_csv(Config.VAL_META)
    if len(df_val) > VAL_SUBSET_SIZE:
        df_val = df_val.sample(n=VAL_SUBSET_SIZE, random_state=Config.SEED)
    df_val.to_csv(subset_val_meta_path, index=False)
    print(f"Created val subset with {len(df_val)} samples.")

    # 3. Feature Extraction
    # We force regeneration of Train/Val features to match the subset.
    # We only generate Test features if they don't exist (Test is always full).

    print("Initializing Feature Extractor...")
    model = DualBackbone().to(device)
    model.eval()

    # Extract Train Features (Subset)
    print("Extracting Train Features (Subset)...")
    process_split(
        "train",
        Config.TRAIN_BSON,
        subset_train_meta_path,
        Config.TRAIN_FEATURES,
        Config.TRAIN_LABELS,
        model,
        device,
    )

    # Extract Val Features (Subset)
    print("Extracting Val Features (Subset)...")
    process_split(
        "val",
        Config.TRAIN_BSON,
        subset_val_meta_path,
        Config.VAL_FEATURES,
        Config.VAL_LABELS,
        model,
        device,
    )

    # Extract Test Features (Full)
    if not os.path.exists(Config.TEST_FEATURES) or not os.path.exists(Config.TEST_IDS):
        print("Extracting Test Features (Full)...")
        process_split(
            "test",
            Config.TEST_BSON,
            Config.TEST_META,
            Config.TEST_FEATURES,
            Config.TEST_IDS,
            model,
            device,
        )
    else:
        print("Test features already exist. Skipping extraction.")

    # Free up GPU memory from the backbone model
    del model
    torch.cuda.empty_cache()

    # 4. Training
    print("Starting Ensemble Training...")
    # train_ensemble automatically uses Config.TRAIN_FEATURES which we just populated
    train_ensemble()

    # 5. Validation & Failure Analysis
    print("Performing Validation and Failure Analysis...")

    # Load Validation Data
    hierarchy_map = HierarchyMap(load_cached_data=True)
    val_dataset = FeatureDataset(
        feature_path=Config.VAL_FEATURES,
        label_path=Config.VAL_LABELS,
        hierarchy_map=hierarchy_map,
        mode="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Ensemble Models
    models = []
    for i in range(Config.ENSEMBLE_SIZE):
        model_path = os.path.join(Config.MODEL_DIR, f"ensemble_model_{i}.pth")
        if os.path.exists(model_path):
            m = HierarchicalMLP().to(device)
            m.load_state_dict(torch.load(model_path, map_location=device))
            m.eval()
            models.append(m)

    if not models:
        print("Error: No models trained.")
        return

    # Inference on Val
    correct_count = 0
    total_count = 0

    feature_norms = []
    errors = []  # 1 if error, 0 if correct

    with torch.no_grad():
        for features, l1, l2, l3 in val_loader:
            features = features.to(device)
            l3 = l3.to(device)

            # Ensemble Prediction
            ensemble_probs = torch.zeros(
                features.size(0), Config.NUM_CLASSES_L3, device=device
            )
            for m in models:
                _, _, l3_logits = m(features)
                ensemble_probs += torch.softmax(l3_logits, dim=1)
            ensemble_probs /= len(models)

            _, preds = torch.max(ensemble_probs, 1)

            # Accuracy Stats
            batch_correct = (preds == l3).cpu().numpy()
            correct_count += batch_correct.sum()
            total_count += features.size(0)

            # Failure Analysis Stats
            # Error: 1 if wrong, 0 if correct.
            batch_errors = 1.0 - batch_correct.astype(float)
            errors.append(batch_errors)

            # Feature Norms (L2) to check if "outlier" features correlate with error
            norms = torch.norm(features, p=2, dim=1).cpu().numpy()
            feature_norms.append(norms)

    final_acc = correct_count / total_count
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis Calculation
    all_errors = np.concatenate(errors)
    all_norms = np.concatenate(feature_norms)

    # Correlation
    if np.std(all_errors) > 0 and np.std(all_norms) > 0:
        correlation = np.corrcoef(all_errors, all_norms)[0, 1]
        print(
            f"Correlation between Error Magnitude and Feature Norm: {correlation:.4f}"
        )
    else:
        print("Correlation could not be computed (zero variance in error or norms).")

    # 6. Submission
    THRESHOLD = 0.6239621493939094
    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy {final_acc} > {THRESHOLD}. Generating submission..."
        )

        # Free memory before inference
        del models
        del val_loader
        del val_dataset
        torch.cuda.empty_cache()

        predict_ensemble(batch_size=Config.TRAIN_BATCH_SIZE, device=device)
    else:
        print(f"Validation accuracy {final_acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
