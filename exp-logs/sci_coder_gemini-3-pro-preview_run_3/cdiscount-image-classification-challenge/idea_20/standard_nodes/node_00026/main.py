import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Import library modules
from library.config import Config
from library.utils import set_seed, HierarchyMapper
from library.feature_extraction import FeatureExtractor
from library.dataset import CachedFeatureDataset
from library.model import DualStreamProjectedNetwork
from library.engine import fit_model, evaluate, predict_test


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths for our subset operations to ensure speed within the 9-minute limit
    subset_dir = Config.WORKING_DIR
    os.makedirs(subset_dir, exist_ok=True)

    train_meta_path = os.path.join(subset_dir, "train_subset.csv")
    val_meta_path = os.path.join(subset_dir, "val_subset.csv")
    test_meta_path = os.path.join(subset_dir, "test_subset.csv")

    train_feat_path = os.path.join(subset_dir, "train_feat.npy")
    train_label_path = os.path.join(subset_dir, "train_label.npy")
    val_feat_path = os.path.join(subset_dir, "val_feat.npy")
    val_label_path = os.path.join(subset_dir, "val_label.npy")
    test_feat_path = os.path.join(subset_dir, "test_feat.npy")
    test_id_path = os.path.join(subset_dir, "test_id.npy")

    # 2. Create Subsets of Metadata
    # We use a small number of samples to ensure the pipeline finishes.
    N_TRAIN = 10000
    N_VAL = 2000
    N_TEST = 2000

    print("Creating metadata subsets...")
    if os.path.exists(Config.TRAIN_META):
        df_train = pd.read_csv(Config.TRAIN_META)
        df_train.head(N_TRAIN).to_csv(train_meta_path, index=False)

        df_val = pd.read_csv(Config.VAL_META)
        df_val.head(N_VAL).to_csv(val_meta_path, index=False)

        df_test = pd.read_csv(Config.TEST_META)
        df_test.head(N_TEST).to_csv(test_meta_path, index=False)
    else:
        print("Error: Metadata files not found.")
        return

    # 3. Feature Extraction
    extractor = FeatureExtractor(device=device)

    print("Extracting features (Train)...")
    extractor.process_dataset(
        metadata_path=train_meta_path,
        bson_path=Config.TRAIN_BSON,
        output_feat_path=train_feat_path,
        output_label_path=train_label_path,
        is_test=False,
        load_cached=False,
    )

    print("Extracting features (Val)...")
    extractor.process_dataset(
        metadata_path=val_meta_path,
        bson_path=Config.TRAIN_BSON,
        output_feat_path=val_feat_path,
        output_label_path=val_label_path,
        is_test=False,
        load_cached=False,
    )

    print("Extracting features (Test)...")
    extractor.process_dataset(
        metadata_path=test_meta_path,
        bson_path=Config.TEST_BSON,
        output_feat_path=test_feat_path,
        output_label_path=test_id_path,
        is_test=True,
        load_cached=False,
    )

    # 4. Data Loading
    print("Loading datasets...")
    # Initialize HierarchyMapper once to ensure cache exists
    _ = HierarchyMapper(load_cached_data=True)

    train_dataset = CachedFeatureDataset(
        train_feat_path, train_label_path, is_train=True, mixup_alpha=Config.MIXUP_ALPHA
    )
    val_dataset = CachedFeatureDataset(val_feat_path, val_label_path, is_train=False)
    test_dataset = CachedFeatureDataset(test_feat_path, test_id_path, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Training
    print("Initializing model...")
    model = DualStreamProjectedNetwork().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    print("Training...")
    # Train for limited epochs due to time constraint
    best_acc = fit_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=3,
        save_path=os.path.join(Config.WORKING_DIR, "model_subset.pth"),
        patience=2,
    )

    # 6. Validation & Failure Analysis
    print("Running final validation...")
    val_loss, val_acc = evaluate(model, val_loader, device)
    # Required output format
    print(f"Final Validation Metric: {val_acc}")

    print("Performing failure analysis...")
    # Correlation between error and feature magnitude
    model.eval()
    errors = []
    feature_norms = []

    with torch.no_grad():
        for batch_data in val_loader:
            features, l1, l2, l3 = batch_data
            features = features.to(device)
            l3 = l3.to(device)

            _, _, out_l3 = model(features)
            preds = torch.argmax(out_l3, dim=1)

            # Error: 1 if incorrect, 0 if correct
            batch_errors = (preds != l3).float().cpu().numpy()

            # Feature Norm: L2 norm of input features
            batch_norms = torch.norm(features, p=2, dim=1).cpu().numpy()

            errors.extend(batch_errors)
            feature_norms.extend(batch_norms)

    errors = np.array(errors)
    feature_norms = np.array(feature_norms)

    if len(errors) > 0 and np.std(errors) > 0 and np.std(feature_norms) > 0:
        correlation = np.corrcoef(errors, feature_norms)[0, 1]
        print(f"Correlation between Error and Feature Norm: {correlation:.4f}")
    else:
        print("Correlation could not be computed (constant error or norms).")

    # 7. Submission
    THRESHOLD = 0.6239621493939094
    if val_acc > THRESHOLD:
        print(f"Validation accuracy {val_acc} > {THRESHOLD}. Generating submission...")

        # Predict on Test Subset
        test_ids, test_preds = predict_test(model, test_loader, device)

        # Convert model indices back to category_ids
        mapper = HierarchyMapper(load_cached_data=True)
        category_ids = mapper.inverse_transform_targets(test_preds)

        submission_df = pd.DataFrame({"_id": test_ids, "category_id": category_ids})

        # Save
        sub_path = Config.SUBMISSION_PATH
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation accuracy {val_acc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
