import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.feature_extract import FeatureExtractor
from library.dataset import FeatureDataset, TestFeatureDataset
from library.model import DualStreamMultiTaskNetwork, Trainer
from library.utils import HierarchyMapper


def main():
    # ==========================================
    # 1. CONFIGURATION OVERRIDES FOR FAST BASELINE
    # ==========================================
    # Limit data size and epochs to ensure execution within 21 minutes
    Config.DEBUG_SIZE = 50000
    Config.EPOCHS = 3
    Config.BATCH_SIZE = 4096

    print(f"Running with DEBUG_SIZE={Config.DEBUG_SIZE}, EPOCHS={Config.EPOCHS}")

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # ==========================================
    # 2. FEATURE EXTRACTION
    # ==========================================
    print("Step 1: Checking/Extracting Features...")
    # This will skip if files exist, or generate small files if they don't and debug_size is used inside logic
    # Note: If full files exist, they are loaded, and we subset later.
    extractor = FeatureExtractor(debug_size=Config.DEBUG_SIZE)
    extractor.extract_all(load_cached_data=True)

    # ==========================================
    # 3. DATASET PREPARATION
    # ==========================================
    print("Step 2: Loading Datasets...")

    # Load Train
    train_ds = FeatureDataset(
        Config.TRAIN_FEATS_RESNET, Config.TRAIN_FEATS_EFFNET, Config.TRAIN_LABELS
    )
    # Load Val
    val_ds = FeatureDataset(
        Config.VAL_FEATS_RESNET, Config.VAL_FEATS_EFFNET, Config.VAL_LABELS
    )
    # Load Test
    test_ds = TestFeatureDataset(
        Config.TEST_FEATS_RESNET, Config.TEST_FEATS_EFFNET, Config.TEST_IDS
    )

    # Apply Subsetting
    if Config.DEBUG_SIZE:
        print(f"Subsetting datasets to {Config.DEBUG_SIZE} samples.")
        train_indices = list(range(min(len(train_ds), Config.DEBUG_SIZE)))
        train_ds = Subset(train_ds, train_indices)

        val_indices = list(range(min(len(val_ds), Config.DEBUG_SIZE)))
        val_ds = Subset(val_ds, val_indices)

        test_indices = list(range(min(len(test_ds), Config.DEBUG_SIZE)))
        test_ds = Subset(test_ds, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 4. ENSEMBLE TRAINING
    # ==========================================
    print("Step 3: Training Ensemble (3 Models)...")

    models = []
    for i in range(3):
        print(f"\n--- Training Model {i+1}/3 ---")
        # Set distinct seed for each model to ensure diversity
        current_seed = Config.SEED + i
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)

        model = DualStreamMultiTaskNetwork().to(device)
        trainer = Trainer(model, device, train_loader, val_loader)
        trainer.fit()

        models.append(model)

    # ==========================================
    # 5. VALIDATION & METRIC
    # ==========================================
    print("\nStep 4: Validation Inference...")

    # Accumulate probabilities
    val_probs = torch.zeros((len(val_ds), Config.NUM_CLASSES_L3), dtype=torch.float32)
    val_targets = []

    # Collect targets once
    for batch in val_loader:
        val_targets.append(batch["label_l3"])
    val_targets = torch.cat(val_targets).numpy()

    # Inference loop
    with torch.no_grad():
        for model in models:
            model.eval()
            batch_start = 0
            for batch in val_loader:
                r_feat = batch["resnet_feat"].to(device)
                e_feat = batch["effnet_feat"].to(device)

                _, _, logits = model(r_feat, e_feat)
                probs = F.softmax(logits, dim=1).cpu()

                batch_size = probs.size(0)
                val_probs[batch_start : batch_start + batch_size] += probs
                batch_start += batch_size

    # Average probabilities
    val_probs /= len(models)
    val_preds = torch.argmax(val_probs, dim=1).numpy()

    # Calculate Accuracy
    accuracy = (val_preds == val_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 6. FAILURE ANALYSIS
    # ==========================================
    print("\nStep 5: Failure Analysis...")

    # Calculate Error Vector (1 if wrong, 0 if correct)
    errors = (val_preds != val_targets).astype(int)

    # Calculate Input Feature Norms (Proxy for signal magnitude)
    # We need to iterate loader again or just grab from dataset if small enough.
    # Since we used Subset, we can iterate the loader.
    feature_norms = []

    # We need to align with the order of val_probs, which matches val_loader order
    with torch.no_grad():
        for batch in val_loader:
            r_feat = batch["resnet_feat"]  # CPU
            e_feat = batch["effnet_feat"]  # CPU

            # Concatenate features to get full input vector
            # r_feat: (B, 2048), e_feat: (B, 1280)
            full_feats = torch.cat([r_feat, e_feat], dim=1)

            # Calculate L2 norm per sample
            norms = torch.norm(full_feats, p=2, dim=1).numpy()
            feature_norms.extend(norms)

    feature_norms = np.array(feature_norms)

    # Calculate Correlation
    if len(errors) > 1 and np.std(errors) > 0 and np.std(feature_norms) > 0:
        corr, _ = pearsonr(errors, feature_norms)
        print(f"Correlation between Error Magnitude and Input Feature Norm: {corr:.6f}")
    else:
        print("Correlation could not be calculated (constant values).")

    # ==========================================
    # 7. SUBMISSION
    # ==========================================
    threshold = 0.6239621493939094
    if accuracy > threshold:
        print(f"\nValidation metric {accuracy} > {threshold}. Generating submission...")

        # Test Inference
        test_probs = torch.zeros(
            (len(test_ds), Config.NUM_CLASSES_L3), dtype=torch.float32
        )
        test_ids = []

        # Collect IDs
        for batch in test_loader:
            test_ids.extend(batch["_id"].numpy())
        test_ids = np.array(test_ids)

        with torch.no_grad():
            for model in models:
                model.eval()
                batch_start = 0
                for batch in test_loader:
                    r_feat = batch["resnet_feat"].to(device)
                    e_feat = batch["effnet_feat"].to(device)

                    _, _, logits = model(r_feat, e_feat)
                    probs = F.softmax(logits, dim=1).cpu()

                    batch_size = probs.size(0)
                    test_probs[batch_start : batch_start + batch_size] += probs
                    batch_start += batch_size

        final_preds_idx = torch.argmax(test_probs, dim=1).numpy()

        # Map indices to category_ids
        mapper = HierarchyMapper(load_cached_data=True)
        idx_to_id_map = np.zeros(Config.NUM_CLASSES_L3, dtype=np.int64)
        for idx, cat_id in mapper.l3_idx_to_id.items():
            idx_to_id_map[idx] = cat_id

        final_category_ids = idx_to_id_map[final_preds_idx]

        # Save
        df_sub = pd.DataFrame({"_id": test_ids, "category_id": final_category_ids})
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(f"\nValidation metric {accuracy} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
