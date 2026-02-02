import os
import gc
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.feature_extraction import run_feature_extraction
from library.dataset import CachedFeatureDataset
from library.utils import seed_everything, get_hierarchy_mappings, save_submission
from library.training import train_ensemble_member
from library.model import HierarchicalMLP


def main():
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    # Optimize for speed to meet the 2-hour deadline while maintaining the full-data strategy.
    # We reduce the ensemble size and epochs, and increase batch size for the A100.
    Config.ENSEMBLE_SIZE = 3
    Config.EPOCHS = 7
    Config.BATCH_SIZE = 4096

    print("=== Configuration Overrides ===")
    print(f"Ensemble Size: {Config.ENSEMBLE_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    seed_everything(Config.SEED)

    # ==========================================
    # 2. FEATURE EXTRACTION
    # ==========================================
    # Extracts ResNet50 features from the full dataset (7M images) if not cached.
    # This is the most time-consuming step but essential for accuracy.
    print("\n=== Feature Extraction Phase ===")
    run_feature_extraction(load_cached_data=True, debug=False)

    # Cleanup to free VRAM/RAM before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================
    # 3. DATA LOADING
    # ==========================================
    print("\n=== Data Loading Phase ===")
    # Load hierarchy maps for decoding predictions later
    raw_to_l3, l3_to_raw, l3_to_l1, l3_to_l2 = get_hierarchy_mappings(
        load_cached_data=True
    )

    # Load Train Data (RAM intensive, but fast access)
    train_dataset = CachedFeatureDataset(
        features_path=Config.TRAIN_FEATURES_PATH,
        labels_path=Config.TRAIN_LABELS_PATH,
        is_test=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Validation Data
    val_dataset = CachedFeatureDataset(
        features_path=Config.VAL_FEATURES_PATH,
        labels_path=Config.VAL_LABELS_PATH,
        is_test=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 4. ENSEMBLE TRAINING
    # ==========================================
    print("\n=== Training Phase ===")
    trained_models = []

    for member_id in range(Config.ENSEMBLE_SIZE):
        # Train one MLP on the full dataset features
        train_ensemble_member(member_id, train_loader, val_loader)

        # Load the best checkpoint for this member
        model = HierarchicalMLP(
            input_dim=Config.EMBEDDING_DIM,
            num_classes_l1=Config.NUM_CLASSES_L1,
            num_classes_l2=Config.NUM_CLASSES_L2,
            num_classes_l3=Config.NUM_CLASSES_L3,
        )
        ckpt_path = os.path.join(Config.CACHE_DIR, f"mlp_ensemble_{member_id}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()
        trained_models.append(model)

    # ==========================================
    # 5. VALIDATION & FAILURE ANALYSIS
    # ==========================================
    print("\n=== Validation & Analysis Phase ===")

    all_preds_l3 = []
    all_targets_l3 = []
    all_feat_norms = []

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(Config.DEVICE)
            # targets is tuple (y1, y2, y3)
            _, _, y3 = targets

            # Collect targets
            all_targets_l3.append(y3.cpu().numpy())

            # Collect feature statistics (L2 norm) for failure analysis
            norms = torch.norm(features, p=2, dim=1)
            all_feat_norms.append(norms.cpu().numpy())

            # Ensemble Inference
            member_probs = []
            for model in trained_models:
                _, _, logits_l3 = model(features)
                probs = torch.softmax(logits_l3, dim=1)
                member_probs.append(probs)

            # Average probabilities across ensemble
            avg_probs = torch.stack(member_probs).mean(dim=0)
            _, preds = torch.max(avg_probs, dim=1)
            all_preds_l3.append(preds.cpu().numpy())

    # Aggregate results
    y_true = np.concatenate(all_targets_l3)
    y_pred = np.concatenate(all_preds_l3)
    feat_norms = np.concatenate(all_feat_norms)

    # Metric Calculation
    accuracy = np.mean(y_true == y_pred)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    # Correlation between Error (1=Wrong, 0=Right) and Feature Signal Strength (L2 Norm)
    errors = (y_true != y_pred).astype(int)
    corr = np.corrcoef(errors, feat_norms)[0, 1]
    print(f"Correlation between Error and Feature L2 Norm: {corr:.6f}")

    # ==========================================
    # 6. SUBMISSION
    # ==========================================
    threshold = 0.6239621493939094

    if accuracy > threshold:
        print(f"\nScore {accuracy} > {threshold}. Generating submission...")

        # Free memory
        del train_loader, train_dataset, val_loader, val_dataset
        del all_preds_l3, all_targets_l3, all_feat_norms
        gc.collect()

        # Load Test Data
        test_dataset = CachedFeatureDataset(
            features_path=Config.TEST_FEATURES_PATH,
            labels_path=Config.TEST_IDS_PATH,
            is_test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ids_list = []
        test_preds_list = []

        with torch.no_grad():
            for features, ids in test_loader:
                features = features.to(Config.DEVICE)

                # Ensemble Inference
                member_probs = []
                for model in trained_models:
                    _, _, logits_l3 = model(features)
                    probs = torch.softmax(logits_l3, dim=1)
                    member_probs.append(probs)

                avg_probs = torch.stack(member_probs).mean(dim=0)
                _, preds = torch.max(avg_probs, dim=1)

                test_preds_list.append(preds.cpu().numpy())
                test_ids_list.append(ids.numpy())

        final_preds = np.concatenate(test_preds_list)
        final_ids = np.concatenate(test_ids_list)

        save_submission(final_ids, final_preds, l3_to_raw)

    else:
        print(f"\nScore {accuracy} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
