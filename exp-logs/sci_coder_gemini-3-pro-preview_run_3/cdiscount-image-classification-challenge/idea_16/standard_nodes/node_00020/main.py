import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import seed_everything, setup_logging, HierarchyMapper
from library.data import get_training_loader, get_test_loader, CachedTensorDataset
from library.models import DeepFeatureCascade
from library.engine import train_model, evaluate
from library.feature_engineering import extract_features_for_split


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    setup_logging()

    # Adjust Config for Fast Baseline Execution
    Config.NUM_EPOCHS = 5
    Config.BATCH_SIZE_TRAIN = 4096

    # Override Submission Path to match Task Requirement
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Define Subsets for fast training/validation
    # We use a large enough subset to get a good signal, but small enough to be fast.
    TRAIN_SUBSET_SIZE = 100000
    VAL_SUBSET_SIZE = 20000

    print("Initializing Hierarchy Mapper...")
    mapper = HierarchyMapper(Config.CATEGORY_NAMES_PATH, Config.HIERARCHY_MAPPING_PATH)
    mapper.process_hierarchy(load_cached_data=True)

    # 2. Feature Engineering (Train/Val)
    # Check if files exist, otherwise extract
    if not (
        os.path.exists(Config.TRAIN_FEATURES_PATH)
        and os.path.exists(Config.TRAIN_LABELS_PATH)
    ):
        extract_features_for_split(
            split_name="Train",
            meta_path=Config.TRAIN_META_PATH,
            bson_path=Config.TRAIN_BSON_PATH,
            feat_save_path=Config.TRAIN_FEATURES_PATH,
            target_save_path=Config.TRAIN_LABELS_PATH,
            is_test=False,
            subset_size=TRAIN_SUBSET_SIZE,
        )
    else:
        print("Train features already exist.")

    if not (
        os.path.exists(Config.VAL_FEATURES_PATH)
        and os.path.exists(Config.VAL_LABELS_PATH)
    ):
        extract_features_for_split(
            split_name="Validation",
            meta_path=Config.VAL_META_PATH,
            bson_path=Config.TRAIN_BSON_PATH,
            feat_save_path=Config.VAL_FEATURES_PATH,
            target_save_path=Config.VAL_LABELS_PATH,
            is_test=False,
            subset_size=VAL_SUBSET_SIZE,
        )
    else:
        print("Validation features already exist.")

    # 3. Data Loading
    print("Loading DataLoaders...")
    train_loader = get_training_loader(
        Config.TRAIN_FEATURES_PATH,
        Config.TRAIN_LABELS_PATH,
        mapper,
        shuffle=True,
        subset_size=TRAIN_SUBSET_SIZE,
    )

    val_loader = get_training_loader(
        Config.VAL_FEATURES_PATH,
        Config.VAL_LABELS_PATH,
        mapper,
        shuffle=False,
        subset_size=VAL_SUBSET_SIZE,
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = DeepFeatureCascade()

    # 5. Training
    print("Starting Training...")
    model = train_model(
        model, train_loader, val_loader, model_name="baseline_model.pth"
    )

    # 6. Final Validation
    print("Performing Final Validation...")
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, Config.DEVICE)

    print(f"Final Validation Metric: {val_acc}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []
    all_features_norm = []

    # We iterate manually to get features for correlation analysis
    with torch.no_grad():
        for features, l1, l2, l3 in val_loader:
            features = features.to(Config.DEVICE)
            l3 = l3.to(Config.DEVICE)

            # Forward pass
            out_l1, out_l2, out_l3 = model(features)

            # Get predictions
            _, predicted_labels = torch.max(out_l3, 1)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_targets.extend(l3.cpu().numpy())

            # Calculate feature norm (signal strength) per sample
            norms = torch.norm(features, p=2, dim=1).cpu().numpy()
            all_features_norm.extend(norms)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_features_norm = np.array(all_features_norm)

    # Error: 1 if incorrect, 0 if correct
    errors = (all_preds != all_targets).astype(float)

    # Correlation
    if len(errors) > 1:
        correlation = np.corrcoef(errors, all_features_norm)[0, 1]
        print(f"Correlation between Error Magnitude and Feature Norm: {correlation}")
    else:
        print("Not enough samples for correlation analysis.")

    # 8. Submission Logic
    THRESHOLD = 0.6239621493939094

    if val_acc > THRESHOLD:
        print(f"Validation metric {val_acc} > {THRESHOLD}. Generating submission...")

        # Extract Test Features (Full)
        # We need the FULL test set for submission.
        if not (
            os.path.exists(Config.TEST_FEATURES_PATH)
            and os.path.exists(Config.TEST_IDS_PATH)
        ):
            extract_features_for_split(
                split_name="Test",
                meta_path=Config.TEST_META_PATH,
                bson_path=Config.TEST_BSON_PATH,
                feat_save_path=Config.TEST_FEATURES_PATH,
                target_save_path=Config.TEST_IDS_PATH,
                is_test=True,
                subset_size=None,  # Must be None to process all test records
            )
        else:
            print("Test features found.")

        # Create Test Loader
        test_loader = get_test_loader(Config.TEST_FEATURES_PATH, Config.TEST_IDS_PATH)

        # Inference
        model.eval()
        predictions = []
        product_ids = []

        print("Running Inference on Test Set...")
        with torch.no_grad():
            for features, p_ids in test_loader:
                features = features.to(Config.DEVICE)

                # Forward
                _, _, out_l3 = model(features)

                # Get predictions
                probs = torch.softmax(out_l3, dim=1)
                _, pred_indices = torch.max(probs, 1)

                predictions.extend(pred_indices.cpu().numpy())
                product_ids.extend(p_ids.numpy())

        # Map indices back to category_ids
        predictions = np.array(predictions)
        predicted_category_ids = mapper.get_category_id_from_label(predictions)

        # Save Submission
        submission_df = pd.DataFrame(
            {"_id": product_ids, "category_id": predicted_category_ids}
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Validation metric {val_acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
