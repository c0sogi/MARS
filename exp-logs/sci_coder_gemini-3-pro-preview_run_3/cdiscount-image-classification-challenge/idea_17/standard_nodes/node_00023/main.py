import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_hierarchy_mappings
from library.feature_extraction import FeatureExtractor
from library.dataset import CachedFeatureDataset
from library.model import PDFCNet
from library.training import train_model


def run():
    # 1. Configuration for Fast Baseline
    # We enforce a debug configuration to ensure execution finishes within the time limit.
    Config.DEBUG = False
    # Config.DEBUG_SIZE = 20000
    # Config.EPOCHS = 2
    # Config.BATCH_SIZE = 512
    Config.NUM_MODELS = 3
    Config.WORKING_DIR = "./working/idea_17"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Update artifact paths to point to the current working directory
    Config.TRAIN_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "train_features.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.npy")
    Config.VAL_LABELS_PATH = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.TEST_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "test_features.npy")
    Config.TEST_IDS_PATH = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # 2. Feature Extraction
    print("Step 1: Feature Extraction")
    # This will check for existing files and only extract if necessary/missing.
    # If extracting, it respects Config.DEBUG_SIZE.
    extractor = FeatureExtractor()
    extractor.extract_features(load_cached_data=True)

    # 3. Train Ensemble
    print("\nStep 2: Training Ensemble")
    model_paths = []
    for i in range(Config.NUM_MODELS):
        print(f"\n--- Training Model {i+1}/{Config.NUM_MODELS} ---")
        # Set unique save path and seed for each model in the ensemble
        Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")
        Config.SEED = 42 + i

        # Train the model (train_model uses Config internally)
        train_model()
        model_paths.append(Config.MODEL_SAVE_PATH)

    # 4. Validation (Ensemble)
    print("\nStep 3: Ensemble Validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Validation Data
    val_dataset = CachedFeatureDataset(
        feature_path=Config.VAL_FEATURES_PATH,
        label_path=Config.VAL_LABELS_PATH,
        is_test=False,
    )

    # Apply subsetting if in debug mode to match training distribution size
    if Config.DEBUG:
        indices = list(range(min(len(val_dataset), Config.DEBUG_SIZE)))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load all models in the ensemble
    models = []
    for path in model_paths:
        m = PDFCNet().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    # Inference Loop
    all_preds = []
    all_targets = []
    all_features_norm = []

    with torch.no_grad():
        for features, l1, l2, l3 in val_loader:
            features = features.to(device)
            l3 = l3.to(device)

            all_targets.append(l3.cpu().numpy())

            # Compute L2 norm for failure analysis
            norms = torch.norm(features, p=2, dim=1)
            all_features_norm.append(norms.cpu().numpy())

            # Ensemble Prediction (Average Softmax)
            avg_probs = None
            for model in models:
                _, _, logits3 = model(features)
                probs = torch.softmax(logits3, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(models)
            preds = torch.argmax(avg_probs, dim=1)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_features_norm = np.concatenate(all_features_norm)

    # Compute and Print Metric
    # Metric is Categorization Accuracy (0.0 to 1.0)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Final Validation Metric: {acc}")

    # Failure Analysis
    print("\nStep 4: Failure Analysis")
    # Error vector: 1 if incorrect, 0 if correct
    errors = (all_preds != all_targets).astype(int)

    if len(np.unique(errors)) > 1:
        corr = np.corrcoef(all_features_norm, errors)[0, 1]
        print(f"Correlation between Error Magnitude and Input Feature Norm: {corr:.4f}")
    else:
        print("Correlation cannot be computed (single class in errors).")

    # 5. Submission
    print(f"\nStep 5: Generating Submission (Metric {acc:.4f})")

    # Load Test Data
    test_dataset = CachedFeatureDataset(
        feature_path=Config.TEST_FEATURES_PATH,
        id_path=Config.TEST_IDS_PATH,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_ids = []
    test_preds = []

    print(f"Processing {len(test_dataset)} test samples...")

    with torch.no_grad():
        for features, product_ids in test_loader:
            features = features.to(device)

            # Ensemble Prediction
            avg_probs = None
            for model in models:
                _, _, logits3 = model(features)
                probs = torch.softmax(logits3, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(models)
            preds = torch.argmax(avg_probs, dim=1)

            test_preds.append(preds.cpu().numpy())
            test_ids.append(product_ids.numpy())

    test_preds = np.concatenate(test_preds)
    test_ids = np.concatenate(test_ids)

    # Map indices back to category_ids
    _, idx_to_cat = get_hierarchy_mappings(load_cached_data=True)
    final_cat_ids = [idx_to_cat[p] for p in test_preds]

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"_id": test_ids, "category_id": final_cat_ids})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} rows.")


if __name__ == "__main__":
    run()
