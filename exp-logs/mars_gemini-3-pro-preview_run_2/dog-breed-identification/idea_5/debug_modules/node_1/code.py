import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.dataset as dataset_lib
import library.backbones as backbones_lib
import library.feature_engine as feature_engine_lib
import library.classifier as classifier_lib
import library.ensemble as ensemble_lib


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_dataset_and_loader():
    """Demonstrates dataset creation, transforms, and dataloader iteration."""
    print("\n=== Demo: Dataset and DataLoader ===")

    # 1. Test Class Mapping
    label_map = dataset_lib.get_class_mapping()
    print(f"Number of classes: {len(label_map)}")
    assert len(label_map) == config.NUM_CLASSES, "Class count mismatch"

    # 2. Test Transforms
    transforms_dict = dataset_lib.get_model_transforms(config.MODEL_A_WEIGHTS)
    assert "standard" in transforms_dict
    assert "global" in transforms_dict
    assert "local" in transforms_dict
    print("Transforms dictionary created successfully.")

    # 3. Instantiate Dataset (Debug Mode)
    # Using Train CSV
    ds = dataset_lib.MultiViewDataset(
        csv_path=config.TRAIN_CSV,
        transform_dict=transforms_dict,
        label_map=label_map,
        debug=True,
    )
    print(f"Dataset (Debug) length: {len(ds)}")
    assert len(ds) == config.DEBUG_DATASET_SIZE, "Debug dataset size mismatch"

    # 4. Fetch one item
    views, label, img_id = ds[0]
    print(f"Sample ID: {img_id}, Label: {label}")

    # Verify View Shapes
    # Standard: CenterCrop defined in config
    assert views["standard"].shape == (
        3,
        config.VIEW_STANDARD_CROP,
        config.VIEW_STANDARD_CROP,
    )
    # Global: Resize defined in config
    assert views["global"].shape == (
        3,
        config.VIEW_GLOBAL_SIZE,
        config.VIEW_GLOBAL_SIZE,
    )
    # Local: CenterCrop defined in config
    assert views["local"].shape == (3, config.VIEW_LOCAL_CROP, config.VIEW_LOCAL_CROP)
    print("View shapes verified.")

    # 5. Test DataLoader
    loader = dataset_lib.get_dataloader(
        csv_path=config.TRAIN_CSV,
        model_weights=config.MODEL_A_WEIGHTS,
        batch_size=8,
        shuffle=True,
        debug=True,
    )

    batch_views, batch_labels, batch_ids = next(iter(loader))
    print(f"Batch labels shape: {batch_labels.shape}")
    assert batch_labels.shape[0] == 8
    print("DataLoader iteration successful.")


def demo_backbone():
    """Demonstrates loading a backbone and running a forward pass."""
    print("\n=== Demo: Backbone Loading ===")

    device = config.DEVICE
    model_name = config.MODEL_A_NAME
    weights_name = config.MODEL_A_WEIGHTS

    # Load feature extractor
    model = backbones_lib.load_feature_extractor(
        model_name=model_name, weights_name=weights_name, device=device, freeze=True
    )

    # Create dummy input: Batch size 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Backbone output shape: {output.shape}")
    # ConvNeXt Large has 1536 embedding dimension
    assert output.shape == (
        2,
        config.MODEL_A_EMBED_DIM,
    ), f"Expected shape (2, {config.MODEL_A_EMBED_DIM}), got {output.shape}"
    print("Backbone forward pass successful.")


def demo_feature_extraction():
    """Demonstrates feature extraction using the engine."""
    print("\n=== Demo: Feature Extraction ===")

    # We will extract features for Train set using Model A (ConvNeXt)
    # Setting load_cached_data=False to force execution of the extraction loop

    embeddings, labels, ids = feature_engine_lib.extract_features(
        dataset_key="train",
        model_name=config.MODEL_A_NAME,
        weights_name=config.MODEL_A_WEIGHTS,
        batch_size=32,
        load_cached_data=False,
        debug=True,
    )

    print(f"Extracted Embeddings Shape: {embeddings.shape}")
    print(f"Extracted Labels Shape: {labels.shape}")

    # Expected shape: (N, Total_Dim)
    # Total Dim = Embed_Dim * 3 (Standard + Global + Local)
    expected_dim = config.MODEL_A_TOTAL_DIM
    assert (
        embeddings.shape[1] == expected_dim
    ), f"Expected embedding dim {expected_dim}, got {embeddings.shape[1]}"
    assert len(embeddings) == config.DEBUG_DATASET_SIZE
    print("Feature extraction verified.")

    return embeddings, labels


def demo_classifier_training(X_train, y_train):
    """Demonstrates training the logistic regression classifier."""
    print("\n=== Demo: Classifier Training ===")

    # Split the debug data into train/val for this micro-demo
    split = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    model, val_probs = classifier_lib.train_logreg(
        X_tr, y_tr, X_val, y_val, model_name="Demo_Model"
    )

    print(f"Validation Probs Shape: {val_probs.shape}")
    assert val_probs.shape[0] == len(y_val)
    assert val_probs.shape[1] == config.NUM_CLASSES
    print("Classifier training and prediction verified.")


def demo_full_pipeline():
    """Demonstrates the full end-to-end pipeline managed by classifier_lib."""
    print("\n=== Demo: Full Classifier Pipeline ===")

    # This function orchestrates:
    # 1. Feature Extraction (Train/Val/Test) for Model A & B
    # 2. Training LogReg for A & B
    # 3. Ensemble Optimization
    # 4. Submission Generation

    # We run with debug=True to keep it fast.
    # It will reuse the cache generated in demo_feature_extraction for Model A Train.

    classifier_lib.run_classifier_pipeline(debug=True, load_cached_data=True)

    # Verify outputs
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df.shape}")

    # In debug mode, the dataset class slices the dataframe to DEBUG_DATASET_SIZE (100).
    assert (
        len(df) == config.DEBUG_DATASET_SIZE
    ), f"Expected {config.DEBUG_DATASET_SIZE} rows in submission, got {len(df)}"

    # Columns: 'id' + 120 breeds
    assert (
        df.shape[1] == config.NUM_CLASSES + 1
    ), "Incorrect number of columns in submission."

    print("Full pipeline execution successful.")


if __name__ == "__main__":
    set_seed(42)

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    try:
        # 1. Dataset Components
        demo_dataset_and_loader()

        # 2. Backbone Components
        demo_backbone()

        # 3. Feature Engine Components
        # We extract features manually to show the function and inspect outputs
        feats, lbls = demo_feature_extraction()

        # 4. Classifier Components
        # We train a model manually on the extracted features
        demo_classifier_training(feats, lbls)

        # 5. Full Pipeline
        # We run the high-level orchestrator to verify end-to-end logic
        demo_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        # Re-raise to ensure the script fails explicitly if something is wrong
        raise e
