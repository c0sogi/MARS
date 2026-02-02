import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.data_utils as data_utils
import library.model as model_lib
import library.feature_extractor as feat_extractor
import library.training as training_lib

if __name__ == "__main__":
    print("=== Starting Library Usage Demo ===")

    # Define a temporary directory for demo outputs
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Configuration & Patching for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Patch constants in the training library to ensure the demo runs quickly
    # The original config has 20 epochs and ensemble size 5
    training_lib.NUM_EPOCHS = 2
    training_lib.ENSEMBLE_SIZE = 1
    training_lib.BATCH_SIZE = 16

    print(f"Device: {config.DEVICE}")
    print(f"Demo Working Directory: {DEMO_DIR}")

    # ---------------------------------------------------------
    # 2. Metadata Handling (Creating Subsets)
    # ---------------------------------------------------------
    print("\n[2] Preparing Metadata Subsets...")
    # Load the pre-generated metadata files
    # We use the paths defined in config.py
    train_meta_full = pd.read_csv(config.TRAIN_META)
    val_meta_full = pd.read_csv(config.VAL_META)
    test_meta_full = pd.read_csv(config.TEST_META)

    # Create mini subsets (e.g., 50 samples) for the demo
    mini_train = train_meta_full.head(50).copy()
    mini_val = val_meta_full.head(20).copy()
    mini_test = test_meta_full.head(20).copy()

    # Save these mini metadata files to the working directory
    mini_train_path = os.path.join(DEMO_DIR, "train_meta.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val_meta.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test_meta.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(
        f"Created mini metadata: {len(mini_train)} train, {len(mini_val)} val, {len(mini_test)} test."
    )

    # ---------------------------------------------------------
    # 3. Category Encoder Verification
    # ---------------------------------------------------------
    print("\n[3] Testing CategoryEncoder...")
    # This loads/fits the encoder based on the FULL training metadata (as defined in config)
    # This ensures consistency even when we use subsets
    encoder = data_utils.get_category_encoder(load_cached_data=True)

    # Verify encoding/decoding logic
    sample_cat_id = mini_train.iloc[0]["category_id"]
    encoded_idx = encoder.transform([sample_cat_id])[0]
    decoded_cat_id = encoder.inverse_transform([encoded_idx])[0]

    print(
        f"Category ID: {sample_cat_id} -> Index: {encoded_idx} -> Decoded: {decoded_cat_id}"
    )

    # Assertions to verify logic
    assert sample_cat_id == decoded_cat_id, "Encoder inverse transform failed"
    assert 0 <= encoded_idx < config.NUM_CLASSES, "Encoded index out of bounds"

    # ---------------------------------------------------------
    # 4. Data Loading & Processing (Raw Images from BSON)
    # ---------------------------------------------------------
    print("\n[4] Testing RawImageDataset & BSON Loading...")
    # Instantiate dataset with the mini metadata
    # The offsets in mini_train still point to the correct locations in the large TRAIN_BSON file
    dataset = data_utils.RawImageDataset(
        metadata_df=mini_train, bson_path=config.TRAIN_BSON, encoder=encoder
    )

    # Fetch one sample to verify BSON reading and image processing
    images, label, _id = dataset[0]

    print(f"Sample 0 - Images Tensor Shape: {images.shape}, Label: {label}, ID: {_id}")

    # Validation
    # Shape should be (N_images, 3, H, W). N_images is variable (1-4).
    assert images.ndim == 4, "Images tensor should be 4D (N, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (RGB)"
    assert (
        images.shape[2] == config.IMG_SIZE and images.shape[3] == config.IMG_SIZE
    ), f"Images should be resized to {config.IMG_SIZE}x{config.IMG_SIZE}"
    assert isinstance(label, (int, np.integer)), "Label should be an integer"

    # ---------------------------------------------------------
    # 5. Feature Extraction (Simulation)
    # ---------------------------------------------------------
    print("\n[5] Running Feature Extraction on Subsets...")

    # Define output paths for the extracted features
    train_feat_path = os.path.join(DEMO_DIR, "train_features.npy")
    train_label_path = os.path.join(DEMO_DIR, "train_labels.npy")
    val_feat_path = os.path.join(DEMO_DIR, "val_features.npy")
    val_label_path = os.path.join(DEMO_DIR, "val_labels.npy")
    test_feat_path = os.path.join(DEMO_DIR, "test_features.npy")
    test_ids_path = os.path.join(DEMO_DIR, "test_ids.npy")

    # Extract Train Features
    feat_extractor.extract_features_to_disk(
        metadata_path=mini_train_path,
        bson_path=config.TRAIN_BSON,
        output_feat_path=train_feat_path,
        output_label_path=train_label_path,
        load_cached_data=False,  # Force extraction for demo
        debug=False,  # We already manually subsetted the metadata
        split_name="demo_train",
    )

    # Extract Validation Features
    feat_extractor.extract_features_to_disk(
        metadata_path=mini_val_path,
        bson_path=config.TRAIN_BSON,
        output_feat_path=val_feat_path,
        output_label_path=val_label_path,
        load_cached_data=False,
        debug=False,
        split_name="demo_val",
    )

    # Extract Test Features
    feat_extractor.extract_features_to_disk(
        metadata_path=mini_test_path,
        bson_path=config.TEST_BSON,
        output_feat_path=test_feat_path,
        output_id_path=test_ids_path,
        load_cached_data=False,
        debug=False,
        split_name="demo_test",
    )

    # Verify outputs
    train_feats = np.load(train_feat_path)
    print(f"Generated Train Features Shape: {train_feats.shape}")
    assert train_feats.shape == (
        50,
        config.EMBEDDING_DIM,
    ), f"Feature shape mismatch. Expected (50, {config.EMBEDDING_DIM}), got {train_feats.shape}"

    # ---------------------------------------------------------
    # 6. Model Training (MLP on Features)
    # ---------------------------------------------------------
    print("\n[6] Training Ensemble (Mini)...")

    # Train the model using the features we just generated
    models = training_lib.train_ensemble(
        train_feat_path=train_feat_path,
        train_label_path=train_label_path,
        val_feat_path=val_feat_path,
        val_label_path=val_label_path,
        ensemble_size=1,  # We patched this to 1 earlier
        output_dir=DEMO_DIR,
    )

    assert len(models) == 1, "Should have trained exactly 1 model"
    trained_model = models[0]

    # Test Forward Pass with dummy input to verify architecture
    dummy_input = torch.randn(2, config.EMBEDDING_DIM).to(config.DEVICE)
    trained_model.eval()
    with torch.no_grad():
        dummy_out = trained_model(dummy_input)

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Model output shape incorrect. Expected (2, {config.NUM_CLASSES})"

    # ---------------------------------------------------------
    # 7. Inference & Submission
    # ---------------------------------------------------------
    print("\n[7] Running Inference on Test Subset...")

    # Load extracted test features and IDs
    test_feats = np.load(test_feat_path)
    test_ids = np.load(test_ids_path)

    # Create Dataset and Loader for inference
    test_dataset = data_utils.FeatureDataset(test_feats)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds = []

    with torch.no_grad():
        for batch_feats in test_loader:
            batch_feats = batch_feats.to(config.DEVICE)
            logits = trained_model(batch_feats)
            # Get the class index with the highest probability
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # Decode predictions (Index -> Category ID)
    predicted_category_ids = encoder.inverse_transform(all_preds)

    # Create submission dataframe
    submission = pd.DataFrame({"_id": test_ids, "category_id": predicted_category_ids})

    print("Sample Predictions:")
    print(submission.head())

    # Save submission
    sub_path = os.path.join(DEMO_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")
