import os
import sys
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
import library.config
import library.model_training
import library.feature_extraction
import library.vision_transform
from library.utils import seed_everything
from library.feature_extraction import get_tabular_features, TabularFeatureExtractor
from library.vision_transform import get_vision_data, VisionTransformer
from library.data_loader import get_data_loaders, VolcanoDataset
from library.cnn_architecture import VolcanoEfficientNet
from library.model_training import run_cross_validation, prepare_data_for_cv


def main():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. CONFIGURATION OVERRIDES (SPEED OPTIMIZATION)
    # ==========================================
    print("\n[1] Overriding Configuration for Speed...")

    # Patch constants in library.config
    library.config.CNN_EPOCHS = 1
    library.config.LGB_ROUNDS = 10
    library.config.LGB_EARLY_STOPPING_ROUNDS = 5
    library.config.NUM_FOLDS = 2
    library.config.BATCH_SIZE = 4
    library.config.DEBUG = True

    # Patch constants in library.model_training (since they are imported directly there)
    library.model_training.CNN_EPOCHS = 1
    library.model_training.LGB_ROUNDS = 10
    library.model_training.LGB_EARLY_STOPPING_ROUNDS = 5
    library.model_training.NUM_FOLDS = 2
    library.model_training.BATCH_SIZE = 4

    seed_everything(42)
    print("Configuration patched: Epochs=1, Folds=2, Batch=4.")

    # ==========================================
    # 2. DATA LOADING & SUBSETTING
    # ==========================================
    print("\n[2] Loading and Subsetting Metadata...")

    train_meta_path = "./metadata/train.csv"
    val_meta_path = "./metadata/val.csv"
    test_meta_path = "./metadata/test.csv"

    # Load full metadata
    df_train_full = pd.read_csv(train_meta_path)
    df_val_full = pd.read_csv(val_meta_path)
    df_test_full = pd.read_csv(test_meta_path)

    # Create small subsets for demonstration
    # We take 40 for train, 10 for val to ensure we have enough for 2-fold CV
    subset_train = df_train_full.head(40).copy()
    subset_val = df_val_full.head(10).copy()
    subset_test = df_test_full.head(5).copy()

    print(
        f"Subset sizes -> Train: {len(subset_train)}, Val: {len(subset_val)}, Test: {len(subset_test)}"
    )

    # ==========================================
    # 3. TABULAR FEATURE EXTRACTION
    # ==========================================
    print("\n[3] Demonstrating Tabular Feature Extraction...")

    # Instantiate extractor directly to test single file processing
    extractor = TabularFeatureExtractor(cache_dir="./working/demo_cache")

    # Test processing a single segment
    sample_seg_id = subset_train.iloc[0]["segment_id"]
    sample_file_path = subset_train.iloc[0]["file_path"]
    single_feats = extractor.process_segment(sample_seg_id, sample_file_path)

    assert single_feats is not None, "Feature extraction returned None for valid file."
    assert "segment_id" in single_feats, "segment_id missing from extracted features."
    assert "virtual_source_mean" in single_feats, "Virtual source feature missing."
    print(f"Single segment extraction successful. Features count: {len(single_feats)}")

    # Generate features for the subsets using the wrapper
    # We disable loading from cache to force computation for the demo
    print("Generating features for subsets (force re-compute)...")
    train_feats, val_feats, test_feats = get_tabular_features(
        subset_train, subset_val, subset_test, load_cached_data=False
    )

    # Assertions
    assert len(train_feats) == len(subset_train)
    assert len(val_feats) == len(subset_val)
    assert len(test_feats) == len(subset_test)
    assert train_feats.shape[1] > 100, "Expected > 100 tabular features."

    print("Tabular features generated successfully.")

    # ==========================================
    # 4. VISION DATA GENERATION
    # ==========================================
    print("\n[4] Demonstrating Vision Data Generation (Spectrograms)...")

    # Instantiate transformer
    # We use a separate cache dir for demo to avoid conflicts
    vision_transformer = VisionTransformer(cache_dir="./working/demo_cache")

    # Generate spectrograms
    # Returns tuple: (X, y, ids)
    train_vision, val_vision, test_vision = get_vision_data(
        subset_train, subset_val, subset_test, load_cached_data=False
    )

    X_train_vis, y_train_vis, ids_train_vis = train_vision

    # Assertions
    # Shape should be (N, 10, n_mels, time_steps)
    # Based on config: n_mels=128. time_steps depends on hop_length.
    # 60001 samples / 64 hop ~ 938 time steps.
    print(f"Vision Data Shape: {X_train_vis.shape}")

    assert X_train_vis.ndim == 4
    assert X_train_vis.shape[0] == len(subset_train)
    assert X_train_vis.shape[1] == 10  # 10 sensors
    assert X_train_vis.shape[2] == 128  # SPEC_N_MELS
    assert isinstance(X_train_vis, np.ndarray)
    assert isinstance(y_train_vis, np.ndarray)

    print("Vision data generated successfully.")

    # ==========================================
    # 5. DATA LOADER & DATASET
    # ==========================================
    print("\n[5] Demonstrating Data Loaders...")

    # Instantiate Dataset
    dataset = VolcanoDataset(X_train_vis, y_train_vis)

    # Test __getitem__
    img_tensor, target_tensor = dataset[0]
    assert torch.is_tensor(img_tensor)
    assert torch.is_tensor(target_tensor)
    assert img_tensor.shape[0] == 10

    # Instantiate Loaders
    train_loader, val_loader = get_data_loaders(
        X_train_vis,
        y_train_vis,
        val_vision[0],
        val_vision[1],
        batch_size=4,
        num_workers=0,  # Use 0 workers for simple script execution
    )

    # Iterate one batch
    batch_imgs, batch_targets = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {batch_imgs.shape}, Targets: {batch_targets.shape}")

    assert batch_imgs.shape[0] == 4
    assert batch_imgs.shape[1] == 10

    print("Data Loaders verified.")

    # ==========================================
    # 6. CNN ARCHITECTURE CHECK
    # ==========================================
    print("\n[6] Demonstrating VolcanoEfficientNet...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VolcanoEfficientNet(pretrained=False).to(device)  # False to speed up init

    # Forward pass with dummy batch
    dummy_input = torch.randn(2, 10, 128, 256).to(device)  # (B, C, H, W)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1)

    print("CNN Architecture verified.")

    # ==========================================
    # 7. TRAINING PIPELINE (CV + META LEARNER)
    # ==========================================
    print("\n[7] Demonstrating Full Training Pipeline (2-Fold CV)...")

    # Prepare combined data for CV
    full_feats, full_vision_X, full_targets = prepare_data_for_cv(
        subset_train, subset_val, train_feats, val_feats, train_vision, val_vision
    )

    # Run Cross Validation
    # This will train LGBM and CNN for 2 folds (patched to 1 epoch/10 rounds)
    lgb_models, cnn_paths, meta_model, oof_df = run_cross_validation(
        full_feats, full_vision_X, full_targets
    )

    # Verifications
    assert len(lgb_models) == 2
    assert len(cnn_paths) == 2
    assert len(oof_df) == len(full_targets)
    assert "pred_lgb" in oof_df.columns
    assert "pred_cnn" in oof_df.columns

    print("\nMeta-Learner trained.")
    print(f"Meta-Learner Intercept: {meta_model.intercept_:.4f}")
    print(f"Meta-Learner Coefs: {meta_model.coef_}")

    print("Training pipeline executed successfully.")

    # ==========================================
    # 8. INFERENCE DEMONSTRATION
    # ==========================================
    print("\n[8] Demonstrating Inference on Test Subset...")

    # Tabular Inference (using first fold LGBM)
    drop_cols = ["segment_id", "file_path", "time_to_eruption", "virtual_source"]
    test_feats_clean = test_feats.drop(columns=drop_cols, errors="ignore")
    # Ensure columns match training
    # (In a real scenario, we would align columns strictly)
    lgb_test_preds = lgb_models[0].predict(test_feats_clean)

    # Vision Inference (using first fold CNN)
    cnn_model = VolcanoEfficientNet(pretrained=False).to(device)
    # Load weights (assuming saved correctly in run_cross_validation)
    cnn_model.load_state_dict(torch.load(cnn_paths[0], map_location=device))
    cnn_model.eval()

    test_loader = torch.utils.data.DataLoader(
        VolcanoDataset(test_vision[0], test_vision[1]), batch_size=4, shuffle=False
    )

    cnn_test_preds_log = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            out = cnn_model(imgs)
            cnn_test_preds_log.append(out.cpu().numpy())

    cnn_test_preds = np.expm1(np.concatenate(cnn_test_preds_log)).flatten()

    # Meta Inference
    X_meta_test = np.column_stack([lgb_test_preds, cnn_test_preds])
    final_preds = meta_model.predict(X_meta_test)

    print(f"Generated {len(final_preds)} predictions for test subset.")
    print(f"Sample predictions: {final_preds}")

    assert len(final_preds) == len(subset_test)

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
