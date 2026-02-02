import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Ensure the local library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_metric, save_submission
from library.data_processing import get_class_mapping, LeafDataset, get_transforms
from library.feature_extraction import FeatureExtractor
from library.stream_pipeline import StreamClassifier
from library.ensemble_optimizer import optimize_weights, apply_weights


def main():
    # 1. Setup and Configuration
    print(">>> Setting up environment...")
    seed_everything(42)

    # Adjust Config for the demo to run fast and without multiprocessing issues
    Config.NUM_WORKERS = 0
    Config.BATCH_SIZE = 4

    # Define temporary demo directory for outputs
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)
    Config.CACHE_DIR = demo_dir  # Redirect cache to demo folder

    # 2. Data Loading
    print("\n>>> Loading Metadata...")
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    df_full = pd.read_csv(Config.TRAIN_METADATA)

    # Select a small subset: 5 samples from different species to ensure LDA has classes to separate
    # We group by species and take the first one, then take the first 5 groups
    df_subset = df_full.groupby("species").head(1).head(5).reset_index(drop=True)
    print(f"Subset created with {len(df_subset)} samples.")

    # Generate class mapping
    print("Generating class mapping...")
    classes, class_to_idx = get_class_mapping(
        Config.TRAIN_METADATA, load_cached_data=False
    )

    # Verify mapping
    assert len(classes) > 0, "Class list is empty"
    assert len(class_to_idx) == len(classes), "Mismatch between class list and dict"
    print(f"Total classes: {len(classes)}")

    # Instantiate Dataset (just to verify logic)
    print("Instantiating LeafDataset...")
    dataset = LeafDataset(
        df_subset,
        transforms=get_transforms(),
        class_to_idx=class_to_idx,
        split_name="demo_train",
        load_cached_data=False,
    )
    sample = dataset[0]

    # Verify Dataset Output
    # Images: (4 views, 3 channels, 224, 224) -> shape check
    assert sample["images"].shape == (
        4,
        3,
        224,
        224,
    ), f"Incorrect image tensor shape: {sample['images'].shape}"
    assert "tabular" in sample, "Tabular data missing from sample"
    assert "label" in sample, "Label missing from sample"
    print("Dataset verification passed.")

    # 3. Feature Extraction
    print("\n>>> Extracting Features (Visual)...")
    extractor = FeatureExtractor(device=Config.DEVICE)

    # We force re-computation to demonstrate the model forward pass
    # This will download/load DINOv2 and ConvNeXt.
    # Note: In a real constrained environment without internet, this relies on cached weights.
    # Given the prompt context, we assume the environment is ready.
    dino_feats, conv_feats, ids = extractor.extract_features(
        df_subset,
        split_name="demo_subset",
        class_to_idx=class_to_idx,
        load_cached_data=False,
    )

    # Verify Feature Shapes
    # DINOv2-large: 1024 dim, ConvNeXt-large: 1536 dim
    assert dino_feats.shape == (
        5,
        1024,
    ), f"Unexpected DINO features shape: {dino_feats.shape}"
    assert conv_feats.shape == (
        5,
        1536,
    ), f"Unexpected ConvNeXt features shape: {conv_feats.shape}"
    assert len(ids) == 5
    print("Feature extraction successful.")

    # 4. Pipeline Training (StreamClassifier)
    print("\n>>> Training Stream Classifiers...")

    # Prepare targets for the subset
    y_subset = df_subset["species"].map(class_to_idx).values

    # A. Visual Stream (using DINO features as example)
    print("Training Visual Stream (PCA + LDA)...")
    visual_pipeline = StreamClassifier(stream_type="visual")
    visual_pipeline.fit(dino_feats, y_subset)

    # Predict
    vis_probs = visual_pipeline.predict_proba(dino_feats)
    print(f"Visual probs shape: {vis_probs.shape}")

    # Verify probabilities
    assert vis_probs.shape[0] == 5
    # Note: LDA in sklearn outputs probs for classes seen in `fit`.
    # Since we selected 5 distinct species, it should output 5 columns.
    assert vis_probs.shape[1] == 5, f"Expected 5 classes, got {vis_probs.shape[1]}"
    assert np.allclose(vis_probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"

    # B. Tabular Stream
    print("Training Tabular Stream (QuantileTransformer + LDA)...")
    # Load tabular data manually for the subset
    # We can use the dataset object we created earlier to get the tabular X
    tab_X = dataset.X

    tabular_pipeline = StreamClassifier(stream_type="tabular")
    tabular_pipeline.fit(tab_X, y_subset)

    tab_probs = tabular_pipeline.predict_proba(tab_X)
    print(f"Tabular probs shape: {tab_probs.shape}")

    # 5. Ensemble Optimization
    print("\n>>> Optimizing Ensemble Weights...")

    # Construct a dictionary of predictions
    # We'll pretend we have 3 streams: DINO (visual), ConvNeXt (simulated), Tabular
    # For the simulation, we'll just reuse vis_probs with slight noise for the second stream
    conv_probs_sim = vis_probs * 0.9 + 0.1 / 5
    conv_probs_sim = conv_probs_sim / conv_probs_sim.sum(axis=1, keepdims=True)

    oof_preds = {"dino": vis_probs, "convnext": conv_probs_sim, "tabular": tab_probs}

    # The optimizer requires y_true to be integers (indices)
    # However, the predictions `vis_probs` only correspond to the 5 classes in the subset.
    # We need to map y_subset (which are indices into the global 99 classes)
    # to indices [0..4] corresponding to the columns of our local probability matrices.
    # Since we sorted the subset by species implicitly (groupby), the columns of LDA
    # are sorted by class label.

    # Let's verify the classes LDA saw
    lda_classes = visual_pipeline.pipeline.classes_  # These are the global indices

    # We need to map the global y_subset to the local index [0, 1, 2, 3, 4]
    # to use log_loss correctly inside the optimizer if we passed the raw probability matrix.
    # HOWEVER, `calculate_metric` uses sklearn log_loss which can take labels.
    # But `optimize_weights` assumes `y_true` matches the columns of `preds` implicitly
    # or that preds are full-sized.
    # Given the complexity of partial fitting, for this demo we will remap y_true to 0..4

    local_label_map = {
        global_idx: local_idx for local_idx, global_idx in enumerate(lda_classes)
    }
    y_local = np.array([local_label_map[y] for y in y_subset])

    weights = optimize_weights(oof_preds, y_local)

    # Verify weights
    total_weight = sum(weights.values())
    assert np.isclose(total_weight, 1.0), f"Weights sum to {total_weight}, expected 1.0"

    # Apply weights
    final_probs = apply_weights(oof_preds, weights)
    assert final_probs.shape == vis_probs.shape
    print("Ensemble optimization and application successful.")

    # 6. Submission Generation
    print("\n>>> Generating Submission...")

    # Calculate final metric on this subset
    loss = calculate_metric(y_local, final_probs)
    print(f"Demo Log Loss: {loss:.4f}")

    # Save submission
    # We need to provide the class names corresponding to the columns.
    # `lda_classes` holds the global indices. We need the names.
    # Invert class_to_idx to get names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    subset_class_names = [idx_to_class[i] for i in lda_classes]

    submission_path = os.path.join(demo_dir, "submission_demo.csv")
    save_submission(ids, subset_class_names, final_probs, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Check file content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created with shape: {df_sub.shape}")
    print(df_sub.head())

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
