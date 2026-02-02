import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import timm

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_image, rotate_image, clip_probabilities
from library.feature_extraction import FeatureExtractor
from library.data_processing import DataProcessor
from library.model_pipeline import LeafClassifier
from library.engine import Engine


def create_demo_metadata():
    """
    Creates a small, valid subset of the metadata to ensure StratifiedKFold
    works correctly during the demo (needs >1 sample per class per fold).
    """
    print("Creating demo metadata subsets...")
    full_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select top 2 frequent species to ensure we have enough samples
    top_species = full_train["species"].value_counts().head(2).index.tolist()

    # Take 10 samples from these 2 species
    demo_train = (
        full_train[full_train["species"].isin(top_species)].groupby("species").head(5)
    )
    demo_train = demo_train.reset_index(drop=True)

    # Save to working directory
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_train.to_csv(demo_train_path, index=False)

    # Create a dummy test set (just use the same images for demo purposes)
    demo_test = demo_train.drop(columns=["species"])
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")
    demo_test.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_test_path


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> 1. Configuring Environment for Demo")
    seed_everything(42)

    # Override Config for speed and demo purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small enough for speed, large enough for 2-fold
    Config.BATCH_SIZE = 4
    Config.N_FOLDS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Use smaller models for speed (ResNet18 output dim is 512)
    # Original models: DINOv2 (1024) + ConvNeXt (1536) = 2560
    # Demo models: ResNet18 (512) + ResNet18 (512) = 1024
    Config.MODEL_DINOV2 = "resnet18"
    Config.MODEL_CONVNEXT = "resnet18"

    # Set working directories to avoid overwriting real cache
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Prepare data
    demo_train_path, demo_test_path = create_demo_metadata()
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.TEST_METADATA_PATH = demo_test_path
    # Point val to train for simplicity in this demo
    Config.VAL_METADATA_PATH = demo_train_path

    # ==========================================
    # 2. Utils Demonstration
    # ==========================================
    print("\n>>> 2. Demonstrating library.utils")

    # Load a real image path from the csv
    df = pd.read_csv(demo_train_path)
    sample_rel_path = df.iloc[0]["file_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    # Test load_image
    img = load_image(sample_full_path, target_size=(224, 224))
    print(f"Loaded image shape: {img.shape}")
    assert img.shape == (224, 224), "Image loading/resizing failed"

    # Test rotate_image
    rot_img = rotate_image(img, angle=45)
    print(f"Rotated image shape: {rot_img.shape}")
    assert rot_img.shape == (224, 224), "Rotation changed image dimensions"

    # Test clip_probabilities
    probs = np.array([-0.1, 0.5, 1.2])
    clipped = clip_probabilities(probs)
    print(f"Clipped probabilities: {clipped}")
    assert (clipped >= 1e-15).all() and (clipped <= 1 - 1e-15).all(), "Clipping failed"

    # ==========================================
    # 3. Feature Extraction Demonstration
    # ==========================================
    print("\n>>> 3. Demonstrating library.feature_extraction")

    extractor = FeatureExtractor()

    # Define cache paths for the demo
    cache_img = os.path.join(Config.CACHE_DIR, "demo_train_img.npy")
    cache_tab = os.path.join(Config.CACHE_DIR, "demo_train_tab.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, "demo_train_ids.npy")
    cache_lbl = os.path.join(Config.CACHE_DIR, "demo_train_lbl.npy")

    # Run extraction
    img_feats, tab_feats, ids, labels = extractor.extract_dataset_features(
        demo_train_path,
        cache_img,
        cache_tab,
        cache_ids,
        cache_lbl,
        load_cached_data=False,
    )

    print(f"Extracted Image Features: {img_feats.shape}")
    print(f"Extracted Tabular Features: {tab_feats.shape}")

    # Validation
    # ResNet18 (512) + ResNet18 (512) = 1024 feature dim
    # 12 views per image
    N = len(df)
    assert img_feats.shape == (
        N,
        12,
        1024,
    ), f"Expected ({N}, 12, 1024), got {img_feats.shape}"
    assert tab_feats.shape == (N, 192), "Tabular features dimension mismatch"

    # ==========================================
    # 4. Data Processing Demonstration
    # ==========================================
    print("\n>>> 4. Demonstrating library.data_processing")

    processor = DataProcessor()

    # Manually test internal methods to verify logic
    # 1. Compute Centroids (12 views -> 3 centroids)
    centroids = processor._compute_centroids(img_feats)
    print(f"Computed Centroids: {centroids.shape}")
    assert centroids.shape == (N, 3, 1024), "Centroid computation incorrect"

    # 2. Densify (Flatten centroids and replicate metadata)
    d_img, d_tab, d_ids, d_lbl = processor._densify(centroids, tab_feats, ids, labels)
    print(
        f"Densified Data Shapes: Img={d_img.shape}, Tab={d_tab.shape}, IDs={d_ids.shape}"
    )

    assert d_img.shape == (N * 3, 1024), "Densified image features incorrect"
    assert d_tab.shape == (N * 3, 192), "Densified tabular features incorrect"
    assert d_ids.shape[0] == N * 3, "Densified IDs count incorrect"

    # ==========================================
    # 5. Model Pipeline Demonstration
    # ==========================================
    print("\n>>> 5. Demonstrating library.model_pipeline")

    clf = LeafClassifier()

    # Create synthetic data for the pipeline matching the demo dimensions
    # Visual (1024) + Tabular (192) = 1216
    X_synthetic = np.random.rand(20, 1216).astype(np.float32)
    y_synthetic = np.random.choice(["ClassA", "ClassB"], size=20)

    print("Fitting LeafClassifier on synthetic data...")
    clf.fit(X_synthetic, y_synthetic)

    probs = clf.predict_proba(X_synthetic[:5])
    print(f"Prediction Probabilities Shape: {probs.shape}")
    assert probs.shape == (5, 2), "Prediction output shape mismatch"
    assert np.isclose(probs.sum(axis=1), 1.0).all(), "Probabilities do not sum to 1"

    # ==========================================
    # 6. Engine (Full Workflow) Demonstration
    # ==========================================
    print("\n>>> 6. Demonstrating library.engine (Full Workflow)")

    engine = Engine()

    # We need to ensure the DataProcessor inside Engine uses the cache we just generated
    # or knows where to look.
    # The Engine uses processor.get_train_data(), which uses Config.TRAIN_METADATA_PATH
    # and checks Config.WORKING_DIR/densified_train_...
    # Since we manually ran the extraction to custom cache paths above, the Engine
    # might re-run extraction or we can just let it run.
    # Given we set Config.WORKING_DIR and Config.CACHE_DIR, it will handle itself.

    # However, we need to make sure the raw cache map in DataProcessor aligns with what we want.
    # The DataProcessor uses Config.CACHE_TRAIN_IMG_FEATURES etc.
    # Let's update Config cache filenames to match what we might want, or just let Engine run fresh.
    # Letting it run fresh is safer as it uses the internal logic of DataProcessor completely.

    print("Running Engine.run()...")
    engine.run()

    # Verify submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated successfully with shape: {sub_df.shape}")
        print(sub_df.head(2))
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
