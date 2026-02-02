import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Import library modules
from library.config import Config
from library.dataset import HotelDataset, get_transforms, get_class_to_idx
from library.model import HotelRecognitionModel
from library.train import run_training
from library.inference import run_inference
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Hotel ID Recognition Demo ===")

    # 1. Setup Environment and Paths
    # ----------------------------
    seed_everything(42)
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)
    print(f"Working directory: {demo_dir}")

    # 2. Prepare Mini-Datasets (Speed Optimization)
    # ---------------------------------------------
    print("\n[Data Preparation] Creating mini-datasets...")

    # Load original metadata
    full_train_df = pd.read_csv(Config.train_metadata_path)
    full_val_df = pd.read_csv(Config.val_metadata_path)
    full_test_df = pd.read_csv(Config.test_metadata_path)

    # Create subsets
    # We select a small number of samples.
    # To ensure validation works without missing class errors, we select samples from frequent classes
    # that are guaranteed to exist in both training and validation sets.
    top_classes = full_train_df["hotel_id"].value_counts().head(20).index.tolist()

    mini_train_df = (
        full_train_df[full_train_df["hotel_id"].isin(top_classes)].head(32).copy()
    )

    # Filter validation set to only include classes present in the mini training set
    train_classes = mini_train_df["hotel_id"].unique()
    mini_val_df = (
        full_val_df[full_val_df["hotel_id"].isin(train_classes)].head(16).copy()
    )

    mini_test_df = full_test_df.head(16).copy()

    # Save mini metadata
    mini_train_path = os.path.join(demo_dir, "train_meta.csv")
    mini_val_path = os.path.join(demo_dir, "val_meta.csv")
    mini_test_path = os.path.join(demo_dir, "test_meta.csv")

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)
    mini_test_df.to_csv(mini_test_path, index=False)

    print(f"  Mini Train: {len(mini_train_df)} samples")
    print(f"  Mini Val:   {len(mini_val_df)} samples")
    print(f"  Mini Test:  {len(mini_test_df)} samples")

    # 3. Configure Config Overrides
    # -----------------------------
    # We modify the Config class attributes directly to affect all modules
    print("\n[Configuration] Patching Config for demo...")

    Config.working_dir = demo_dir
    Config.submission_dir = demo_dir

    # Point to mini datasets
    Config.train_metadata_path = mini_train_path
    Config.val_metadata_path = mini_val_path
    Config.test_metadata_path = mini_test_path

    # Output paths
    Config.model_save_path = os.path.join(demo_dir, "model.pth")
    Config.gallery_embeddings_path = os.path.join(demo_dir, "gallery.parquet")
    Config.query_embeddings_path = os.path.join(demo_dir, "query.parquet")
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    # Training Hyperparameters for Speed
    Config.epochs = 1
    Config.batch_size = 8
    Config.num_workers = 0  # Disable multiprocessing for simple script execution
    Config.print_freq = 1

    # Model Hyperparameters
    # Use ResNet18 for speed and disable pretraining to avoid downloads
    Config.backbone_name = "resnet18"
    Config.pretrained = False
    Config.embedding_size = 128

    # Determine number of classes in mini-set to avoid dimension mismatch
    # In a real scenario, this would be the full set count, but for demo we match the data.
    unique_hotels = sorted(mini_train_df["hotel_id"].unique())
    Config.num_classes = len(unique_hotels)
    print(f"  Num Classes set to: {Config.num_classes}")

    # 4. Demonstrate Dataset Loading
    # ------------------------------
    print("\n[Component Test] HotelDataset...")
    class_to_idx = get_class_to_idx(mini_train_df)

    ds = HotelDataset(
        df=mini_train_df,
        transform=get_transforms(mode="train"),
        mode="train",
        class_to_idx=class_to_idx,
    )

    # Fetch one item
    img, label = ds[0]

    # Validation
    assert isinstance(img, torch.Tensor), "Image should be a torch tensor"
    assert img.shape == (
        3,
        Config.crop_size,
        Config.crop_size,
    ), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch tensor"
    print("  Dataset item retrieval successful.")

    # 5. Demonstrate Model Architecture
    # ---------------------------------
    print("\n[Component Test] HotelRecognitionModel...")
    model = HotelRecognitionModel(
        n_classes=Config.num_classes,
        backbone_name=Config.backbone_name,
        pretrained=Config.pretrained,
        embedding_size=Config.embedding_size,
    )
    model.to(Config.device)
    model.eval()

    # Create dummy batch
    dummy_imgs = torch.randn(2, 3, Config.crop_size, Config.crop_size).to(Config.device)
    dummy_labels = torch.tensor([0, 1]).to(Config.device)

    # Test Inference Forward Pass (Embeddings)
    with torch.no_grad():
        embs = model(dummy_imgs)
        assert embs.shape == (2, Config.embedding_size), "Embedding shape mismatch"

    # Test Training Forward Pass (Logits)
    logits = model(dummy_imgs, dummy_labels)
    assert logits.shape == (2, Config.num_classes), "Logits shape mismatch"

    print("  Model forward pass successful.")

    # 6. Run Training Pipeline
    # ------------------------
    print("\n[Pipeline] Running Training Loop...")
    # run_training uses the Config paths we patched
    run_training(debug=False, epochs=Config.epochs)

    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError("Model checkpoint was not saved.")
    print("  Training complete. Checkpoint saved.")

    # 7. Run Inference Pipeline
    # -------------------------
    print("\n[Pipeline] Running Inference Loop...")
    # run_inference uses the Config paths and loads the model we just saved
    run_inference(load_cached_data=False)

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    # Verify submission format
    sub_df = pd.read_csv(Config.submission_path)
    print(f"  Submission generated with {len(sub_df)} rows.")

    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) == len(mini_test_df), "Submission row count mismatch"

    # Check prediction format (space delimited)
    example_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(example_pred, str), "Prediction should be a string"
    assert len(example_pred.split()) > 0, "Prediction string is empty"

    print("  Inference complete. Submission verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
