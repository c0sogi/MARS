import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for module resolution
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.data import get_folds, get_loaders, get_test_loader
from library.models import HeterogeneousExpert
from library.train import fit_model
from library.inference import predict_with_tta


def run_demo():
    print("==== Starting Demonstration ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Demo...")

    # Use a separate working directory for the demo to avoid conflicts
    demo_working_dir = "./working/demo_test"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Monkey-patch Config attributes
    Config.working_dir = demo_working_dir
    Config.epochs = 1
    Config.n_folds = 2
    Config.batch_size = 4
    # Use a lightweight model for the demo
    demo_backbone = "resnet18"
    demo_img_size = 224
    Config.models_config = [(demo_backbone, demo_img_size)]

    # Seed everything
    seed_everything(Config.seed)
    print("Configuration updated successfully.")

    # ---------------------------------------------------------
    # 2. Verify Data Pipeline
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Test get_folds
    # We load the raw metadata to pass to get_folds, simulating the flow
    train_meta_path = os.path.join(Config.metadata_dir, "train.csv")
    train_df_raw = pd.read_csv(train_meta_path)

    # Force recalculation by disabling cache loading for the demo
    folds_df = get_folds(train_df_raw, n_folds=Config.n_folds, load_cached_data=False)

    assert "fold" in folds_df.columns, "Folds DataFrame missing 'fold' column"
    assert (
        folds_df["fold"].nunique() == Config.n_folds
    ), f"Expected {Config.n_folds} folds, got {folds_df['fold'].nunique()}"
    print("get_folds: OK")

    # Test calculate_class_weights
    weights = calculate_class_weights(
        folds_df, Config.target_cols, load_cached_data=False
    )
    assert isinstance(weights, torch.Tensor), "Class weights should be a torch.Tensor"
    assert (
        len(weights) == Config.num_classes
    ), f"Expected {Config.num_classes} weights, got {len(weights)}"
    print("calculate_class_weights: OK")

    # Test get_loaders (with debug=True to load small subset)
    train_loader, valid_loader = get_loaders(
        fold=0,
        img_size=demo_img_size,
        batch_size=Config.batch_size,
        debug=True,
        load_cached_data=False,
    )

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Shapes - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        demo_img_size,
        demo_img_size,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.batch_size,), "Incorrect label batch shape"
    print("get_loaders: OK")

    # ---------------------------------------------------------
    # 3. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = HeterogeneousExpert(
        backbone_name=demo_backbone, num_classes=Config.num_classes, pretrained=False
    )
    model.to(Config.device)
    model.eval()

    # Dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, demo_img_size, demo_img_size).to(Config.device)
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.num_classes,
    ), f"Expected output shape (2, {Config.num_classes}), got {output.shape}"
    print("HeterogeneousExpert Forward Pass: OK")

    del model, dummy_input, output
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 4. Verify Training Loop
    # ---------------------------------------------------------
    print("\n[4] Verifying Training Loop (fit_model)...")

    # Run training for 1 epoch on a debug subset
    # This will save the model to Config.working_dir
    best_auc = fit_model(
        backbone_name=demo_backbone,
        img_size=demo_img_size,
        fold=0,
        epochs=Config.epochs,
        patience=1,
        debug=True,
        load_cached_data=False,
    )

    expected_model_path = os.path.join(
        Config.working_dir, f"{demo_backbone}_fold_0.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"Training complete. Model saved to {expected_model_path}")

    # ---------------------------------------------------------
    # 5. Verify Inference
    # ---------------------------------------------------------
    print("\n[5] Verifying Inference (predict_with_tta)...")

    # Run inference using the trained model
    # Note: predict_with_tta iterates over Config.models_config and Config.n_folds.
    # We set n_folds=2, but only trained fold 0.
    # The inference code checks `if not os.path.exists... continue`.
    # So it will process fold 0 and skip fold 1, which is valid behavior.

    predict_with_tta(debug=True)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not generated"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Verify submission format
    assert "image_id" in sub_df.columns, "image_id column missing in submission"
    for col in Config.target_cols:
        assert col in sub_df.columns, f"Target column {col} missing in submission"

    # Verify values are probabilities (approximate check)
    # Since we used a debug model trained on very few samples, predictions might be garbage,
    # but they should be floats.
    assert (
        sub_df[Config.target_cols]
        .dtypes.apply(lambda x: np.issubdtype(x, np.number))
        .all()
    ), "Predictions are not numeric"

    print("Inference verification complete.")
    print("\n==== Demonstration Finished Successfully ====")


if __name__ == "__main__":
    run_demo()
