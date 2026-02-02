import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_dataset_metadata, get_transforms, PathologyDataset
from library.models import get_model
from library.engine import train_one_epoch, evaluate_tta, predict_test_tta
from library.meta_learner import (
    prepare_stacking_data,
    train_meta_learner,
    predict_meta_learner,
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("--- Starting Library Demo ---")

    # 1. Configuration & Setup
    # Override Config parameters for speed in this demonstration
    Config.WORKING_DIR = "./working/demo_execution"
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.META_MODEL_PARAMS["n_estimators"] = 2  # Reduce trees for instant training

    # Initialize environment
    Config.setup()
    seed_everything(42)

    # 2. Data Loading & Subsetting
    print("\n[Data] Loading metadata and creating subsets...")
    # Load metadata (skipping cache to ensure fresh load for demo)
    df_train = load_dataset_metadata("train", load_cached_data=False)
    df_val = load_dataset_metadata("val", load_cached_data=False)
    df_test = load_dataset_metadata("test", load_cached_data=False)

    # Create tiny subsets for rapid execution
    subset_size = 20
    df_train_sub = df_train.head(subset_size).copy()
    df_val_sub = df_val.head(subset_size).copy()
    df_test_sub = df_test.head(subset_size).copy()

    # Initialize Datasets
    train_ds = PathologyDataset(df_train_sub, transforms=get_transforms("train"))
    val_ds = PathologyDataset(df_val_sub, transforms=get_transforms("val"))
    test_ds = PathologyDataset(df_test_sub, transforms=get_transforms("test"))

    # Initialize DataLoaders (num_workers=0 for simple sequential execution)
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(
        f"Subset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}"
    )

    # 3. Model Initialization & Training
    print("\n[Model] Instantiating ConvNeXt Tiny...")
    device = torch.device(Config.DEVICE)
    # Use pretrained=False to avoid downloading weights during this time-constrained demo
    model = get_model("convnext_tiny", pretrained=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    print("[Model] Running 1 epoch of training...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=0)
    print(f"  Train Loss: {train_loss:.4f}")

    # Validation: Loss should be a valid number
    assert not np.isnan(train_loss), "Training loss returned NaN."

    # 4. Evaluation (TTA)
    print("\n[Eval] Running TTA Evaluation on Validation set...")
    val_auc, val_preds = evaluate_tta(model, val_loader, device)
    print(f"  Val AUC (TTA): {val_auc:.4f}")

    # Validation: AUC should be valid
    assert 0.0 <= val_auc <= 1.0, "AUC score is out of bounds."
    assert len(val_preds) == len(df_val_sub), "Mismatch in validation prediction count."

    # 5. Inference
    print("\n[Inference] Generating Test predictions...")
    test_preds = predict_test_tta(model, test_loader, device)
    assert len(test_preds) == len(df_test_sub), "Mismatch in test prediction count."

    # 6. Meta-Learner (Stacking) Pipeline
    print("\n[Stacking] Simulating OOF predictions and training Meta-Learner...")

    # Simulate OOF predictions (usually generated via K-Fold CV)
    # We create a DataFrame matching the structure required by prepare_stacking_data
    oof_df = df_train_sub[["id"]].copy()
    oof_df["pred"] = np.random.rand(len(oof_df))  # Dummy probabilities

    test_pred_df = df_test_sub[["id"]].copy()
    test_pred_df["pred"] = test_preds

    # Dictionaries mapping model names to their prediction DataFrames
    oof_predictions = {"convnext_tiny": oof_df}
    test_predictions = {"convnext_tiny": test_pred_df}

    # Prepare feature matrices
    train_stack, test_stack = prepare_stacking_data(
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        train_labels=df_train_sub,  # Use subset as ground truth
        test_ids=df_test_sub,
        load_cached_data=False,
    )

    # Train XGBoost Meta-Learner
    meta_model, features = train_meta_learner(train_stack)

    # Generate Final Submission
    submission_df = predict_meta_learner(meta_model, test_stack, features)

    # Validation
    assert len(submission_df) == len(df_test_sub), "Submission length mismatch."
    expected_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        expected_path
    ), f"Submission file not found at {expected_path}"

    print(f"\n[Success] Demo completed. Submission saved to {expected_path}")


if __name__ == "__main__":
    main()
