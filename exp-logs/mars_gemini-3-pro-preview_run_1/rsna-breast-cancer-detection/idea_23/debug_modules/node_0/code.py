import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, ProbabilisticF1, get_logger
from library.data import get_dataloaders
from library.model import SiameseEfficientNetFPN
from library.engine import train_one_epoch, evaluate, predict_and_submit


def run_demo():
    print("=== Starting Breast Cancer Detection Library Demo ===")

    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config for speed in this demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 16  # Small subset for quick execution
    Config.IS_DEMO = True  # Custom flag if needed, but we use existing logic

    # Initialize directories and seeds
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # 2. Metric Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Probabilistic F1 Metric...")
    pf1_metric = ProbabilisticF1()

    # Case A: Perfect Prediction
    y_pred_perfect = np.array([1.0, 1.0, 0.0, 0.0])
    y_true_perfect = np.array([1.0, 1.0, 0.0, 0.0])
    score_perfect = pf1_metric(y_pred_perfect, y_true_perfect)

    # Case B: Partial Prediction
    # pTP = 0.8 + 0.0 = 0.8
    # pFP = 0.0 + 0.2 = 0.2
    # TP_total = 1
    # pPrec = 0.8 / (0.8 + 0.2) = 0.8
    # pRec = 0.8 / 1 = 0.8
    # pF1 = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8
    y_pred_partial = np.array([0.8, 0.2])
    y_true_partial = np.array([1.0, 0.0])
    score_partial = pf1_metric(y_pred_partial, y_true_partial)

    print(f"    Perfect Score (Expected ~1.0): {score_perfect:.4f}")
    print(f"    Partial Score (Expected ~0.8): {score_partial:.4f}")

    assert np.isclose(score_perfect, 1.0), "Metric failed on perfect predictions"
    assert np.isclose(score_partial, 0.8), "Metric failed on partial predictions"
    print("    Metric verification passed.")

    # 3. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[3] Initializing Data Pipeline (Debug Mode)...")

    # Force reload cache to ensure we use the debug subset
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
        os.makedirs(Config.CACHE_DIR)

    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cache=False
    )

    # Verify Train Loader
    target_img, contra_img, labels = next(iter(train_loader))

    print(f"    Train Batch Shapes:")
    print(f"      Target: {target_img.shape} (Expected: B, 3, 768, 768)")
    print(f"      Contra: {contra_img.shape} (Expected: B, 3, 768, 768)")
    print(f"      Labels: {labels.shape} (Expected: B)")

    assert target_img.shape == (
        Config.BATCH_SIZE,
        3,
        768,
        768,
    ), "Incorrect target image shape"
    assert contra_img.shape == (
        Config.BATCH_SIZE,
        3,
        768,
        768,
    ), "Incorrect contralateral image shape"
    assert labels.shape[0] == Config.BATCH_SIZE, "Incorrect label batch size"

    # Verify Test Loader (returns prediction_id instead of label)
    t_img, c_img, pred_ids = next(iter(test_loader))
    print(f"    Test Batch Prediction IDs: {pred_ids[:2]}...")
    assert len(pred_ids) == Config.BATCH_SIZE, "Incorrect test batch size"

    print("    Data pipeline verification passed.")

    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Initializing Siamese EfficientNet FPN...")

    model = SiameseEfficientNetFPN(
        backbone_name=Config.BACKBONE, pretrained=False
    )  # Pretrained=False for speed/offline
    model = model.to(device)

    # Run forward pass with the batch fetched earlier
    target_img = target_img.to(device)
    contra_img = contra_img.to(device)

    with torch.no_grad():
        logits = model(target_img, contra_img)

    print(f"    Logits Shape: {logits.shape} (Expected: B, 1)")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("    Model forward pass verification passed.")

    # 5. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Setup optimizer and loss
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch
    avg_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=0
    )
    print(f"    Epoch 0 Average Loss: {avg_loss:.6f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # 6. Evaluation Verification
    # ---------------------------------------------------------
    print("\n[6] Running Evaluation...")

    val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)
    print(f"    Validation Loss: {val_loss:.6f}")
    print(f"    Validation pF1:  {val_pf1:.6f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_pf1 <= 1.0, "pF1 score out of range [0, 1]"

    # 7. Inference & Submission Verification
    # ---------------------------------------------------------
    print("\n[7] Running Inference & Generating Submission...")

    predict_and_submit(model, test_loader, device)

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission File Generated at: {Config.SUBMISSION_PATH}")
        print(f"    Rows: {len(df_sub)}")
        print(f"    Columns: {list(df_sub.columns)}")

        assert "prediction_id" in df_sub.columns, "Missing prediction_id column"
        assert "cancer" in df_sub.columns, "Missing cancer column"
        assert len(df_sub) > 0, "Submission file is empty"
        print("    Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
