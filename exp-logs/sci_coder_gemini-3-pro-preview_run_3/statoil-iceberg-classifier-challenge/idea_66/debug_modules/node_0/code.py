import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library import utils, model, data_loader, train, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo execution...")

    # Redirect working directories to a demo folder to avoid overwriting main work
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Optimize hyperparameters for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.N_FOLDS = 2  # Run only 2 folds (indices 0 and 1)
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.NUM_DROPOUT_SAMPLES = 2  # Reduce dropout samples in head

    # Set random seed for reproducibility
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loader...")
    # Get loaders for Fold 0. load_cached_data=False forces processing from raw json.
    train_loader, val_loader = data_loader.get_train_val_loaders(
        fold_index=0, load_cached_data=False
    )

    # Fetch one batch to verify shapes
    imgs, angs, lbls = next(iter(train_loader))
    print(
        f"Train Batch - Images: {imgs.shape}, Angles: {angs.shape}, Labels: {lbls.shape}"
    )

    # Assertions
    # Image: (Batch, 3, 75, 75)
    assert imgs.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Unexpected image shape: {imgs.shape}"
    # Angle: (Batch,)
    assert angs.shape == (Config.BATCH_SIZE,), f"Unexpected angle shape: {angs.shape}"
    # Label: (Batch,)
    assert lbls.shape == (Config.BATCH_SIZE,), f"Unexpected label shape: {lbls.shape}"

    print("Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")
    net = model.IAMSI_CNN().to(device)

    # Prepare dummy inputs
    dummy_img = imgs.to(device)
    dummy_ang = angs.to(device)

    # Test Train Mode (Expect output shape: Batch x Num_Samples)
    net.train()
    out_train = net(dummy_img, dummy_ang)
    print(f"Model Output (Train Mode): {out_train.shape}")
    assert out_train.shape == (
        Config.BATCH_SIZE,
        Config.NUM_DROPOUT_SAMPLES,
    ), "Model train output shape mismatch."

    # Test Eval Mode (Expect output shape: Batch x 1, averaged)
    net.eval()
    with torch.no_grad():
        out_eval = net(dummy_img, dummy_ang)
    print(f"Model Output (Eval Mode): {out_eval.shape}")
    assert out_eval.shape == (Config.BATCH_SIZE, 1), "Model eval output shape mismatch."

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Loop (Fold 0)...")
    # This runs training and validation for Config.EPOCHS (1 epoch)
    best_loss_0 = train.run_fold(0)
    print(f"Fold 0 finished. Best Val Loss: {best_loss_0:.4f}")

    # Verify checkpoint creation
    ckpt_0 = os.path.join(Config.WORKING_DIR, "model_best_fold_0.pth")
    assert os.path.exists(ckpt_0), "Checkpoint for Fold 0 was not created."

    print("\n>>> Running Training Loop (Fold 1)...")
    # Run a second fold to demonstrate multi-fold capability for ensembling
    best_loss_1 = train.run_fold(1)
    print(f"Fold 1 finished. Best Val Loss: {best_loss_1:.4f}")

    ckpt_1 = os.path.join(Config.WORKING_DIR, "model_best_fold_1.pth")
    assert os.path.exists(ckpt_1), "Checkpoint for Fold 1 was not created."

    # -------------------------------------------------------------------------
    # 5. Verify Inference and Submission
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference Pipeline...")
    # Generates predictions using the checkpoints from Fold 0 and Fold 1
    predict.generate_predictions(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print("Head of submission:")
    print(df_sub.head())

    # Validate content
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "is_iceberg" in df_sub.columns, "Submission missing 'is_iceberg' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check probability range
    probs = df_sub["is_iceberg"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
