import sys
import os
import shutil
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_factory import get_dataloaders
from library.architectures import WhaleClassifier
from library.trainer import ModelTrainer
from library.inference import Predictor


def main():
    # =========================================================================
    # 1. Configuration Overrides for Speed & Demo
    # =========================================================================
    print(">>> Configuring for Demo execution...")

    # Modify Config attributes at runtime to speed up the demo
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.N_FOLDS = 2  # Use only 2 folds for the demo
    Config.BATCH_SIZE = 8  # Small batch size for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Reduce ensemble to a single model architecture for demonstration
    Config.ENSEMBLE_CONFIGS = [
        {"arch": Config.ARCH_EFFICIENTNET, "objective": "auc", "name": "effnet_b0_auc"},
    ]

    # Clean and recreate working directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading Verification
    # =========================================================================
    print("\n>>> Testing Data Loading...")
    loaders = get_dataloaders(debug=True)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Fetch a single batch to verify shapes
    batch_inputs, batch_targets = next(iter(train_loader))
    print(f"Batch Input Shape: {batch_inputs.shape}")  # Expected: [B, 1, Freq, Time]
    print(f"Batch Target Shape: {batch_targets.shape}")  # Expected: [B]

    # Assertions
    assert batch_inputs.ndim == 4, "Input should be 4D (Batch, Channel, Freq, Time)"
    assert batch_inputs.shape[1] == 1, "Input should have 1 channel (spectrogram)"
    assert batch_targets.ndim == 1, "Targets should be 1D"

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n>>> Testing Model Architecture...")
    # Initialize model (pretrained=False to avoid downloading weights during demo)
    model = WhaleClassifier(Config.ARCH_EFFICIENTNET, pretrained=False)
    model.to(device)

    # Test forward pass
    with torch.no_grad():
        outputs = model(batch_inputs.to(device))

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        batch_inputs.shape[0],
        1,
    ), "Output shape mismatch (should be [Batch, 1])"

    # =========================================================================
    # 4. Training Loop Simulation (Generate Checkpoints)
    # =========================================================================
    print("\n>>> Testing Training Loop & Generating Checkpoints...")

    # We need to train models for all folds defined in Config.N_FOLDS
    # so that the Predictor can find them later.
    config_entry = Config.ENSEMBLE_CONFIGS[0]
    base_name = config_entry["name"]

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # No scheduler for this short demo

    for fold in range(Config.N_FOLDS):
        print(f"--- Simulating Training for Fold {fold} ---")
        save_name = f"{base_name}_fold_{fold}"

        # Re-initialize trainer for each fold
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            objective=config_entry["objective"],
            save_name=save_name,
        )

        # Run training for 1 epoch
        trainer.fit(epochs=Config.EPOCHS, patience=1)

        # Verify checkpoint creation
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{save_name}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint failed to save at {ckpt_path}")

    # =========================================================================
    # 5. Full Inference Pipeline
    # =========================================================================
    print("\n>>> Testing Inference Pipeline (OOF + Meta-Learner + Test)...")

    # Initialize Predictor
    # This class handles:
    # 1. Loading checkpoints
    # 2. Generating Out-Of-Fold (OOF) predictions
    # 3. Training the Meta-Learner (Logistic Regression)
    # 4. Generating Test predictions (Bagging)
    # 5. Creating the submission CSV
    predictor = Predictor(debug=True)

    predictor.create_submission()

    # =========================================================================
    # 6. Final Validation of Submission
    # =========================================================================
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")

    # Validate format
    assert "clip" in df_sub.columns, "Submission missing 'clip' column"
    assert "probability" in df_sub.columns, "Submission missing 'probability' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Validate probability range
    probs = df_sub["probability"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
