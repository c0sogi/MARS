import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_datasets
from library.model import SRPFEModel
from library.engine import train_model, predict


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on CuDNN backend, ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    print("Setting up configuration and seeds...")
    set_seed(Config.SEED)

    # Optimize for speed: Reduce epochs and increase batch size for the demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4096  # A100 can handle large batches
    Config.NUM_WORKERS = 4  # Utilize available vCPUs

    # Setup directories (creates working/idea_36 and submission folders)
    Config.setup()

    # 2. Data Loading
    print("Loading and preparing datasets...")
    # load_cached_data=True will try to use existing cache in working/idea_36 if available
    # otherwise it processes from scratch using the metadata files.
    train_ds, val_ds, test_ds, meta = get_datasets(load_cached_data=True)

    # Validation of loaded data
    print(
        f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}"
    )
    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."
    assert len(test_ds) > 0, "Test dataset is empty."

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing SR-PFE Model...")
    # Retrieve metadata required for model architecture
    vocab_sizes = meta["vocab_sizes"]
    # Get number of continuous features from the dataset sample
    # sample is a dict {'cat': tensor, 'cont': tensor, 'target': tensor}
    sample_input = train_ds[0]
    num_cont_features = sample_input["cont"].shape[0]

    model = SRPFEModel(vocab_sizes=vocab_sizes, num_cont_features=num_cont_features)
    model.to(Config.DEVICE)

    # Verify Model Forward Pass logic
    dummy_cat = sample_input["cat"].unsqueeze(0).to(Config.DEVICE)
    dummy_cont = sample_input["cont"].unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        dummy_out = model(dummy_cat, dummy_cont)

    # Check output shape: (Batch, Num_Streams)
    # The architecture has 5 parallel streams
    expected_shape = (1, Config.NUM_STREAMS)
    assert (
        dummy_out.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {dummy_out.shape}"
    print("Model initialized and verified successfully.")

    # 4. Training
    print("Starting training loop...")
    # train_model returns the best validation AUC achieved
    best_auc = train_model(model, train_loader, val_loader)
    print(f"Training completed. Best Validation AUC: {best_auc:.5f}")

    # 5. Inference
    print("Running inference on test set...")
    # Predict function loads the best model from disk automatically
    predictions = predict(model, test_loader)

    # 6. Submission Generation
    print("Generating submission file...")
    # Load sample submission to ensure correct format and IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Verify alignment
    assert len(predictions) == len(
        sample_sub
    ), f"Prediction count ({len(predictions)}) does not match sample submission ({len(sample_sub)})."

    # Assign predictions
    sample_sub[Config.TARGET_COL] = predictions

    # Save to disk
    sample_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    # 7. Final Output Validation
    df_check = pd.read_csv(Config.SUBMISSION_FILE)
    assert df_check.shape == (100000, 2), "Submission shape is incorrect."
    assert df_check.columns.tolist() == [
        "id",
        "target",
    ], "Submission columns are incorrect."
    # Ensure probabilities are within valid range [0, 1]
    assert (
        df_check["target"].min() >= 0.0 and df_check["target"].max() <= 1.0
    ), "Probabilities out of bounds."

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
