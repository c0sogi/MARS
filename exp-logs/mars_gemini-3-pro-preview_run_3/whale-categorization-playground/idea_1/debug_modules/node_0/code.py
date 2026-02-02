import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import WhalePairsDataset, WhaleInferenceDataset, get_transforms
from library.model import WhaleEmbeddingNet, ContrastiveLoss
from library.engine import train_model, predict_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("Initializing demonstration...")

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Define temporary paths for this demo execution
    DEMO_WORKING_DIR = "./working/demo"
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    DEMO_MODEL_PATH = os.path.join(DEMO_WORKING_DIR, "demo_model.pth")
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "demo_submission.csv")

    # Hyperparameters for rapid demonstration
    SUBSET_SIZE = 50  # Use only 50 samples
    BATCH_SIZE = 8
    EPOCHS = 1
    EMBEDDING_DIM = 64  # Smaller embedding for demo

    # -------------------------------------------------------------------------
    # 2. Dataset Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Dataset Classes ---")

    # Test WhalePairsDataset (Training Data)
    train_dataset = WhalePairsDataset(
        csv_file=Config.TRAIN_CSV,
        subset_size=SUBSET_SIZE,
        transform=get_transforms(mode="train"),
    )

    print(f"Train Dataset size (subset): {len(train_dataset)}")

    # Fetch a single batch item to verify structure
    img1, img2, target = train_dataset[0]

    # Verify shapes and types
    # Images should be (Channels, Height, Width) -> (3, 224, 224)
    assert img1.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Image 1 shape mismatch: {img1.shape}"
    assert img2.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Image 2 shape mismatch: {img2.shape}"
    # Target should be a scalar float (0.0 or 1.0)
    assert isinstance(target.item(), float), "Target is not a float"
    assert target.item() in [0.0, 1.0], f"Invalid target value: {target.item()}"

    print("WhalePairsDataset verification passed: Shapes and types are correct.")

    # Test WhaleInferenceDataset (Test/Gallery Data)
    test_dataset = WhaleInferenceDataset(
        csv_file=Config.TEST_CSV,
        subset_size=SUBSET_SIZE,
        transform=get_transforms(mode="test"),
    )

    img, name, whale_id = test_dataset[0]

    assert img.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Inference Image shape mismatch: {img.shape}"
    assert isinstance(name, str), "Image name is not a string"
    # whale_id might be empty for test set, but should be a string
    assert isinstance(whale_id, str) or pd.isna(whale_id), "Whale ID type mismatch"

    print("WhaleInferenceDataset verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Architecture ---")

    device = Config.DEVICE
    model = WhaleEmbeddingNet(embedding_dim=EMBEDDING_DIM)
    model.to(device)
    model.eval()

    # Create dummy input batch: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape: (Batch_Size, Embedding_Dim)
    assert output.shape == (
        2,
        EMBEDDING_DIM,
    ), f"Model output shape mismatch. Expected (2, {EMBEDDING_DIM}), got {output.shape}"

    print("WhaleEmbeddingNet forward pass verification passed.")

    # Test Loss Function
    criterion = ContrastiveLoss(margin=1.0)
    # Dummy outputs for pair
    out1 = torch.randn(2, EMBEDDING_DIM).to(device)
    out2 = torch.randn(2, EMBEDDING_DIM).to(device)
    # Dummy targets (one positive, one negative)
    targets = torch.tensor([1.0, 0.0]).to(device)

    loss = criterion(out1, out2, targets)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("ContrastiveLoss verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Engine Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Training Loop ---")

    # We use the provided train_model function.
    # We override parameters to ensure it runs quickly (1 epoch, small subset).
    trained_model = train_model(
        train_csv=Config.TRAIN_CSV,
        val_csv=Config.VAL_CSV,
        output_path=DEMO_MODEL_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=0.001,
        device=device,
        subset_size=SUBSET_SIZE,  # Limit data for speed
        patience=1,
    )

    # Verify model file was created
    if os.path.exists(DEMO_MODEL_PATH):
        print(f"Training complete. Model saved to {DEMO_MODEL_PATH}")
    else:
        raise FileNotFoundError("Model file was not generated by train_model.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Submission Generation ---")

    # Generate submission using the model we just trained.
    # We use subsets for gallery (train) and query (test) to speed up distance calculation.
    predict_submission(
        model_path=DEMO_MODEL_PATH,
        train_csv=Config.TRAIN_CSV,
        test_csv=Config.TEST_CSV,
        submission_path=DEMO_SUBMISSION_PATH,
        batch_size=BATCH_SIZE,
        device=device,
        subset_size=SUBSET_SIZE,  # Limit gallery/query size for speed
        load_cached_data=False,  # Force regeneration of embeddings
    )

    # Verify submission file format
    if not os.path.exists(DEMO_SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)

    # Check dimensions (subset_size rows + header)
    assert (
        len(df_sub) == SUBSET_SIZE
    ), f"Submission rows mismatch. Expected {SUBSET_SIZE}, got {len(df_sub)}"

    # Check columns
    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Submission columns missing."

    # Check prediction format (should be 5 space-separated labels)
    sample_pred = df_sub.iloc[0]["Id"]
    preds = sample_pred.split(" ")
    assert (
        len(preds) == 5
    ), f"Prediction format incorrect. Expected 5 labels, got {len(preds)}: {preds}"

    print(f"Submission generated successfully at {DEMO_SUBMISSION_PATH}")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
