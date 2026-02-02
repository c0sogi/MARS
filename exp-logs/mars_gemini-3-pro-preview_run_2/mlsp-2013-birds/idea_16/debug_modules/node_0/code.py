import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library
from library.utils import seed_everything, get_device, calculate_metrics
from library.dataset import get_loader, BirdDataset, load_and_cache_images
from library.models import BirdModel
from library.trainer import train_model
from library.inference import TTADataset, predict_with_model


def main():
    # 1. Setup
    print("1. Setting up environment...")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_resnet18_fold_0.pth")

    # Clean working directory if exists to ensure a fresh run
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 2. Demonstrating Data Loading
    print("\n2. Demonstrating Data Loading...")
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    # Load dataframes
    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            "Metadata CSV files not found. Ensure ./metadata exists."
        )

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # Subset for speed (use a very small subset for demonstration)
    # We ensure we have enough samples for a batch
    df_train_sub = df_train.head(16).copy()
    df_val_sub = df_val.head(8).copy()

    print(f"Train subset size: {len(df_train_sub)}")
    print(f"Val subset size: {len(df_val_sub)}")

    # Create DataLoaders using library function
    # get_loader handles image caching internally. We use a custom cache dir.
    train_loader = get_loader(
        df=df_train_sub,
        model_name="resnet18",
        phase="train",
        batch_size=8,
        num_workers=0,  # 0 for simple debugging/demo to avoid multiprocessing overhead
        load_cached_data=False,  # Force reload/cache creation in our temp dir
        cache_dir=CACHE_DIR,
    )

    val_loader = get_loader(
        df=df_val_sub,
        model_name="resnet18",
        phase="val",
        batch_size=8,
        num_workers=0,
        load_cached_data=True,  # Use the cache created by train_loader
        cache_dir=CACHE_DIR,
    )

    # Validate DataLoader output
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {images.shape}")  # Expected: (8, 3, 224, 448) for resnet18
    print(f"Targets shape: {targets.shape}")  # Expected: (8, 19)

    assert images.shape == (8, 3, 224, 448), f"Unexpected image shape: {images.shape}"
    assert targets.shape == (8, 19), f"Unexpected targets shape: {targets.shape}"
    assert len(ids) == 8, "IDs length mismatch"

    # 3. Demonstrating Model Instantiation
    print("\n3. Demonstrating Model Instantiation...")
    # Initialize BirdModel with ResNet18 backbone
    # We use pretrained=False to ensure it runs offline without download attempts
    model = BirdModel(model_name="resnet18", num_classes=19, pretrained=False)
    model.to(device)

    # Verify forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 448).to(device)
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, 19), f"Model output shape mismatch: {output.shape}"

    # 4. Demonstrating Training Loop
    print("\n4. Demonstrating Training Loop...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train for a minimal number of epochs to verify the loop works
    trained_model, best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        patience=2,
        save_path=MODEL_SAVE_PATH,
    )

    print(f"Training complete. Best AUC: {best_auc}")
    assert os.path.exists(MODEL_SAVE_PATH), "Model file was not saved."
    assert isinstance(best_auc, float), "AUC is not a float."

    # 5. Demonstrating Inference Components
    print("\n5. Demonstrating Inference Components...")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test = pd.read_csv(test_csv_path)
    df_test_sub = df_test.head(5).copy()

    # Load images for test set (using the cache function directly to populate dict)
    image_dict = load_and_cache_images(df_test_sub, CACHE_DIR, load_cached_data=True)

    # Create TTA Dataset (Test Time Augmentation)
    # We demonstrate 0% shift (standard inference)
    test_dataset = TTADataset(
        df=df_test_sub, image_dict=image_dict, height=224, width=448, shift_pct=0.0
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Run prediction
    preds_dict = predict_with_model(trained_model, test_loader, device)

    print(f"Predictions generated for {len(preds_dict)} samples.")
    sample_id = df_test_sub.iloc[0]["rec_id"]
    print(f"Sample ID: {sample_id}, Prediction shape: {preds_dict[sample_id].shape}")

    assert len(preds_dict) == 5, "Did not get predictions for all test samples."
    assert preds_dict[sample_id].shape == (19,), "Prediction vector shape mismatch."
    assert np.all(
        (preds_dict[sample_id] >= 0) & (preds_dict[sample_id] <= 1)
    ), "Probabilities out of range."

    # 6. Demonstrating Metric Calculation
    print("\n6. Demonstrating Metric Calculation...")
    # Create dummy ground truth and predictions
    y_true = np.random.randint(0, 2, size=(10, 19))
    y_pred = np.random.rand(10, 19)

    auc = calculate_metrics(y_true, y_pred)
    print(f"Calculated Dummy AUC: {auc}")
    assert 0.0 <= auc <= 1.0, "AUC score out of valid range."

    # 7. Generating Submission File Format
    print("\n7. Generating Submission File Format...")
    submission_rows = []
    for rec_id, probs in preds_dict.items():
        for i, p in enumerate(probs):
            # The submission format requires Id = rec_id * 100 + species_idx
            submission_rows.append({"Id": rec_id * 100 + i, "Probability": p})

    df_sub = pd.DataFrame(submission_rows)
    sub_path = os.path.join(WORKING_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(df_sub.head())
    assert len(df_sub) == 5 * 19, "Submission dataframe length mismatch."

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
