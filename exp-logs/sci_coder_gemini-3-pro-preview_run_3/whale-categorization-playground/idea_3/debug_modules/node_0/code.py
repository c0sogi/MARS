import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import CFG
from library.utils import seed_everything, calc_map5
from library.dataset import WhaleDataset, get_transforms
from library.models import WhaleEfficientNet
from library.losses import CurricularFaceLoss
from library.engine import train_fn, eval_fn, predict_and_submit


def main():
    print("Starting Whale Species Identification Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo Speed
    # -------------------------------------------------------------------------
    # We modify the global configuration to run a fast, lightweight demo.
    CFG.debug = True
    CFG.image_size = 224  # Reduced from 448 for speed
    CFG.batch_size = 8  # Small batch size
    CFG.epochs = 1  # Only 1 epoch
    CFG.pretrained = False  # Skip downloading weights for speed/offline safety
    CFG.working_dir = "./working/demo_run"

    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(CFG.seed)
    print("Configuration configured for fast demonstration.")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Micro-Subsets)
    # -------------------------------------------------------------------------
    print("\nPreparing micro-datasets...")

    # Load original metadata
    df_train_full = pd.read_csv(CFG.train_csv)
    df_val_full = pd.read_csv(CFG.val_csv)
    df_test_full = pd.read_csv(CFG.test_csv)

    # Create small subsets (enough for a few batches)
    # We ensure we have enough samples for k-reciprocal ranking (k1=20 default)
    df_train_demo = df_train_full.head(32).copy()
    df_val_demo = df_val_full.head(16).copy()
    df_test_demo = df_test_full.head(16).copy()

    # Save demo metadata to working directory
    train_demo_path = os.path.join(CFG.working_dir, "train_demo.csv")
    val_demo_path = os.path.join(CFG.working_dir, "val_demo.csv")
    test_demo_path = os.path.join(CFG.working_dir, "test_demo.csv")

    df_train_demo.to_csv(train_demo_path, index=False)
    df_val_demo.to_csv(val_demo_path, index=False)
    df_test_demo.to_csv(test_demo_path, index=False)
    print(f"Micro-datasets saved to {CFG.working_dir}")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing Datasets and DataLoaders...")

    # Train Dataset
    train_dataset = WhaleDataset(
        csv_file=train_demo_path,
        mode="train",
        transform=get_transforms("train"),
        id_map=None,  # Let the dataset build the ID map from the data
    )

    # Retrieve the generated ID map
    id_map = train_dataset.get_id_map()
    num_classes = len(id_map)
    print(f"Derived {num_classes} unique classes from the training subset.")

    # Validation Dataset (Must use the same id_map)
    val_dataset = WhaleDataset(
        csv_file=val_demo_path,
        mode="val",
        transform=get_transforms("val"),
        id_map=id_map,
    )

    # Test Dataset
    test_dataset = WhaleDataset(
        csv_file=test_demo_path, mode="test", transform=get_transforms("test")
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # VERIFICATION: Check batch shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}")

    if images.shape != (CFG.batch_size, 3, CFG.image_size, CFG.image_size):
        raise AssertionError("Incorrect image batch shape.")
    if labels.shape != (CFG.batch_size,):
        raise AssertionError("Incorrect label batch shape.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    device = CFG.device
    model = WhaleEfficientNet(num_classes=num_classes)
    model.to(device)

    # VERIFICATION: Forward pass in Train mode (should return Logits)
    model.train()
    logits = model(images.to(device))
    if logits.shape != (CFG.batch_size, num_classes):
        raise AssertionError(
            f"Model train output shape mismatch. Expected {(CFG.batch_size, num_classes)}, got {logits.shape}"
        )

    # VERIFICATION: Forward pass in Eval mode (should return Embeddings)
    model.eval()
    embeddings = model(images.to(device))
    if embeddings.shape != (CFG.batch_size, CFG.embedding_size):
        raise AssertionError(
            f"Model eval output shape mismatch. Expected {(CFG.batch_size, CFG.embedding_size)}, got {embeddings.shape}"
        )

    print("Model initialized and verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Training and Evaluation Loop
    # -------------------------------------------------------------------------
    print("\nRunning Training Loop (1 Epoch)...")

    # Loss and Optimizer
    criterion = CurricularFaceLoss(s=30.0, m=0.50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    # Train
    train_loss = train_fn(train_loader, model, criterion, optimizer, device)
    print(f"Training completed. Loss: {train_loss:.4f}")

    if np.isnan(train_loss):
        raise AssertionError("Training loss returned NaN.")

    # Evaluate
    print("Running Evaluation Loop...")
    val_map = eval_fn(val_loader, model, device, id_map)
    print(f"Validation MAP@5: {val_map:.4f}")

    if not (0.0 <= val_map <= 1.0):
        raise AssertionError("Validation MAP score is out of bounds [0, 1].")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\nRunning Inference and Generating Submission...")
    submission_path = os.path.join(CFG.working_dir, "submission.csv")

    # This function uses the train_loader to build a gallery and test_loader for queries
    # It applies k-Reciprocal Re-ranking internally
    predict_and_submit(
        model, train_loader, test_loader, device, id_map, submission_path
    )

    # VERIFICATION: Check submission file
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created with {len(df_sub)} rows.")

    if len(df_sub) != len(df_test_demo):
        raise AssertionError(
            f"Submission row count mismatch. Expected {len(df_test_demo)}, got {len(df_sub)}"
        )

    if list(df_sub.columns) != ["Image", "Id"]:
        raise AssertionError("Submission columns mismatch. Expected ['Image', 'Id']")

    # Check format of Id column (should be space-separated strings)
    sample_id = df_sub.iloc[0]["Id"]
    if not isinstance(sample_id, str) or len(sample_id.split()) > 5:
        raise AssertionError(
            "Submission Id format incorrect. Should be space-separated string with max 5 labels."
        )

    # -------------------------------------------------------------------------
    # 7. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\nVerifying Metric Logic (MAP@5)...")

    # Case 1: Perfect prediction (Rank 1)
    # Ground Truth: "w_1", Preds: ["w_1", "w_2", ...] -> Score 1.0
    score_perfect = calc_map5(["w_1"], [["w_1", "w_2", "w_3", "w_4", "w_5"]])
    if abs(score_perfect - 1.0) > 1e-6:
        raise AssertionError("Metric logic failed for perfect prediction.")

    # Case 2: Correct at Rank 2
    # Ground Truth: "w_1", Preds: ["w_2", "w_1", ...] -> Score 1/2 = 0.5
    score_rank2 = calc_map5(["w_1"], [["w_2", "w_1", "w_3", "w_4", "w_5"]])
    if abs(score_rank2 - 0.5) > 1e-6:
        raise AssertionError("Metric logic failed for Rank 2 prediction.")

    # Case 3: Incorrect
    score_zero = calc_map5(["w_1"], [["w_2", "w_3", "w_4", "w_5", "w_6"]])
    if score_zero != 0.0:
        raise AssertionError("Metric logic failed for incorrect prediction.")

    print("Metric logic verified.")
    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
