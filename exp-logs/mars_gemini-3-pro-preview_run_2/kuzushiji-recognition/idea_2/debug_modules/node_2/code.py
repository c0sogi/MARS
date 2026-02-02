import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader, Subset

# Import provided library modules
from library.utils import (
    get_label_map,
    parse_ground_truth,
    format_prediction_string,
    calculate_kuzushiji_metrics,
)
from library.dataset import KuzushijiDataset, get_transforms, collate_fn
from library.model import get_kuzushiji_model
from library.engine import train_kuzushiji_model, inference

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def create_compatible_subset(dataset, indices):
    """
    Creates a Subset that retains specific attributes of the parent dataset
    required by the library.engine functions.
    """
    subset = Subset(dataset, indices)
    # Propagate attributes expected by engine.evaluate and engine.inference
    if hasattr(dataset, "int_to_char"):
        subset.int_to_char = dataset.int_to_char
    if hasattr(dataset, "char_to_int"):
        subset.char_to_int = dataset.char_to_int
    if hasattr(dataset, "image_id_map"):
        subset.image_id_map = dataset.image_id_map
    if hasattr(dataset, "df"):
        subset.df = dataset.df
    return subset


def demo_utils():
    print("\n=== Demonstrating library.utils ===")

    # 1. Test get_label_map
    print("1. Testing get_label_map...")
    char_to_int, int_to_char = get_label_map(
        UNICODE_MAP_PATH, cache_dir=os.path.join(WORKING_DIR, "cache")
    )
    assert len(char_to_int) > 0, "Label map should not be empty"
    assert len(int_to_char) == len(char_to_int), "Map lengths should match"
    print(f"   Loaded {len(char_to_int)} characters.")

    # 2. Test parse_ground_truth
    print("2. Testing parse_ground_truth...")
    test_char = list(char_to_int.keys())[0]
    test_id = char_to_int[test_char]
    # Format: Char X Y W H
    label_str = f"{test_char} 10 20 50 60"
    boxes, labels = parse_ground_truth(label_str, char_to_int)

    assert len(boxes) == 1
    assert len(labels) == 1
    # Expected box: [x, y, x+w, y+h] -> [10, 20, 60, 80]
    assert np.allclose(boxes[0], [10, 20, 60, 80]), f"Box mismatch: {boxes[0]}"
    assert labels[0] == test_id
    print("   Ground truth parsing verified.")

    # 3. Test format_prediction_string
    print("3. Testing format_prediction_string...")
    # Center of [10, 20, 60, 80] is (35, 50)
    pred_str = format_prediction_string(
        [np.array([10, 20, 60, 80])], [test_id], int_to_char
    )
    expected_str = f"{test_char} 35 50"
    assert pred_str == expected_str, f"Expected '{expected_str}', got '{pred_str}'"
    print("   Prediction formatting verified.")

    # 4. Test calculate_kuzushiji_metrics
    print("4. Testing calculate_kuzushiji_metrics...")
    # Simulate 1 True Positive
    gt_df = pd.DataFrame({"image_id": ["img1"], "labels": [f"{test_char} 10 10 10 10"]})
    pred_df = pd.DataFrame(
        {
            "image_id": ["img1"],
            "labels": [f"{test_char} 15 15"],  # Center (15,15) is inside (10,10,10,10)
        }
    )
    metrics = calculate_kuzushiji_metrics(gt_df, pred_df)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    print(f"   Metrics verified: {metrics}")


def demo_dataset():
    print("\n=== Demonstrating library.dataset ===")

    # Initialize Train Dataset
    ds = KuzushijiDataset(mode="train", transforms=get_transforms(train=True))
    print(f"   Dataset length: {len(ds)}")

    # Test __getitem__
    img, target = ds[0]
    print(f"   Sample image shape: {img.shape}")

    assert isinstance(img, torch.Tensor)
    assert "boxes" in target
    assert "labels" in target
    assert "area" in target

    # Test collate_fn with a small batch
    batch_size = 2
    subset = Subset(ds, range(batch_size))
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn)
    images, targets = next(iter(loader))

    assert len(images) == batch_size
    assert len(targets) == batch_size
    print("   DataLoader and Collate verified.")
    return ds


def demo_pipeline(full_dataset):
    print("\n=== Demonstrating library.model and library.engine ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # 1. Initialize Model
    # num_classes = chars + background
    num_classes = len(full_dataset.char_to_int) + 1
    print(f"   Initializing model with {num_classes} classes...")
    model = get_kuzushiji_model(num_classes)
    model.to(device)

    # 2. Create tiny subsets for speed
    # We use create_compatible_subset to ensure engine functions work
    train_subset = create_compatible_subset(full_dataset, range(0, 4))
    val_subset = create_compatible_subset(full_dataset, range(4, 6))

    train_loader = DataLoader(
        train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    # 3. Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 4. Run Training Loop (1 Epoch)
    print("   Starting training demo (1 epoch)...")
    # Using provided engine function
    train_kuzushiji_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        num_epochs=1,
        patience=1,
        save_dir=WORKING_DIR,
    )

    # Verify model save
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model file was not saved."
    print("   Training verified.")

    # 5. Run Inference Demo
    print("   Starting inference demo...")
    # Load Test Dataset
    test_ds = KuzushijiDataset(mode="test", transforms=get_transforms(train=False))
    # Create subset for first 2 images only
    test_subset = create_compatible_subset(test_ds, range(0, 2))
    test_loader = DataLoader(
        test_subset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    submission_path = os.path.join(WORKING_DIR, "submission_demo.csv")
    inference(model, test_loader, device, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    # The inference function merges results with the full test ID list, so len should be full test size
    assert len(df_sub) == len(
        test_ds.df
    ), f"Submission length {len(df_sub)} != Test Set length {len(test_ds.df)}"
    print(f"   Inference verified. Submission generated with {len(df_sub)} rows.")


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    set_seeds(42)

    try:
        demo_utils()
        dataset = demo_dataset()
        demo_pipeline(dataset)
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
