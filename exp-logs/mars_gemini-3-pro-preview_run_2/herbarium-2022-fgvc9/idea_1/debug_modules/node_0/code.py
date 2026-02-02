import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
    SUBMISSION_DIR,
    SEED,
    IMG_SIZE,
    BATCH_SIZE,
)
from library.utils import seed_everything
from library.dataset import PlantDataset, get_transforms
from library.prototype_manager import PrototypeClassifier


def create_subset_metadata(
    n_classes=10, samples_per_class_train=20, samples_per_class_val=5, n_test_samples=50
):
    """
    Creates subset CSVs for training, validation, and testing to allow for
    rapid demonstration of the pipeline.
    """
    print("Creating data subsets for rapid demonstration...")

    # Load full metadata
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # select top N classes by frequency in training data to ensure we have enough samples
    top_classes = df_train["label"].value_counts().head(n_classes).index.tolist()

    # Filter training data
    df_train_sub = (
        df_train[df_train["label"].isin(top_classes)]
        .groupby("label")
        .head(samples_per_class_train)
    )

    # Filter validation data
    df_val_sub = (
        df_val[df_val["label"].isin(top_classes)]
        .groupby("label")
        .head(samples_per_class_val)
    )

    # Sample test data (randomly)
    df_test_sub = df_test.sample(n=n_test_samples, random_state=SEED)

    # Save subsets to working directory
    train_subset_path = os.path.join(WORKING_DIR, "train_subset.csv")
    val_subset_path = os.path.join(WORKING_DIR, "val_subset.csv")
    test_subset_path = os.path.join(WORKING_DIR, "test_subset.csv")

    df_train_sub.to_csv(train_subset_path, index=False)
    df_val_sub.to_csv(val_subset_path, index=False)
    df_test_sub.to_csv(test_subset_path, index=False)

    print(f"  Train subset: {len(df_train_sub)} samples ({len(top_classes)} classes)")
    print(f"  Val subset:   {len(df_val_sub)} samples")
    print(f"  Test subset:  {len(df_test_sub)} samples")

    return train_subset_path, val_subset_path, test_subset_path


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Prepare Data (Subset for speed)
    train_csv, val_csv, test_csv = create_subset_metadata()

    # 3. Instantiate Datasets and DataLoaders
    # We use the get_transforms function from library.dataset
    transforms = get_transforms(img_size=IMG_SIZE)

    # Training Dataset
    train_dataset = PlantDataset(
        csv_file=train_csv, transform=transforms, test_mode=False
    )

    # Validation Dataset
    val_dataset = PlantDataset(csv_file=val_csv, transform=transforms, test_mode=False)

    # Test Dataset
    test_dataset = PlantDataset(csv_file=test_csv, transform=transforms, test_mode=True)

    # Verify Dataset Logic
    assert len(train_dataset) > 0, "Training dataset is empty."
    sample_img, sample_label = train_dataset[0]
    assert isinstance(sample_img, torch.Tensor), "Dataset should return a Tensor image."
    assert sample_img.shape == (
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {IMG_SIZE}, {IMG_SIZE})."
    assert isinstance(
        sample_label, int
    ), "Training dataset should return integer labels."

    # DataLoaders
    # Using a smaller batch size for the demo to ensure no OOM on smaller environments,
    # though A100 can handle much more.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 4. Model Initialization and Training (Prototype Computation)
    print("\nInitializing Prototype Classifier...")
    classifier = PrototypeClassifier()

    # Fit the model (compute prototypes)
    # We set load_cached_data=False to ensure we demonstrate the computation logic
    # on our new subset data rather than loading old cached files.
    classifier.fit(train_loader, load_cached_data=False)

    # Verify internal state
    assert classifier.prototypes is not None, "Prototypes were not computed."
    assert classifier.class_ids is not None, "Class IDs were not stored."
    assert (
        classifier.prototypes.shape[1] == 1280
    ), "Prototype feature dimension mismatch (Expected 1280 for EfficientNet-B0)."

    # 5. Evaluation
    print("\nEvaluating on Validation Set...")
    f1_score = classifier.evaluate(val_loader)

    # Verify Metric
    assert 0.0 <= f1_score <= 1.0, "F1 Score is out of valid range [0, 1]."
    print(f"Verified F1 Score: {f1_score:.4f}")

    # 6. Inference and Submission
    print("\nGenerating Submission for Test Set...")
    classifier.generate_submission(test_loader)

    # Verify Submission File
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["Id", "Predicted"], "Submission columns mismatch."
    assert len(df_sub) == len(
        test_dataset
    ), "Submission row count does not match test dataset size."
    assert (
        df_sub["Id"].dtype == int or df_sub["Id"].dtype == "int64"
    ), "Id column should be integer."
    assert (
        df_sub["Predicted"].dtype == int or df_sub["Predicted"].dtype == "int64"
    ), "Predicted column should be integer."

    print(f"Successfully generated submission with {len(df_sub)} rows.")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
