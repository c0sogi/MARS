import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from library.utils import set_seed, load_checkpoint
from library.data_loader import LabelEncoder, WhaleDataset
from library.model import WhaleClassifier, generate_predictions


def generate_submission(
    checkpoint_path="./working/model_best.pth.tar",
    metadata_dir="./metadata",
    data_dir="./input",
    output_file="./submission/submission.csv",
    batch_size=32,
    num_workers=4,
    image_size=224,
    device=None,
    cache_dir="./working/idea_1",
    max_test_samples=None,
):
    """
    Loads a trained model and generates a submission file for the test set.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        metadata_dir (str): Directory containing metadata CSV files.
        data_dir (str): Root directory containing image files.
        output_file (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of subprocesses for data loading.
        image_size (int): Image size to resize inputs to.
        device (str, optional): Device to run inference on ('cuda' or 'cpu').
        cache_dir (str): Directory to load cached LabelEncoder classes from.
        max_test_samples (int, optional): Limit the number of test samples for debugging.
    """
    # Ensure reproducibility
    set_seed(42)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting inference on device: {device}")

    # 1. Setup Label Encoder
    # We need the exact same class mapping as used during training.
    # We attempt to load it from the cache directory.
    label_encoder = LabelEncoder()
    cache_file = os.path.join(cache_dir, "classes.npy")

    if os.path.exists(cache_file):
        print(f"Loading label classes from cache: {cache_file}")
        label_encoder.fit(None, cache_dir=cache_dir, load_cached_data=True)
    else:
        # Fallback: If cache is missing, we must refit on the training metadata
        # to ensure we have the correct class-to-index mapping.
        print("Cache not found. Loading train.csv to fit label encoder...")
        train_csv_path = os.path.join(metadata_dir, "train.csv")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(
                "Neither cached classes nor training metadata found."
            )

        df_train = pd.read_csv(train_csv_path)
        label_encoder.fit(df_train["Id"], cache_dir=cache_dir, load_cached_data=False)

    num_classes = label_encoder.num_classes()
    print(f"Total classes: {num_classes}")

    # 2. Setup Model
    print(f"Initializing model architecture (ResNet18) for {num_classes} classes...")
    model = WhaleClassifier(num_classes=num_classes)
    model = model.to(device)

    # Load weights
    print(f"Loading checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Use the library utility to load the checkpoint
    # This loads the 'state_dict' from the file into the model
    try:
        load_checkpoint(checkpoint_path, model)
    except RuntimeError as e:
        # Fallback for potential device mismatch issues if load_checkpoint doesn't handle map_location
        print(
            f"Standard load failed ({e}). Attempting to load with map_location={device}..."
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # 3. Setup Test Data
    test_csv_path = os.path.join(metadata_dir, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)

    # Optional debugging: limit dataset size
    if max_test_samples is not None:
        print(f"Debugging: Limiting test set to {max_test_samples} samples.")
        df_test = df_test.iloc[:max_test_samples]

    # Define transforms (must match training normalization)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )

    # Create Dataset and DataLoader
    test_dataset = WhaleDataset(
        df_test,
        root_dir=data_dir,
        transform=test_transform,
        label_encoder=None,  # Not needed for test set (no targets)
        is_test=True,  # Returns (image, filename)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Generate Predictions
    # This function handles the inference loop, top-5 selection, decoding, and file saving.
    generate_predictions(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
        device=device,
        output_file=output_file,
    )
