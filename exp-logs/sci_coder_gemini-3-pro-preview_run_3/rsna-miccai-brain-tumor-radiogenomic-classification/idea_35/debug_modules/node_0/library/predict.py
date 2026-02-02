import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.utils import seed_everything, get_device, load_data
from library.data_loader import DualStreamDataset
from library.model import DSSVNet


def generate_submission(
    model_path: str = "./working/idea_35/best_model.pth",
    output_file: str = "./submission/submission.csv",
    input_dir: str = "./input",
    cache_dir: str = "./working/idea_35/",
    batch_size: int = 16,
    limit_size: int = None,
    seed: int = 42,
):
    """
    Generates predictions for the test set using the trained DSSVNet model.

    Args:
        model_path: Path to the trained model checkpoint (.pth).
        output_file: Path where the submission CSV will be saved.
        input_dir: Root directory of input data.
        cache_dir: Directory to store/load cached data.
        batch_size: Batch size for inference.
        limit_size: Limit test set size for debugging.
        seed: Random seed for reproducibility.
    """
    # 1. Setup
    seed_everything(seed)
    device = get_device()

    # 2. Load Test Data
    # We use load_data directly to avoid loading train/val sets which saves time and memory.
    # The load_data function handles caching and metadata reading.
    X_test, y_test, ids_test = load_data(
        split="test",
        load_cached_data=True,
        limit_size=limit_size,
        cache_dir=cache_dir,
        input_dir=input_dir,
    )

    # Create Dataset and DataLoader
    test_dataset = DualStreamDataset(X_test, y_test, ids_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Load Model
    # Initialize model architecture. We set pretrained=False because we are loading
    # a specific checkpoint state_dict, avoiding unnecessary downloads.
    model = DSSVNet(pretrained=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for i, ((even_stream, odd_stream), _) in enumerate(test_loader):
            even_stream = even_stream.to(device)
            odd_stream = odd_stream.to(device)

            # Forward pass
            logits = model(even_stream, odd_stream)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Get IDs for this batch
            # Since shuffle=False, we can slice the original IDs array
            start_idx = i * batch_size
            end_idx = start_idx + len(probs)
            batch_ids = ids_test[start_idx:end_idx]

            all_probs.extend(probs)
            all_ids.extend(batch_ids)

    # 5. Save Submission
    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
