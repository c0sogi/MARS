import os
import torch
from torch_geometric.loader import DataLoader
from library.config import Config
from library.model import IceCubeDGCN
from library.data import IceCubeGraphDataset
from library.engine import Engine


def generate_submission(
    model_weights_path, output_csv_path=None, batch_ids=None, num_workers=4
):
    """
    Generates a submission file for the test set using a trained DGCN model.

    This function loads the test dataset (handling data processing and caching via
    IceCubeGraphDataset), loads the model weights, and runs the inference engine
    to produce the final submission CSV.

    Args:
        model_weights_path (str): Path to the saved model state dict (.pth file).
        output_csv_path (str, optional): Path to save the submission CSV.
                                         Defaults to Config.SUBMISSION_PATH.
        batch_ids (list, optional): List of batch IDs to process. If None, processes
                                    all batches found in the test metadata.
        num_workers (int): Number of subprocesses to use for data loading.
    """

    # 1. Configuration & Setup
    if output_csv_path is None:
        output_csv_path = Config.SUBMISSION_PATH

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    device = Config.DEVICE
    print(f"Initializing inference on device: {device}")

    # 2. Load Data
    # IceCubeGraphDataset handles loading metadata, filtering, processing raw pulses
    # into graph features, and caching the results to disk.
    print("Setting up Test Dataset...")
    test_dataset = IceCubeGraphDataset(mode="test", batch_ids=batch_ids)

    # Use PyG DataLoader to collate individual Data objects into batches
    # shuffle=False is mandatory for inference/submission to maintain order if needed,
    # though the submission file includes event_ids so strict order isn't critical
    # but good practice.
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Load Model
    print(f"Loading model architecture and weights from {model_weights_path}...")
    model = IceCubeDGCN()
    model.to(device)

    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model weights file not found at: {model_weights_path}"
        )

    # 4. Initialize Engine
    # We pass None for optimizer and scheduler as they are not required for inference.
    engine = Engine(model=model, device=device, optimizer=None, scheduler=None)

    # 5. Run Prediction
    # The engine.predict method handles the forward pass, vector-to-angle conversion,
    # and saving the DataFrame to CSV.
    print(f"Starting inference on approximately {len(test_dataset)} events...")
    engine.predict(test_loader, output_csv_path)

    print(f"Inference complete. Submission saved to {output_csv_path}")
