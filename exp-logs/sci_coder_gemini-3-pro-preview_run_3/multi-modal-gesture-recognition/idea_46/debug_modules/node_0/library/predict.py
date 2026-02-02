import os
import torch
from library.config import Paths, DataConfig
from library.utils import set_seed, rle_encode, predictions_to_string
from library.data_loader import get_dataloaders
from library.model import CKARFNet


def generate_submission(device_str="cuda", num_workers=2):
    """
    Generates the submission file for the test dataset using the best trained model.

    Args:
        device_str (str): Device to run inference on ('cuda' or 'cpu').
        num_workers (int): Number of workers for data loading.
    """
    # 1. Setup
    set_seed(42)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 2. Load Data
    # We use get_dataloaders to ensure consistent preprocessing.
    # batch_size argument here affects training loader, but test loader is hardcoded to 1
    # in get_dataloaders to handle variable sequence lengths.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=num_workers)

    # 3. Initialize Model
    print("Initializing model...")
    model = CKARFNet().to(device)

    # 4. Load Checkpoint
    checkpoint_path = os.path.join(Paths.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random initialization."
        )

    # 5. Inference Loop
    model.eval()
    submission_lines = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            sample_ids = batch["sample_id"]

            # Forward pass
            # CKARFNet returns a tuple of logits for deep supervision: (logits1, logits2, logits3)
            outputs = model(features)

            # We use the final stage output (Refinement Stage 2) for predictions
            final_logits = outputs[-1]  # Shape: (Batch, Time, NumClasses)

            # Decode probabilities
            # Argmax over the class dimension
            preds = torch.argmax(final_logits, dim=2)  # Shape: (Batch, Time)

            # Process each sample in the batch (Batch size is usually 1 for test)
            for i in range(len(sample_ids)):
                sample_id = sample_ids[i]
                pred_seq = preds[i].cpu().numpy()

                # Post-processing: Run-Length Encoding + Min Duration Filter
                # Filter out segments shorter than 5 frames and remove background (class 0)
                pred_gestures = rle_encode(pred_seq, min_duration=5, background_class=0)

                # Format output string
                line = predictions_to_string(sample_id, pred_gestures)
                submission_lines.append(line)

    # 6. Save Submission
    os.makedirs(Paths.SUBMISSION_DIR, exist_ok=True)
    out_file = Paths.SUBMISSION_FILE

    with open(out_file, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {out_file}")
    print(f"Total sequences processed: {len(submission_lines)}")
