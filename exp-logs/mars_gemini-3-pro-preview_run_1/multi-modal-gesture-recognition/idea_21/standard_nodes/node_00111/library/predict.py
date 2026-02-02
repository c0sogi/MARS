import os
import csv
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, decode_predictions_rle
from library.data_loader import GestureDataset, collate_fn
from library.model import GCINet


def generate_submission(checkpoint_path=None, batch_size=Config.BATCH_SIZE):
    """
    Generates the submission file for the test dataset using the trained model.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        batch_size (int): Batch size for inference. Defaults to Config.BATCH_SIZE.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Initializing model and loading weights from {checkpoint_path}...")
    model = GCINet().to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint file {checkpoint_path} not found!")
        return

    model.eval()

    # 3. Load Test Data
    # We use load_cached_data=True to leverage the caching mechanism in GestureDataset
    print("Initializing Test Dataset...")
    test_dataset = GestureDataset(split="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Inference Loop
    results = []
    print(f"Starting inference on {len(test_dataset)} samples...")

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["ids"]

            # Forward pass
            # Output: (B, T, NumClasses+1)
            logits = model(skeleton, audio, lengths, mask)

            # Process batch
            batch_size_curr = logits.shape[0]
            for i in range(batch_size_curr):
                sample_id = ids[i]
                length = lengths[i].item()

                # Extract valid sequence logits
                seq_logits = logits[i, :length, :]

                # Decode predictions
                # Applies Argmax -> Median Filter -> RLE -> Filter Background/Short Segments
                pred_list = decode_predictions_rle(seq_logits)

                results.append((sample_id, pred_list))

    # 5. Write Submission
    # Sort by Sample ID for consistency
    results.sort(key=lambda x: x[0])

    output_path = Config.SUBMISSION_PATH
    print(f"Writing submission to {output_path}...")

    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            for sample_id, preds in results:
                # Format: SessionID,Label1,Label2,...
                row = [sample_id] + preds
                writer.writerow(row)
        print("Submission file generated successfully.")
    except Exception as e:
        print(f"Error writing submission file: {e}")
