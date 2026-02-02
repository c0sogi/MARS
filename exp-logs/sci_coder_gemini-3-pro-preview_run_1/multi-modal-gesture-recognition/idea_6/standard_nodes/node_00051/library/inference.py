import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed, decode_predictions
from library.data_loader import GestureDataset
from library.model import CGRNet


def inference_collate_fn(batch):
    """
    Custom collate function for inference that handles padding of inputs
    and generation of masks, ignoring labels since they are not needed/available for test.
    """
    # Sort by length (descending) to optimize padding (though not strictly required for this model)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad inputs
    skeletons_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Create Mask (True for valid positions, False for padding)
    max_len = skeletons_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return {
        "skeleton": skeletons_padded,
        "audio": audios_padded,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": sample_ids,
    }


def generate_predictions(load_cached_data=True, max_samples=None):
    """
    Generates predictions for the test set using the best trained model.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        max_samples (int, optional): Limit the number of samples for debugging.
    """
    set_seed(Config.SEED)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Loading test data (Cached: {load_cached_data})...")
    test_dataset = GestureDataset(
        split="test", load_cached_data=load_cached_data, max_samples=max_samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=inference_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    print("Initializing model...")
    model = CGRNet().to(Config.DEVICE)

    # Load best checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading weights from {Config.BEST_MODEL_PATH}...")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {Config.BEST_MODEL_PATH}. Using random weights."
        )

    model.eval()
    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            skeleton = batch["skeleton"].to(Config.DEVICE)
            audio = batch["audio"].to(Config.DEVICE)
            mask = batch["mask"].to(Config.DEVICE)
            sample_ids = batch["sample_ids"]
            lengths = batch["lengths"]

            # Forward pass
            logits = model(skeleton, audio, mask)  # (B, T, NumClasses)

            # Decode predictions for each sample in the batch
            for i in range(logits.size(0)):
                valid_len = lengths[i]
                # Slice logits to the valid sequence length
                seq_logits = logits[i, :valid_len, :]

                # Decode frame-wise logits to sequence of gesture IDs
                pred_seq = decode_predictions(
                    seq_logits, threshold=5, bg_class=Config.BACKGROUND_CLASS_ID
                )

                # Format: SessionID,1,2,3
                pred_str = ",".join(map(str, pred_seq))
                results.append(f"{sample_ids[i]},{pred_str}")

    # Save submission file
    print(f"Saving {len(results)} predictions to {Config.SUBMISSION_PATH}...")
    with open(Config.SUBMISSION_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")

    print("Inference completed successfully.")
