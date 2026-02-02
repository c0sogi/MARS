import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from library import config, model, data_loader, utils


def predict_sequence(model_instance, loader, device):
    """
    Applies the model in a sliding window fashion with 50% overlap,
    aggregates probabilities via averaging, and decodes the sequence.

    Args:
        model_instance: The loaded PyTorch model.
        loader: DataLoader for the test set.
        device: Torch device (CPU/GPU).

    Returns:
        List of lists, where each inner list contains the predicted gesture IDs for a sequence.
    """
    model_instance.eval()
    dataset = loader.dataset

    # Access raw skeletons to determine sequence lengths
    # dataset.skeletons is a list of arrays (T, Joints, 3)
    seq_lengths = [s.shape[0] for s in dataset.skeletons]
    num_seqs = len(dataset.skeletons)

    # Initialize buffers for probability accumulation
    # List of tensors: [(T, Classes), ...]
    # We accumulate probabilities on the device to avoid frequent CPU-GPU transfers
    seq_probs = [torch.zeros(l, config.NUM_CLASSES).to(device) for l in seq_lengths]
    seq_counts = [torch.zeros(l).to(device) for l in seq_lengths]

    with torch.no_grad():
        # We iterate linearly. Since shuffle=False, we can map batch items
        # to dataset indices sequentially using a global index tracker.
        global_idx = 0

        for features, _ in loader:
            batch_size = features.size(0)
            features = features.to(device)

            # Forward pass
            outputs = model_instance(features)
            # Use Stage 3 (Final Refinement) for prediction
            logits = outputs["stage3"]  # (Batch, Classes, Time)
            probs = F.softmax(logits, dim=1)

            # Permute to (Batch, Time, Classes) for easier slicing
            probs = probs.permute(0, 2, 1)

            for b in range(batch_size):
                if global_idx >= len(dataset.indices):
                    break

                # Identify which sequence and start frame this window belongs to
                seq_idx, start_frame = dataset.indices[global_idx]

                # Determine the valid range within the sequence
                seq_len = seq_lengths[seq_idx]
                end_frame = start_frame + config.WINDOW_SIZE

                # Clip to sequence length (ignore padding at the end of sequence if any)
                valid_end = min(end_frame, seq_len)

                # Length of valid data in this window
                # Note: The dataset pads the *input* window if it goes out of bounds.
                # We only want to accumulate the valid part corresponding to the real sequence.
                window_len_valid = valid_end - start_frame

                # Accumulate
                if window_len_valid > 0:
                    seq_probs[seq_idx][start_frame:valid_end] += probs[b][
                        :window_len_valid
                    ]
                    seq_counts[seq_idx][start_frame:valid_end] += 1.0

                global_idx += 1

    # Post-process and Decode
    predictions = []
    for i in range(num_seqs):
        # Average probabilities
        counts = seq_counts[i].unsqueeze(1)
        # Avoid division by zero (though coverage logic ensures counts >= 1 for valid frames)
        counts[counts == 0] = 1.0
        avg_probs = seq_probs[i] / counts

        # Convert to numpy for utils pipeline
        avg_probs_np = avg_probs.cpu().numpy()

        # Decode using utils pipeline (Argmax -> RLE -> Filter -> Remove BG)
        pred_seq = utils.process_predictions(
            avg_probs_np, min_length=config.MIN_GESTURE_LENGTH
        )
        predictions.append(pred_seq)

    return predictions


def run_inference(load_cached_data=True):
    """
    Main function to run the inference pipeline.

    Args:
        load_cached_data (bool): Whether to try loading pre-processed data from cache.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Test Data
    # We use process_metadata directly to avoid loading train/val data which saves memory/time
    test_data = data_loader.process_metadata(
        config.TEST_METADATA_PATH, "test", load_cached_data=load_cached_data
    )

    # 2. Create Dataset and Loader
    # Stride is set to WINDOW_SIZE // 2 for 50% overlap as per strategy
    test_ds = data_loader.GestureDataset(
        test_data, augment=False, stride=config.WINDOW_SIZE // 2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Initialize Model
    net = model.HSGKN().to(device)

    # 4. Load Weights
    if os.path.exists(config.BEST_MODEL_PATH):
        net.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    else:
        # Fallback or error handling if model is missing
        # In a competition context, this might imply using initialized weights or raising error
        pass

    # 5. Generate Predictions
    predictions = predict_sequence(net, test_loader, device)

    # 6. Save Submission
    sample_ids = test_ds.sample_ids
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    with open(config.SUBMISSION_PATH, "w") as f:
        for sid, pred_seq in zip(sample_ids, predictions):
            # Format: SessionID,label1,label2,...
            pred_str = ",".join(map(str, pred_seq))
            f.write(f"{sid},{pred_str}\n")
