import os
import torch
import numpy as np
import library.config as config
from library.utils import set_seed
from library.model import DCSGCN
from library.data_loader import get_loaders
from library.train import decode_predictions


def predict_sequence(model, features, mask):
    """
    Runs the model on input features to get class probabilities.

    Args:
        model (nn.Module): The trained model.
        features (torch.Tensor): Input features (B, T, D).
        mask (torch.Tensor): Sequence mask (B, T).

    Returns:
        torch.Tensor: Class probabilities from Stage 3 (B, T, C).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(features, mask)
        # Use Stage 3 outputs for final prediction as per design
        cls_probs, _ = outputs["stage3"]
    return cls_probs


def run_inference(
    checkpoint_path=None,
    output_path=None,
    batch_size=None,
    device=None,
):
    """
    Orchestrates the inference pipeline: loads data, loads model,
    predicts, post-processes, and saves submission.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device to run inference on.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if output_path is None:
        output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if batch_size is None:
        batch_size = config.HYPERPARAMS["batch_size"]

    set_seed(config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Test Data
    # get_loaders returns train, val, test. We only need test.
    _, _, test_loader = get_loaders(batch_size=batch_size)

    # Initialize Model
    model = DCSGCN().to(device)

    # Load Weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    predictions = []

    print(f"Running inference on {len(test_loader.dataset)} samples...")

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]
            lengths = batch["lengths"]

            # 1. Predict Sequence
            # Returns (B, T, C) probabilities
            probs = predict_sequence(model, features, mask)

            # Convert to CPU numpy for post-processing
            probs_np = probs.cpu().numpy()

            # 2. Post-process Labels
            # decode_predictions implements:
            # - Argmax
            # - Median Filter (size=7, mode='nearest')
            # - Label Collapsing
            # - Background Removal
            batch_decoded = decode_predictions(probs_np, lengths)

            # 3. Format Predictions
            for i, seq in enumerate(batch_decoded):
                sid = sample_ids[i]
                # Format: SessionID,label1,label2,...
                seq_str = ",".join(map(str, seq))
                predictions.append(f"{sid},{seq_str}")

    # Save Submission
    with open(output_path, "w") as f:
        for line in predictions:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")
