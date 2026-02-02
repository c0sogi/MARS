import os
import torch
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import ShowAndTell


def generate_submission(model=None, debug=False):
    """
    Generates predictions for the test set and saves to submission.csv.

    Args:
        model (torch.nn.Module, optional): Trained model instance. If None, the function will
                                           attempt to load the best checkpoint from disk.
        debug (bool): If True, runs inference on a small subset of the test data for debugging.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 1. Load Data
    # We only need the test loader and the tokenizer to decode predictions
    print(f"Initializing Test DataLoader (Debug={debug})...")
    _, _, test_loader, tokenizer = get_dataloaders(
        debug=debug, debug_size=Config.DEBUG_SIZE
    )

    # 2. Load Model
    if model is None:
        print("Loading model for submission...")
        vocab_size = len(tokenizer)
        model = ShowAndTell(vocab_size=vocab_size)
        model = model.to(device)

        # Prioritize the 'best' model, fallback to the latest checkpoint
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")
        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Best model not found at {checkpoint_path}. Checking for regular checkpoint."
            )
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint.pth.tar")

        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print(
                f"Error: No checkpoint found at {checkpoint_path}. Inference will use random weights."
            )

    # Set model to evaluation mode
    model.eval()

    predictions = []
    image_ids = []

    print("Generating predictions on test set...")

    # 3. Inference Loop
    with torch.no_grad():
        for i, (images, batch_ids) in enumerate(test_loader):
            images = images.to(device)

            # A. Encode Image
            # h, c shape: [num_layers, B, hidden_size]
            h, c = model.encoder(images)

            # B. Initialize Decoder Input
            # Start with <SOS> token for every sample in the batch
            # Shape: [B, 1]
            start_token = torch.full(
                (images.size(0), 1),
                tokenizer.sos_token_id,
                dtype=torch.long,
                device=device,
            )
            inputs = start_token

            # List to store predicted token indices for the current batch
            batch_preds_indices = []

            # C. Greedy Decoding
            for _ in range(Config.MAX_PRED_LEN):
                # Decoder forward pass
                # output_logits shape: [B, 1, vocab_size]
                output_logits, (h, c) = model.decoder(inputs, h, c)

                # Greedy selection: choose the token with the highest probability
                # predicted_token shape: [B, 1]
                predicted_token = output_logits.argmax(dim=2)

                batch_preds_indices.append(predicted_token)

                # Feed the predicted token as input for the next timestep
                inputs = predicted_token

            # Concatenate predictions along the sequence dimension: [B, MAX_PRED_LEN]
            batch_preds_indices = torch.cat(batch_preds_indices, dim=1)

            # D. Convert Indices to Text
            # The tokenizer handles stopping at <EOS>
            batch_pred_strs = [
                tokenizer.sequence_to_text(seq) for seq in batch_preds_indices
            ]

            predictions.extend(batch_pred_strs)
            image_ids.extend(batch_ids)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches.")

    # 4. Save Submission
    submission_df = pd.DataFrame({"image_id": image_ids, "InChI": predictions})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print("First 5 predictions:")
    print(submission_df.head())
