import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import CNNTransformerCTC
from library.dataset import InChiDataset, CollateFn
from library.tokenizer import InChiTokenizer
from library.utils import load_checkpoint


def generate_submission(sample_size=None):
    """
    Generates the submission file for the test set by running inference using the trained model.

    Args:
        sample_size (int, optional): If provided, limits the inference to a subset of the test data.
                                     Useful for debugging or quick verification.
    """
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Initialize Model
    model = CNNTransformerCTC().to(device)

    # 3. Load Pre-trained Weights
    if not os.path.exists(Config.MODEL_PATH):
        print(
            f"Error: Model checkpoint not found at {Config.MODEL_PATH}. Please train the model first."
        )
        return

    # load_checkpoint handles loading state_dict into the model
    load_checkpoint(Config.MODEL_PATH, model)
    model.eval()

    # 4. Prepare Test Data
    if not os.path.exists(Config.TEST_CSV):
        print(f"Error: Test metadata file not found at {Config.TEST_CSV}.")
        return

    test_dataset = InChiDataset(
        csv_path=Config.TEST_CSV, mode="test", sample_size=sample_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=CollateFn(),
        pin_memory=(device.type == "cuda"),
    )

    print(f"Starting inference on {len(test_dataset)} test samples...")

    # 5. Inference Loop
    tokenizer = InChiTokenizer()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            image_ids = batch["image_ids"]

            # Forward pass: (N, Seq_Len, Vocab_Size)
            logits = model(images)

            # Decode predictions using CTC greedy decoding
            preds = tokenizer.decode_ctc_greedy(logits, batch_first=True)

            # Store results
            for img_id, pred in zip(image_ids, preds):
                results.append({"image_id": img_id, "InChI": pred})

    # 6. Save Submission
    df_submission = pd.DataFrame(results)

    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Generated {len(df_submission)} predictions.")
