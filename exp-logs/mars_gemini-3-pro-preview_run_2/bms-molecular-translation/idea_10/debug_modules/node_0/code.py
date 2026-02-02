import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.tokenizer import InChiTokenizer
from library.dataset import ChemicalImageDataset, ChemicalCollate
from library.model import HybridResNetTransformer
from library.trainer import Trainer
from library.inference import BeamSearchDecoder


def main():
    print("Initializing Configuration...")
    # Initialize Config in debug mode for speed
    # This reduces epochs to 2, batch size to 8, and subsets the data
    config = Config(debug=True)

    # Ensure reproducibility
    seed_everything(config.seed)

    print(f"Device: {config.device}")
    print(f"Working Directory: {config.working_dir}")

    # ---------------------------------------------------------
    # 1. Tokenizer Demonstration
    # ---------------------------------------------------------
    print("\n--- Tokenizer Demonstration ---")
    # Initialize tokenizer (will build vocab from metadata/train.csv if not cached)
    tokenizer = InChiTokenizer(config, load_cached_data=True)

    print(f"Vocabulary Size: {len(tokenizer)}")

    # Test encoding/decoding
    test_str = "InChI=1S/H2O/h1H2"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_str}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert decoded == test_str, "Decoded string does not match original."
    assert encoded[0] == tokenizer.SOS_ID, "Encoded sequence must start with SOS."
    assert encoded[-1] == tokenizer.EOS_ID, "Encoded sequence must end with EOS."

    # ---------------------------------------------------------
    # 2. Dataset and DataLoader Demonstration
    # ---------------------------------------------------------
    print("\n--- Dataset and DataLoader Demonstration ---")
    # Initialize dataset in train mode
    # Debug mode ensures we only load a small subset
    train_dataset = ChemicalImageDataset(config, tokenizer, mode="train")
    print(f"Train Dataset Size (Debug): {len(train_dataset)}")

    assert len(train_dataset) > 0, "Dataset should not be empty."

    # Initialize Collate function
    collate_fn = ChemicalCollate(config, pad_id=tokenizer.PAD_ID)

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["images"]
    labels = batch["labels"]
    lengths = batch["lengths"]
    image_ids = batch["image_ids"]

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")
    print(f"Batch Lengths: {lengths}")

    # Validations
    # Image shape: (B, 1, H, W). H should be config.image_height (192).
    # W should be multiple of config.horizontal_stride (4).
    assert images.size(0) == config.batch_size
    assert images.size(1) == 1
    assert images.size(2) == config.image_height
    assert images.size(3) % config.horizontal_stride == 0
    assert labels.size(0) == config.batch_size

    # ---------------------------------------------------------
    # 3. Model Architecture Demonstration
    # ---------------------------------------------------------
    print("\n--- Model Architecture Demonstration ---")
    model = HybridResNetTransformer(config, tokenizer)
    model.to(config.device)

    # Move batch to device
    images = images.to(config.device)
    labels = labels.to(config.device)
    lengths = lengths.to(config.device)

    # Forward Pass
    outputs = model(images, targets=labels)

    ctc_logits = outputs["ctc_logits"]
    decoder_logits = outputs["decoder_logits"]

    print(f"CTC Logits Shape: {ctc_logits.shape}")
    print(f"Decoder Logits Shape: {decoder_logits.shape}")

    # Validations
    # CTC Logits: (B, Seq_Len_Encoder, Vocab)
    # ResNet34 with stride 4 horizontal -> Width / 4
    expected_seq_len = images.size(3) // 4
    assert ctc_logits.size(0) == config.batch_size
    assert ctc_logits.size(1) == expected_seq_len
    assert ctc_logits.size(2) == len(tokenizer)

    # Decoder Logits: (B, Target_Len - 1, Vocab) (due to teacher forcing shifting)
    assert decoder_logits.size(0) == config.batch_size
    assert decoder_logits.size(1) == labels.size(1) - 1
    assert decoder_logits.size(2) == len(tokenizer)

    # Loss Calculation
    loss_dict = model.calc_loss(outputs, labels, lengths, ctc_weight=config.ctc_weight)
    print(f"Calculated Loss: {loss_dict['total'].item():.4f}")
    assert not torch.isnan(loss_dict["total"]), "Loss should not be NaN"

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n--- Training Loop Demonstration ---")
    # We use the Trainer class which handles the loop
    # In debug mode, this runs for 2 epochs on a tiny subset
    trainer = Trainer(config)

    # Override print frequency to see logs for every step in this short demo
    trainer.config.print_freq = 1

    print("Starting training (fit)...")
    trainer.fit()

    assert os.path.exists(config.model_path), "Best model checkpoint was not saved."

    # ---------------------------------------------------------
    # 5. Inference Demonstration
    # ---------------------------------------------------------
    print("\n--- Inference Demonstration ---")

    # Initialize Beam Search Decoder
    # We use a small beam size for demonstration speed
    decoder = BeamSearchDecoder(model, tokenizer, beam_size=2, max_len=50)

    print("Running Beam Search Decoding on the sample batch...")
    # Decode the batch we fetched earlier
    best_sequences = decoder.decode(images)

    print(f"Best Sequences Shape: {best_sequences.shape}")

    # Convert indices to strings
    decoded_preds = []
    for i in range(len(best_sequences)):
        seq = best_sequences[i]
        pred_str = tokenizer.decode(seq)
        decoded_preds.append(pred_str)

    print("\nSample Predictions:")
    for i in range(min(3, len(decoded_preds))):
        print(f"ID: {image_ids[i]}")
        print(f"Pred: {decoded_preds[i]}")
        # Decode the ground truth label for comparison
        gt_str = tokenizer.decode(labels[i])
        print(f"True: {gt_str}")
        print("-" * 20)

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
