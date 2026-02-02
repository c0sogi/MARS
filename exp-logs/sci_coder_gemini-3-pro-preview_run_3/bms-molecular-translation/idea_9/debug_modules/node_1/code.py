import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import seed_everything
from library.tokenizer import InchiTokenizer
from library.dataset import InchiDataset
from library.model import MixerTransformer
from library.engine import train_fn, eval_fn


def create_mini_metadata(source_path, dest_path, num_rows=32):
    """Creates a small subset of metadata for demonstration purposes."""
    print(f"Creating mini metadata: {dest_path} from {source_path}")
    df = pd.read_csv(source_path)
    df_mini = df.head(num_rows).copy()
    df_mini.to_csv(dest_path, index=False)
    return df_mini


def run_demonstration():
    # 1. Setup
    print("--- 1. Setup ---")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Define paths
    full_train_meta = "./metadata/train_metadata.csv"
    full_val_meta = "./metadata/val_metadata.csv"

    # Create working directory for mini files
    work_dir = "./working/demo_files"
    os.makedirs(work_dir, exist_ok=True)

    mini_train_meta = os.path.join(work_dir, "mini_train.csv")
    mini_val_meta = os.path.join(work_dir, "mini_val.csv")

    # Create mini datasets to speed up initialization (atom count calculation)
    create_mini_metadata(full_train_meta, mini_train_meta, num_rows=32)
    create_mini_metadata(full_val_meta, mini_val_meta, num_rows=16)

    # 2. Tokenizer Demonstration
    print("\n--- 2. Tokenizer Demonstration ---")
    # Initialize tokenizer with the mini train file to build vocab quickly
    # In a real scenario, you'd use the full file or a pre-computed vocab.json
    tokenizer = InchiTokenizer(metadata_path=mini_train_meta, load_cached_data=False)

    sample_text = "InChI=1S/H2O/h1H2"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Verification
    assert tokenizer.sos_token_id == encoded[0].item(), "First token must be SOS"
    assert tokenizer.eos_token_id == encoded[-1].item(), "Last token must be EOS"
    # Note: Decoded string might exclude special tokens, so it should match original if vocab covers it
    # Since we built vocab from mini_train, 'H', '2', 'O' might be in it.

    # 3. Dataset Demonstration
    print("\n--- 3. Dataset Demonstration ---")
    # Initialize dataset
    train_dataset = InchiDataset(
        metadata_path=mini_train_meta,
        tokenizer=tokenizer,
        max_length=128,
        mode="train",
        load_cached_data=False,  # Force re-compute on mini data to avoid loading full cache if exists
    )

    print(f"Dataset length: {len(train_dataset)}")
    sample_item = train_dataset[0]

    print("Sample Item Keys:", sample_item.keys())
    print("Image Shape:", sample_item["image"].shape)
    print("Input IDs Shape:", sample_item["input_ids"].shape)
    print("Atom Counts Shape:", sample_item["atom_counts"].shape)

    # Verification
    assert sample_item["image"].dim() == 3, "Image must be 3D tensor (C, H, W)"
    assert sample_item["input_ids"].dim() == 1, "Input IDs must be 1D tensor"
    assert (
        sample_item["atom_counts"].shape[0] == 12
    ), "Atom counts must have size 12 (len(ATOM_VOCAB))"

    # 4. Model Demonstration
    print("\n--- 4. Model Demonstration ---")
    # Instantiate a smaller model for speed
    model = MixerTransformer(
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=128,  # Reduced for demo
        encoder_depth=2,  # Reduced for demo
        token_dim=64,  # Reduced for demo
        channel_dim=256,  # Reduced for demo
        decoder_depth=2,  # Reduced for demo
        nhead=4,  # Reduced for demo
        vocab_size=len(tokenizer),
        max_len=512,
        dropout=0.1,
    ).to(device)

    # Create dummy batch
    batch_size = 4
    dummy_imgs = torch.randn(batch_size, 3, 384, 384).to(device)
    dummy_text = torch.randint(0, len(tokenizer), (batch_size, 64)).to(device)
    dummy_mask = dummy_text == tokenizer.pad_token_id

    # Forward pass
    logits, aux_preds = model(
        dummy_imgs, text_input_ids=dummy_text, padding_mask=dummy_mask
    )

    print("Logits Shape:", logits.shape)
    print("Aux Preds Shape:", aux_preds.shape)

    # Verification
    assert logits.shape == (batch_size, 64, len(tokenizer)), "Logits shape mismatch"
    assert aux_preds.shape == (batch_size, 12), "Aux preds shape mismatch"

    # Test Generation (Inference)
    print("Testing Generation...")
    gen_preds = model.generate(dummy_imgs[:1], tokenizer, max_len=20, device=device)
    print("Generated Prediction:", gen_preds[0])

    # 5. Training Loop Demonstration
    print("\n--- 5. Training Loop Demonstration ---")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_aux = nn.SmoothL1Loss()

    print("Starting training for 1 epoch on mini dataset...")
    avg_loss = train_fn(
        dataloader=train_loader,
        model=model,
        criterion_ce=criterion_ce,
        criterion_aux=criterion_aux,
        optimizer=optimizer,
        device=device,
        epoch=0,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"Training finished. Average Loss: {avg_loss:.4f}")

    # 6. Evaluation Loop Demonstration
    print("\n--- 6. Evaluation Loop Demonstration ---")
    val_dataset = InchiDataset(
        metadata_path=mini_val_meta,
        tokenizer=tokenizer,
        max_length=128,
        mode="valid",
        load_cached_data=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print("Starting evaluation on mini validation set...")
    score = eval_fn(
        dataloader=val_loader, model=model, tokenizer=tokenizer, device=device
    )
    print(f"Evaluation finished. Score (Levenshtein Dist): {score:.4f}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
