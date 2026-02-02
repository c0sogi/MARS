import os
import torch
import shutil
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.tokenizer import Tokenizer
from library.dataset import get_train_val_loaders, get_test_loader
from library.model_components import PatchEmbed, MixerBlock, Attention
from library.model import InChIModel
from library.trainer import Trainer
from library.inference import run_inference


def run_demo():
    print("=== Starting Demonstration of Image-to-InChI Pipeline ===")

    # 1. Setup Configuration for Demo
    print("\n[1] Setting up Configuration...")
    config = Config()

    # Override paths and parameters for a quick demo execution
    config.working_dir = "./working/demo_execution"
    config.checkpoint_dir = os.path.join(config.working_dir, "checkpoints")
    config.submission_dir = config.working_dir
    config.submission_path = os.path.join(config.submission_dir, "submission.csv")
    config.vocab_path = os.path.join(config.working_dir, "vocab.json")

    # Ensure directories exist
    os.makedirs(config.working_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Speed optimizations for demo
    config.debug = True
    config.debug_sample_size = 32  # Small sample size for speed
    config.epochs = 1
    config.batch_size = 4
    config.num_workers = 2
    config.print_freq = 1
    config.encoder_depth = 2  # Reduce model depth for speed
    config.decoder_dim = 256  # Reduce hidden dim
    config.encoder_dim = 256
    config.embedding_dim = 256
    config.token_mixing_dim = 128
    config.channel_mixing_dim = 512

    # Set seed for reproducibility
    seed_everything(config.seed)
    print("Configuration setup complete.")

    # 2. Tokenizer Demonstration
    print("\n[2] Initializing Tokenizer...")
    # We force reload_cached_data=False to verify building from metadata works
    # Note: In a real run, we might want to use the existing vocab if valid,
    # but here we test the logic.
    tokenizer = Tokenizer(config, load_cached_data=False)

    print(f"Vocabulary size: {len(tokenizer)}")

    # Validation: Test encoding and decoding
    sample_text = "InChI=1S/H2O/h1H2"
    encoded = tokenizer.text_to_sequence(sample_text)
    decoded = tokenizer.sequence_to_text(encoded)

    print(f"Sample Text: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert sample_text == decoded, "Tokenizer encoding/decoding cycle failed!"
    assert (
        encoded[0] == tokenizer.stoi[config.sos_token]
    ), "Sequence must start with SOS"
    assert encoded[-1] == tokenizer.stoi[config.eos_token], "Sequence must end with EOS"
    print("Tokenizer logic verified.")

    # 3. Dataset and DataLoader Demonstration
    print("\n[3] Creating DataLoaders...")
    train_loader, val_loader = get_train_val_loaders(config, tokenizer)

    # Fetch a batch to verify shapes
    images, labels, lengths = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")  # Should be (B, 1, 256, 256)
    print(f"Batch Label Shape: {labels.shape}")  # Should be (B, Max_Len_In_Batch)
    print(f"Batch Lengths: {lengths}")

    assert images.shape == (
        config.batch_size,
        config.in_channels,
        config.image_size,
        config.image_size,
    )
    assert labels.shape[0] == config.batch_size
    assert lengths.shape[0] == config.batch_size
    print("DataLoader shapes verified.")

    # 4. Model Components Verification
    print("\n[4] Verifying Model Components...")

    # Test PatchEmbed
    patch_embed = PatchEmbed(
        img_size=config.image_size,
        patch_size=config.patch_size,
        in_chans=config.in_channels,
        embed_dim=config.encoder_dim,
    )
    dummy_img = torch.randn(
        config.batch_size, config.in_channels, config.image_size, config.image_size
    )
    patches = patch_embed(dummy_img)
    expected_num_patches = (config.image_size // config.patch_size) ** 2
    print(f"Patch Embed Output Shape: {patches.shape}")
    assert patches.shape == (
        config.batch_size,
        expected_num_patches,
        config.encoder_dim,
    )

    # Test MixerBlock
    mixer_block = MixerBlock(
        num_tokens=expected_num_patches,
        dim=config.encoder_dim,
        token_mixing_dim=config.token_mixing_dim,
        channel_mixing_dim=config.channel_mixing_dim,
    )
    mixed = mixer_block(patches)
    assert mixed.shape == patches.shape
    print("Model components shapes verified.")

    # 5. Full Model Initialization and Forward Pass
    print("\n[5] Initializing Full Model...")
    model = InChIModel(config, vocab_size=len(tokenizer))
    model.to(config.device)

    # Move dummy batch to device
    images = images.to(config.device)
    labels = labels.to(config.device)

    # Forward pass (Training mode)
    outputs = model(images, labels)
    print(f"Model Forward Output Shape: {outputs.shape}")
    # Output should be (B, Max_Len_Label, Vocab_Size)
    assert outputs.shape == (config.batch_size, labels.size(1), len(tokenizer))

    # Generation pass (Inference mode)
    generated_indices = model.generate(
        images,
        max_len=20,
        sos_token_idx=tokenizer.stoi[config.sos_token],
        eos_token_idx=tokenizer.stoi[config.eos_token],
    )
    print(f"Model Generate Output Shape: {generated_indices.shape}")
    assert generated_indices.shape == (config.batch_size, 20)
    print("Model forward and generate methods verified.")

    # 6. Trainer Execution
    print("\n[6] Running Training Loop (1 Epoch)...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Run fit
    trainer.fit()

    # Check if model checkpoint was saved
    checkpoint_path = os.path.join(config.checkpoint_dir, "model_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint successfully saved at {checkpoint_path}")
    else:
        # It's possible validation didn't improve if initialized randomly and run for 1 epoch,
        # but usually metric starts at inf so it should save.
        print(
            "Note: No checkpoint saved (validation metric might not have improved or crashed)."
        )

    # 7. Inference Pipeline
    print("\n[7] Running Inference Pipeline...")
    # We use the run_inference function wrapper, overriding config with our demo config
    # We need to make sure test_metadata exists (it was generated by the metadata script)
    if os.path.exists(config.test_metadata_path):
        run_inference(
            config=config, debug=True, debug_sample_size=10, load_cached_vocab=True
        )

        if os.path.exists(config.submission_path):
            print(f"Submission file created at {config.submission_path}")
            df_sub = pd.read_csv(config.submission_path)
            print(f"Submission rows: {len(df_sub)}")
            assert len(df_sub) > 0
        else:
            raise FileNotFoundError("Submission file was not created.")
    else:
        print("Test metadata not found, skipping inference step.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
