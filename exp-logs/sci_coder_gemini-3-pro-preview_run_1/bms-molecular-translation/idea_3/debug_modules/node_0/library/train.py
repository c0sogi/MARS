import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import AverageMeter, compute_levenshtein
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms, collate_fn
from library.model import VisualTransformer


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    train_loader, model, criterion, optimizer, device, epoch, scheduler=None
):
    model.train()
    scaler = GradScaler()
    losses = AverageMeter("Loss", ":.4f")
    start = time.time()

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        # seq contains <SOS> ... <EOS> <PAD>
        seqs = batch["seq"].to(device)

        # Input to decoder: <SOS> ... token_n (exclude last token which is EOS or PAD)
        # Target: token_1 ... <EOS> (exclude first token which is SOS)
        decoder_input = seqs[:, :-1]
        targets = seqs[:, 1:]

        optimizer.zero_grad()

        with autocast():
            # pad_idx is used for masking inside the model
            # For the loss, we want to ignore the padding in the targets
            preds = model(
                images, decoder_input, pad_idx=0
            )  # 0 is PAD in tokenizer logic usually, checking tokenizer...
            # Tokenizer special tokens: ["<PAD>", "<SOS>", "<EOS>"] -> PAD is index 0.

            # Reshape for CrossEntropyLoss: (Batch * Seq_Len, Vocab_Size) vs (Batch * Seq_Len)
            loss = criterion(preds.reshape(-1, preds.size(-1)), targets.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

        if step % 100 == 0 and step > 0:
            print(
                f"Epoch: [{epoch}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return losses.avg


def greedy_decode(model, images, tokenizer, device, max_len=None):
    """
    Performs greedy decoding for a batch of images.
    """
    if max_len is None:
        max_len = Config.MAX_LEN

    model.eval()
    with torch.no_grad():
        # Encode
        memory = model.encoder(images)

        batch_size = images.size(0)
        # Initialize with SOS
        ys = torch.fill_(torch.zeros(batch_size, 1).long(), tokenizer.sos_token_id).to(
            device
        )

        # Iteratively generate
        for i in range(max_len - 1):
            # Create masks
            tgt_mask, tgt_pad_mask = model.make_masks(ys, tokenizer.pad_token_id)

            # Decode
            out = model.decoder(ys, memory, tgt_mask, tgt_pad_mask)

            # Get last token probability
            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)

            # Append
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Optional: Break if all sequences have hit EOS?
            # For batched simplicity, we just run to max_len or a reasonable limit.

    return ys


def validate(val_loader, model, criterion, tokenizer, device):
    model.eval()
    losses = AverageMeter("Val Loss", ":.4f")

    # For Levenshtein, we will sample a subset to keep runtime manageable
    # Validation on 380k images with greedy decode is too slow for every epoch.
    # We'll compute Loss on all, and Levenshtein on the first N batches.
    metric_batches = 20  # Approx 2500 samples if batch size 128
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            seqs = batch["seq"].to(device)

            decoder_input = seqs[:, :-1]
            targets = seqs[:, 1:]

            # 1. Validation Loss (Teacher Forcing)
            preds = model(images, decoder_input, pad_idx=tokenizer.pad_token_id)
            loss = criterion(preds.reshape(-1, preds.size(-1)), targets.reshape(-1))
            losses.update(loss.item(), images.size(0))

            # 2. Metric Calculation (Greedy Decode) - Subset only
            if step < metric_batches:
                # Generate
                generated_seqs = greedy_decode(
                    model, images, tokenizer, device, max_len=150
                )  # InChI avg len ~126

                # Convert to text
                pred_texts = [tokenizer.sequence_to_text(s) for s in generated_seqs]
                true_texts = batch["original_text"]

                predictions.extend(pred_texts)
                ground_truths.extend(true_texts)

    # Compute Levenshtein
    lev_score = compute_levenshtein(predictions, ground_truths) if predictions else 0.0

    return losses.avg, lev_score


def inference(test_loader, model, tokenizer, device):
    model.eval()
    all_preds = []
    all_ids = []

    print(f"Starting inference on {len(test_loader.dataset)} test images...")

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]

            # Greedy decode
            # Max len 400 covers almost all training examples.
            generated_seqs = greedy_decode(
                model, images, tokenizer, device, max_len=300
            )

            pred_texts = [tokenizer.sequence_to_text(s) for s in generated_seqs]

            all_preds.extend(pred_texts)
            all_ids.extend(image_ids)

            if step % 50 == 0:
                print(f"Inference step {step}/{len(test_loader)}")

    return all_ids, all_preds


def run_training():
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    if Config.DEBUG:
        print("DEBUG Mode: Subsampling data.")
        df_train = df_train.sample(n=2000, random_state=Config.SEED).reset_index(
            drop=True
        )
        df_val = df_val.sample(n=500, random_state=Config.SEED).reset_index(drop=True)

    # 2. Tokenizer
    tokenizer = Tokenizer(load_cached_data=True, debug=Config.DEBUG)

    # 3. Datasets & Loaders
    train_dataset = InChiDataset(df_train, tokenizer, transform=get_transforms("train"))
    val_dataset = InChiDataset(df_val, tokenizer, transform=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Model
    print("Initializing model...")
    model = VisualTransformer(vocab_size=len(tokenizer))
    model = model.to(device)

    # 5. Optimization
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycle or Cosine. Config has T_MAX, suggesting Cosine.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 6. Training Loop
    best_levenshtein = float("inf")
    best_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, scheduler
        )

        # Validate
        val_loss, val_lev = validate(val_loader, model, criterion, tokenizer, device)

        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val Lev (subset): {val_lev:.6f}")

        # Checkpointing based on Loss (more stable than subset metric)
        if val_loss < best_loss:
            print(
                f"Validation Loss improved ({best_loss:.6f} -> {val_loss:.6f}). Saving model..."
            )
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 7. Inference / Submission
    print("\nStarting submission generation...")

    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    # Load Test Data
    df_test = pd.read_csv(Config.TEST_METADATA)
    test_dataset = InChiDataset(df_test, tokenizer, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    ids, preds = inference(test_loader, model, tokenizer, device)

    # Save
    sub_df = pd.DataFrame({"image_id": ids, "InChI": preds})
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print("Head of submission:")
    print(sub_df.head())


if __name__ == "__main__":
    run_training()
