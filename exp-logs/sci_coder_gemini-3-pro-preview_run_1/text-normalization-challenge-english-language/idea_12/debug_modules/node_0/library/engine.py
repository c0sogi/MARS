import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm

from library.config import Config
from library.dataset import (
    TaggerDataset,
    FallbackDataset,
    collate_fn_tagger,
    collate_fn_fallback,
    build_vocabularies,
    KnowledgeBase,
)
from library.models import MorphEnhancedTagger, Seq2SeqFallback


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_class_weights(dataset, device):
    """
    Calculates Square-Root Smoothed class weights.
    W_c = sqrt(N / N_c)
    """
    print("Calculating class weights...")
    # Count classes from the dataset's internal indices
    # dataset.class_indices is a tensor of all class indices in the dataset
    all_classes = dataset.class_indices.tolist()
    counts = Counter(all_classes)

    num_classes = len(dataset.vocab_classes)
    total_samples = len(all_classes)

    weights = torch.ones(num_classes, dtype=torch.float32)

    for cls_idx in range(num_classes):
        # Skip PAD (index 0)
        if cls_idx == 0:
            weights[cls_idx] = 0
            continue

        count = counts.get(cls_idx, 0)
        if count > 0:
            # Square root smoothing
            weights[cls_idx] = np.sqrt(total_samples / count)
        else:
            # If class not present, assign 1.0 or a high weight?
            # Usually 1.0 is safe, or mean weight.
            weights[cls_idx] = 1.0

    return weights.to(device)


def train_tagger_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        # Unpack batch
        word_idxs, char_idxs, explicit_feats, class_idxs = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass
        # logits: (Batch, Seq, Num_Classes)
        logits = model(word_idxs, char_idxs, explicit_feats)

        # Reshape for Loss
        # Flatten batch and sequence dimensions
        # logits: (Batch * Seq, Num_Classes)
        # targets: (Batch * Seq)
        loss = criterion(logits.view(-1, logits.shape[-1]), class_idxs.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Accuracy calculation (ignoring PAD=0)
        preds = logits.argmax(dim=-1)
        mask = class_idxs != 0
        correct += (preds[mask] == class_idxs[mask]).sum().item()
        total += mask.sum().item()

    return epoch_loss / len(dataloader), correct / total if total > 0 else 0


def validate_tagger(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            word_idxs, char_idxs, explicit_feats, class_idxs = [
                b.to(device) for b in batch
            ]

            logits = model(word_idxs, char_idxs, explicit_feats)
            loss = criterion(logits.view(-1, logits.shape[-1]), class_idxs.view(-1))

            epoch_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = class_idxs != 0
            correct += (preds[mask] == class_idxs[mask]).sum().item()
            total += mask.sum().item()

    return epoch_loss / len(dataloader), correct / total if total > 0 else 0


def train_fallback_epoch(
    model, dataloader, criterion, optimizer, device, teacher_forcing_ratio
):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        src, tgt, cls = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass
        # output: (Batch, Max_Len, Vocab)
        # tgt includes <SOS> at index 0.
        output = model(src, tgt, cls, teacher_forcing_ratio)

        # Loss calculation
        # output predictions start from time step 1 (corresponding to tgt[:, 1])
        # output[:, 1:] aligns with tgt[:, 1:] (removing SOS)
        output_dim = output.shape[-1]

        # Slice outputs and targets to ignore the 0-th position (SOS/Init)
        # output[:, 1:] contains predictions for tgt[:, 1:]
        output_sliced = output[:, 1:].reshape(-1, output_dim)
        tgt_sliced = tgt[:, 1:].reshape(-1)

        loss = criterion(output_sliced, tgt_sliced)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def validate_fallback(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src, tgt, cls = [b.to(device) for b in batch]

            # Turn off teacher forcing for validation
            output = model(src, tgt, cls, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output_sliced = output[:, 1:].reshape(-1, output_dim)
            tgt_sliced = tgt[:, 1:].reshape(-1)

            loss = criterion(output_sliced, tgt_sliced)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


class Engine:
    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # Ensure directories exist
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    def train_tagger(self, epochs=Config.EPOCHS, limit=None):
        print("\n=== Training Tagger ===")

        # 1. Load Data
        train_dataset = TaggerDataset("train", limit=limit)
        val_dataset = TaggerDataset("val", limit=limit)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_tagger,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_tagger,
            num_workers=Config.NUM_WORKERS,
        )

        # 2. Model Setup
        vocab_words = train_dataset.vocab_words
        vocab_classes = train_dataset.vocab_classes
        vocab_chars = train_dataset.vocab_chars

        # Calculate Regex Feature Count from one sample
        # explicit_features is (Seq, Feat)
        num_explicit_features = train_dataset.explicit_features.shape[1]

        model = MorphEnhancedTagger(
            vocab_size=len(vocab_words),
            num_classes=len(vocab_classes),
            num_chars=len(vocab_chars),
            num_explicit_features=num_explicit_features,
        ).to(self.device)

        # 3. Optimization Setup
        weights = calculate_class_weights(train_dataset, self.device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)  # PAD=0
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        # 4. Training Loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_tagger_epoch(
                model, train_loader, criterion, optimizer, self.device
            )
            val_loss, val_acc = validate_tagger(
                model, val_loader, criterion, self.device
            )

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.TAGGER_MODEL_PATH)
                print("Saved Best Tagger Model.")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early Stopping Triggered.")
                    break

    def train_fallback(self, epochs=Config.EPOCHS, limit=None):
        print("\n=== Training Fallback Model ===")

        # 1. Load Data
        train_dataset = FallbackDataset("train", limit=limit)
        val_dataset = FallbackDataset("val", limit=limit)

        if len(train_dataset) == 0:
            print("No data for fallback training. Skipping.")
            return

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_fallback,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_fallback,
            num_workers=Config.NUM_WORKERS,
        )

        # 2. Model Setup
        vocab_chars = train_dataset.vocab_chars
        vocab_classes = train_dataset.vocab_classes

        model = Seq2SeqFallback(
            char_vocab_size=len(vocab_chars), num_classes=len(vocab_classes)
        ).to(self.device)

        # 3. Optimization
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD=0
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # 4. Loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_fallback_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                self.device,
                Config.TEACHER_FORCING_RATIO,
            )
            val_loss = validate_fallback(model, val_loader, criterion, self.device)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)
                print("Saved Best Fallback Model.")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early Stopping Triggered.")
                    break

    def generate_submission(self):
        print("\n=== Generating Submission ===")

        # 1. Load Resources
        # Vocabs
        vocab_words, vocab_chars, vocab_classes = build_vocabularies()

        # Knowledge Base
        kb = KnowledgeBase()
        kb.build()

        # Models
        # Re-instantiate models structure to load weights
        # We need explicit feature count. We can get it from Config or by checking a sample.
        # Config.REGEX_PATTERNS determines it.
        num_explicit_features = len(Config.REGEX_PATTERNS)

        tagger = MorphEnhancedTagger(
            vocab_size=len(vocab_words),
            num_classes=len(vocab_classes),
            num_chars=len(vocab_chars),
            num_explicit_features=num_explicit_features,
        ).to(self.device)

        fallback = Seq2SeqFallback(
            char_vocab_size=len(vocab_chars), num_classes=len(vocab_classes)
        ).to(self.device)

        # Load weights
        if os.path.exists(Config.TAGGER_MODEL_PATH):
            tagger.load_state_dict(
                torch.load(Config.TAGGER_MODEL_PATH, map_location=self.device)
            )
            print("Loaded Tagger model.")
        else:
            print("Warning: Tagger model not found. Predictions will be random.")

        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            fallback.load_state_dict(
                torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            )
            print("Loaded Fallback model.")
        else:
            print("Warning: Fallback model not found.")

        tagger.eval()
        fallback.eval()

        # 2. Data Loading for Inference
        # We need to iterate over the test data sentence by sentence to match TaggerDataset structure
        # AND keep track of the original IDs.

        print("Loading Test Data...")
        df_test = pd.read_csv(Config.TEST_DATA, dtype=str, keep_default_na=False)

        # Group df_test by sentence_id to align with Dataset
        # sort=False is critical to match the order TaggerDataset produces
        grouped_df = df_test.groupby("sentence_id", sort=False)

        # Create Dataset and Loader
        test_dataset = TaggerDataset("test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_tagger,
            num_workers=Config.NUM_WORKERS,
        )

        # 3. Inference Loop
        results = []

        # Create an iterator for the dataframe groups
        group_iterator = iter(grouped_df)

        # Pre-fetch special tokens for Fallback
        char_stoi = vocab_chars.stoi
        unk_char = vocab_chars["<UNK>"]
        sos_idx = vocab_chars["<SOS>"]
        eos_idx = vocab_chars["<EOS>"]

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                word_idxs, char_idxs, explicit_feats, _ = [
                    b.to(self.device) for b in batch
                ]

                # Tagger Prediction
                # logits: (Batch, Seq, Num_Classes)
                logits = tagger(word_idxs, char_idxs, explicit_feats)
                preds = logits.argmax(dim=-1)

                batch_size, seq_len = preds.shape

                # Process each sentence in the batch
                for i in range(batch_size):
                    # Get corresponding dataframe group
                    try:
                        _, group_df = next(group_iterator)
                    except StopIteration:
                        break

                    # Validate length alignment
                    # The padded sequence might be longer than the actual sentence
                    actual_len = len(group_df)

                    # Extract predictions for this sentence
                    sent_preds_idx = preds[i, :actual_len].cpu().tolist()

                    # Iterate tokens in the sentence
                    for j, (idx, row) in enumerate(group_df.iterrows()):
                        token_id = row["id"]
                        raw_token = row["before"]

                        pred_class_idx = sent_preds_idx[j]
                        pred_class = vocab_classes.lookup_token(pred_class_idx)

                        # Strategy: KB -> Fallback -> Copy

                        # 1. KB Lookup
                        normalized = kb.query(raw_token, pred_class)

                        if normalized is None:
                            # 2. Fallback Generation
                            # Only if not PLAIN/PUNCT (optimization) or always?
                            # Usually PLAIN maps to itself.
                            if pred_class == "PLAIN" or pred_class == "PUNCT":
                                normalized = raw_token
                            else:
                                # Prepare input for Fallback
                                # Convert raw token to char indices
                                src_indices = [
                                    char_stoi.get(c, unk_char) for c in raw_token
                                ]
                                src_tensor = torch.tensor(
                                    [src_indices], dtype=torch.long
                                ).to(self.device)
                                class_tensor = torch.tensor(
                                    [pred_class_idx], dtype=torch.long
                                ).to(self.device)

                                # Generate
                                gen_out = fallback.generate(
                                    src_tensor,
                                    class_tensor,
                                    max_len=Config.MAX_SEQ_LEN,
                                    sos_idx=sos_idx,
                                    eos_idx=eos_idx,
                                )

                                # Decode
                                # gen_out is (1, Max_Len)
                                gen_indices = gen_out[0].cpu().tolist()
                                decoded_chars = []
                                for char_idx in gen_indices:
                                    if char_idx == sos_idx:
                                        continue
                                    if char_idx == eos_idx:
                                        break
                                    decoded_chars.append(
                                        vocab_chars.lookup_token(char_idx)
                                    )

                                normalized = "".join(decoded_chars)

                        # Fallback for empty string or failure
                        if not normalized:
                            normalized = raw_token

                        # Store result
                        # CSV format requires quoting if contains comma/quotes,
                        # but pandas to_csv handles that.
                        results.append({"id": token_id, "after": normalized})

        # 4. Save Submission
        print(
            f"Saving submission with {len(results)} rows to {Config.SUBMISSION_PATH}..."
        )
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Done.")
