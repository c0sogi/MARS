import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import time
from library.config import (
    ProjectConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    set_seed,
)
from library.data_utils import (
    load_dataset_raw,
    build_vocabularies,
    build_knowledge_base,
    TaggerDataset,
    Seq2SeqDataset,
    collate_fn_tagger,
    collate_fn_seq2seq,
    Vocabulary,
)

# ==========================================
# 1. Model Architectures
# ==========================================


class MultiGranularityTagger(nn.Module):
    def __init__(self, vocab_words_size, vocab_chars_size, num_classes):
        super(MultiGranularityTagger, self).__init__()

        # 1. Word Embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_words_size,
            embedding_dim=ModelConfig.TAGGER_WORD_EMBED_DIM,
            padding_idx=0,
        )

        # 2. Character Branch (CNN)
        self.char_embedding = nn.Embedding(
            num_embeddings=vocab_chars_size,
            embedding_dim=ModelConfig.TAGGER_CHAR_EMBED_DIM,
            padding_idx=0,
        )
        self.char_cnn = nn.Conv1d(
            in_channels=ModelConfig.TAGGER_CHAR_EMBED_DIM,
            out_channels=ModelConfig.TAGGER_CNN_FILTERS,
            kernel_size=ModelConfig.TAGGER_CNN_KERNEL_SIZE,
            padding=1,
        )

        # 3. Backbone (Bi-LSTM)
        # Input dim = Word Embed + Char CNN Filters
        self.fusion_dim = (
            ModelConfig.TAGGER_WORD_EMBED_DIM + ModelConfig.TAGGER_CNN_FILTERS
        )

        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=ModelConfig.TAGGER_LSTM_HIDDEN_DIM,
            num_layers=ModelConfig.TAGGER_LSTM_LAYERS,
            batch_first=True,
            bidirectional=ModelConfig.TAGGER_BIDIRECTIONAL,
            dropout=(
                ModelConfig.TAGGER_DROPOUT if ModelConfig.TAGGER_LSTM_LAYERS > 1 else 0
            ),
        )

        # 4. Classifier
        self.lstm_out_dim = ModelConfig.TAGGER_LSTM_HIDDEN_DIM * (
            2 if ModelConfig.TAGGER_BIDIRECTIONAL else 1
        )
        self.fc_dropout = nn.Dropout(ModelConfig.TAGGER_DROPOUT)
        self.fc = nn.Linear(self.lstm_out_dim, num_classes)

    def forward(self, word_ids, char_ids):
        # word_ids: (B)
        # char_ids: (B, T_char)

        # Word Branch
        w_emb = self.word_embedding(word_ids)  # (B, W_Dim)

        # Char Branch
        c_emb = self.char_embedding(char_ids)  # (B, T, C_Dim)
        c_emb = c_emb.permute(0, 2, 1)  # (B, C_Dim, T)

        c_conv = F.relu(self.char_cnn(c_emb))  # (B, Filters, T)
        # Global Max Pooling
        c_pool = F.max_pool1d(c_conv, kernel_size=c_conv.shape[2]).squeeze(
            2
        )  # (B, Filters)

        # Fusion
        combined = torch.cat([w_emb, c_pool], dim=1)  # (B, Fusion_Dim)

        # LSTM Backbone
        # Unsqueeze to create sequence dimension (B, 1, Fusion_Dim)
        lstm_in = combined.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)  # (B, 1, LSTM_Out_Dim)

        # Classifier
        logits = self.fc(self.fc_dropout(lstm_out.squeeze(1)))  # (B, Num_Classes)
        return logits


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.SEQ_DROPOUT)

    def forward(self, src, class_ids):
        # src: (B, T)
        # class_ids: (B)
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Condition hidden state on class
        # Add class embedding to the final hidden state of the encoder
        class_emb = self.class_embedding(class_ids).unsqueeze(0)  # (1, B, Hidden)
        hidden = hidden + class_emb

        return outputs, hidden


class Seq2SeqAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(Seq2SeqAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, B, Hidden) -> (B, Hidden)
        # encoder_outputs: (B, T, Hidden)

        src_len = encoder_outputs.shape[1]
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)  # (B, T, Hidden)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (B, T)
        return F.softmax(attention, dim=1)


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = Seq2SeqAttention(hidden_dim)
        self.dropout = nn.Dropout(ModelConfig.SEQ_DROPOUT)

    def forward(self, input_step, hidden, encoder_outputs):
        # input_step: (B)
        # hidden: (1, B, Hidden)
        # encoder_outputs: (B, T, Hidden)

        input_step = input_step.unsqueeze(1)  # (B, 1)
        embedded = self.dropout(self.embedding(input_step))  # (B, 1, Emb)

        # Calculate attention weights
        a = self.attention(hidden, encoder_outputs)  # (B, T)
        a = a.unsqueeze(1)  # (B, 1, T)

        # Weighted sum of encoder outputs
        weighted = torch.bmm(a, encoder_outputs)  # (B, 1, Hidden)

        # RNN input: Concatenate embedding and context
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output.squeeze(1))

        return prediction, hidden


class Seq2SeqNormalizer(nn.Module):
    def __init__(self, vocab_chars_size, num_classes):
        super(Seq2SeqNormalizer, self).__init__()
        self.encoder = Seq2SeqEncoder(
            vocab_chars_size,
            ModelConfig.SEQ_EMBED_DIM,
            ModelConfig.SEQ_HIDDEN_DIM,
            num_classes,
        )
        self.decoder = Seq2SeqDecoder(
            vocab_chars_size, ModelConfig.SEQ_EMBED_DIM, ModelConfig.SEQ_HIDDEN_DIM
        )
        self.vocab_size = vocab_chars_size

    def forward(self, src, tgt, class_ids, teacher_forcing_ratio=0.5):
        # src: (B, T_src)
        # tgt: (B, T_tgt)
        # class_ids: (B)

        batch_size = src.shape[0]
        max_len = tgt.shape[1]

        encoder_outputs, hidden = self.encoder(src, class_ids)

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(src.device)

        # First input is SOS token (assumed at index 0 of tgt, which is handled by dataset)
        input_step = tgt[:, 0]

        for t in range(1, max_len):
            output, hidden = self.decoder(input_step, hidden, encoder_outputs)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_step = tgt[:, t] if teacher_force else top1

        return outputs

    def predict(
        self, src, class_ids, max_len=DataConfig.MAX_TOKEN_LEN, sos_idx=2, eos_idx=3
    ):
        # Greedy decoding for inference
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            encoder_outputs, hidden = self.encoder(src, class_ids)

            finished = torch.zeros(batch_size, dtype=torch.bool).to(src.device)
            outputs = []

            # Start with SOS
            input_step = torch.full((batch_size,), sos_idx, dtype=torch.long).to(
                src.device
            )

            for _ in range(max_len):
                output, hidden = self.decoder(input_step, hidden, encoder_outputs)
                top1 = output.argmax(1)
                outputs.append(top1)

                # Update finished status
                finished = finished | (top1 == eos_idx)
                if finished.all():
                    break

                input_step = top1

            return torch.stack(outputs, dim=1)


# ==========================================
# 2. Training Functions
# ==========================================


def train_tagger_model():
    print("\n=== Training Multi-Granularity Tagger ===")
    set_seed(TrainingConfig.SEED)
    device = torch.device(TrainingConfig.DEVICE)

    # 1. Load Data & Vocab
    vocab_words, vocab_chars, vocab_classes = build_vocabularies()

    df_train = load_dataset_raw("train")
    df_val = load_dataset_raw("val")

    train_dataset = TaggerDataset(df_train, vocab_words, vocab_chars, vocab_classes)
    val_dataset = TaggerDataset(df_val, vocab_words, vocab_chars, vocab_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.TAGGER_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.TAGGER_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )

    # 2. Compute Class Weights
    if TrainingConfig.USE_CLASS_WEIGHTS:
        counts = df_train["class"].value_counts().sort_index()
        # Ensure alignment with vocab
        class_indices = [vocab_classes.stoi[c] for c in counts.index]
        weights = np.sqrt(len(df_train) / counts.values)
        # Normalize
        weights = weights / weights.mean()

        # Map back to tensor in correct order
        weight_tensor = torch.ones(len(vocab_classes))
        for idx, w in zip(class_indices, weights):
            weight_tensor[idx] = w
        weight_tensor = weight_tensor.to(device)
    else:
        weight_tensor = None

    # 3. Initialize Model
    model = MultiGranularityTagger(
        len(vocab_words), len(vocab_chars), len(vocab_classes)
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=TrainingConfig.TAGGER_LR,
        weight_decay=TrainingConfig.TAGGER_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=TrainingConfig.SCHEDULER_FACTOR,
        patience=TrainingConfig.SCHEDULER_PATIENCE,
        min_lr=TrainingConfig.SCHEDULER_MIN_LR,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # 4. Training Loop
    best_acc = 0.0

    for epoch in range(TrainingConfig.TAGGER_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            targets = batch["class_ids"].to(device)

            optimizer.zero_grad()
            logits = model(word_ids, char_ids)
            loss = criterion(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), TrainingConfig.TAGGER_GRAD_CLIP
            )
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                word_ids = batch["word_ids"].to(device)
                char_ids = batch["char_ids"].to(device)
                targets = batch["class_ids"].to(device)

                logits = model(word_ids, char_ids)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{TrainingConfig.TAGGER_EPOCHS} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ProjectConfig.TAGGER_MODEL_PATH)
            print("  -> Saved Best Tagger Model")

    return model


def train_seq2seq_model():
    print("\n=== Training Seq2Seq Fallback Model ===")
    set_seed(TrainingConfig.SEED)
    device = torch.device(TrainingConfig.DEVICE)

    # 1. Load Data & Vocab
    _, vocab_chars, vocab_classes = build_vocabularies()

    df_train = load_dataset_raw("train")
    df_val = load_dataset_raw("val")

    train_dataset = Seq2SeqDataset(df_train, vocab_chars, vocab_classes)
    val_dataset = Seq2SeqDataset(df_val, vocab_chars, vocab_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.SEQ_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_seq2seq,
        num_workers=DataConfig.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.SEQ_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_seq2seq,
        num_workers=DataConfig.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = Seq2SeqNormalizer(len(vocab_chars), len(vocab_classes)).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=TrainingConfig.SEQ_LR,
        weight_decay=TrainingConfig.SEQ_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD

    # 3. Training Loop
    best_val_loss = float("inf")

    for epoch in range(TrainingConfig.SEQ_EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            src = batch["src_char_ids"].to(device)
            tgt = batch["tgt_char_ids"].to(device)
            class_ids = batch["class_ids"].to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(src, tgt, class_ids, TrainingConfig.TEACHER_FORCING_RATIO)

            # Reshape for loss: (B * T, Vocab) vs (B * T)
            # tgt output excludes SOS at index 0? No, decoder generates from index 1.
            # outputs shape: (B, MaxLen, Vocab). tgt shape: (B, MaxLen).
            # We compare outputs[:, 1:] with tgt[:, 1:] (skip SOS in target)
            # Actually, model loop runs 1 to max_len. outputs[:, 0] is 0.

            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            targets = tgt[:, 1:].reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.SEQ_GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                src = batch["src_char_ids"].to(device)
                tgt = batch["tgt_char_ids"].to(device)
                class_ids = batch["class_ids"].to(device)

                outputs = model(src, tgt, class_ids, teacher_forcing_ratio=0.0)

                output_dim = outputs.shape[-1]
                outputs = outputs[:, 1:].reshape(-1, output_dim)
                targets = tgt[:, 1:].reshape(-1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{TrainingConfig.SEQ_EPOCHS} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ProjectConfig.SEQ2SEQ_MODEL_PATH)
            print("  -> Saved Best Seq2Seq Model")

    return model


# ==========================================
# 3. Inference & Submission
# ==========================================


def generate_submission():
    print("\n=== Generating Submission ===")
    device = torch.device(TrainingConfig.DEVICE)

    # 1. Load Resources
    vocab_words, vocab_chars, vocab_classes = build_vocabularies()
    kb_dict = build_knowledge_base()

    # 2. Load Models
    tagger = MultiGranularityTagger(
        len(vocab_words), len(vocab_chars), len(vocab_classes)
    )
    tagger.load_state_dict(
        torch.load(ProjectConfig.TAGGER_MODEL_PATH, map_location=device)
    )
    tagger.to(device)
    tagger.eval()

    seq2seq = Seq2SeqNormalizer(len(vocab_chars), len(vocab_classes))
    seq2seq.load_state_dict(
        torch.load(ProjectConfig.SEQ2SEQ_MODEL_PATH, map_location=device)
    )
    seq2seq.to(device)
    seq2seq.eval()

    # 3. Load Test Data
    df_test = load_dataset_raw("test")
    test_dataset = TaggerDataset(
        df_test, vocab_words, vocab_chars, vocab_classes, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainingConfig.TAGGER_BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )

    results = []
    plain_idx = vocab_classes.stoi.get("PLAIN", -1)

    sos_idx = vocab_chars.stoi[DataConfig.SOS_TOKEN]
    eos_idx = vocab_chars.stoi[DataConfig.EOS_TOKEN]

    print("Predicting...")
    with torch.no_grad():
        for batch in test_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            raw_texts = batch["raw_texts"]
            ids = batch["ids"]

            # Step A: Predict Class
            logits = tagger(word_ids, char_ids)
            pred_class_indices = logits.argmax(dim=1).cpu().numpy()

            # Process Batch
            for i, raw_text in enumerate(raw_texts):
                cls_idx = pred_class_indices[i]
                cls_name = vocab_classes.lookup_token(cls_idx)
                row_id = ids[i]

                normalized_text = raw_text  # Default copy

                # Tier 1: Knowledge Base
                kb_key = (raw_text, cls_name)
                if kb_key in kb_dict:
                    normalized_text = kb_dict[kb_key]

                # Tier 2: Seq2Seq Fallback
                # Only if not PLAIN and not in KB (implied by reaching here if we structured if/else)
                # But we just checked KB. If found, we are done.
                # If not found:
                elif cls_name != "PLAIN" and cls_name != "PUNCT":
                    # Prepare input for Seq2Seq
                    # We need single sample batch
                    src_seq = batch["char_ids"][i].unsqueeze(0).to(device)
                    cls_tensor = torch.tensor([cls_idx], dtype=torch.long).to(device)

                    pred_seq = seq2seq.predict(
                        src_seq, cls_tensor, sos_idx=sos_idx, eos_idx=eos_idx
                    )

                    # Decode to string
                    pred_indices = pred_seq[0].cpu().numpy()
                    chars = []
                    for idx in pred_indices:
                        if idx == eos_idx:
                            break
                        char = vocab_chars.lookup_token(idx)
                        if char:
                            chars.append(char)

                    normalized_text = "".join(chars)

                # Format for CSV: Quote if contains special chars? Pandas handles this.
                results.append({"id": row_id, "after": normalized_text})

    # 4. Save
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(ProjectConfig.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {ProjectConfig.SUBMISSION_PATH}")


def run_experiment():
    # Ensure working directory exists
    os.makedirs(ProjectConfig.BASE_DIR, exist_ok=True)

    # Train
    train_tagger_model()
    train_seq2seq_model()

    # Inference
    generate_submission()
