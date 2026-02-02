import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from library.config import Config


class Encoder(nn.Module):
    """
    Encoder module using GRU.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=Config.PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Attention(nn.Module):
    """
    Bahdanau Attention mechanism.
    """

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, dec_hid_dim]
        # encoder_outputs: [batch size, src len, enc_hid_dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    Decoder module using GRU with Attention.
    """

    def __init__(
        self,
        output_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        n_layers,
        dropout,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=Config.PAD_IDX)
        self.rnn = nn.GRU(
            (enc_hid_dim) + emb_dim,
            dec_hid_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear((enc_hid_dim) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch size]
        # hidden: [n_layers, batch size, dec_hid_dim]
        # encoder_outputs: [batch size, src len, enc_hid_dim]

        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        # Calculate attention weights using the last layer of hidden state
        # hidden[-1] is [batch size, dec_hid_dim]
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)

        weighted = torch.bmm(a, encoder_outputs)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model wrapping Encoder, Attention Decoder.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs


class NeuralSolver:
    """
    High-level wrapper for training and inference of the Seq2Seq model.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device(Config.DEVICE)

        input_dim = len(tokenizer)
        output_dim = len(tokenizer)

        # Initialize Attention
        attn = Attention(Config.HIDDEN_DIM, Config.HIDDEN_DIM)

        encoder = Encoder(
            input_dim,
            Config.EMBED_DIM,
            Config.HIDDEN_DIM,
            Config.NUM_LAYERS,
            Config.DROPOUT,
        )

        decoder = Decoder(
            output_dim,
            Config.EMBED_DIM,
            Config.HIDDEN_DIM,
            Config.HIDDEN_DIM,
            Config.NUM_LAYERS,
            Config.DROPOUT,
            attn,
        )

        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

    def train(self, train_loader, val_loader):
        """
        Trains the model with Early Stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Training Loop
            self.model.train()
            epoch_loss = 0

            for batch in train_loader:
                src = batch["input_ids"].to(self.device)
                trg = batch["target_ids"].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(src, trg, Config.TEACHER_FORCING_RATIO)

                # Reshape for loss calculation
                # output: [batch_size, trg_len, output_dim] -> [(batch_size * trg_len) - 1, output_dim]
                # trg: [batch_size, trg_len] -> [(batch_size * trg_len) - 1]
                # We ignore the 0th index (SOS) for loss calculation usually,
                # or the model output at t corresponds to prediction for t.
                # In the loop above, outputs[:, t] is prediction for trg[:, t].
                # So we slice from 1 to end.

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.CLIP_GRAD
                )
                self.optimizer.step()

                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)

            # Validation Loop
            val_loss = self.evaluate(val_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {train_loss}")
            print(f"\t Val. Loss: {val_loss}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"\tSaved best model to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(
                    f"\tNo improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, loader):
        """
        Evaluates the model on a dataset.
        """
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in loader:
                src = batch["input_ids"].to(self.device)
                trg = batch["target_ids"].to(self.device)

                # Turn off teacher forcing for validation
                output = self.model(src, trg, 0)

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(loader)

    def predict(self, loader):
        """
        Generates predictions for the given loader using greedy decoding.

        Returns:
            list: List of predicted strings.
        """
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            self.model.eval()
        else:
            print("Warning: No checkpoint found. Predicting with current weights.")

        predictions = []

        with torch.no_grad():
            for batch in loader:
                src = batch["input_ids"].to(self.device)
                batch_size = src.shape[0]

                # Encode
                encoder_outputs, hidden = self.model.encoder(src)

                input_token = torch.tensor([Config.SOS_IDX] * batch_size).to(
                    self.device
                )

                decoded_indices = torch.zeros(
                    batch_size, Config.MAX_SEQ_LEN, dtype=torch.long
                ).to(self.device)

                for t in range(Config.MAX_SEQ_LEN):
                    output, hidden = self.model.decoder(
                        input_token, hidden, encoder_outputs
                    )
                    pred_token = output.argmax(1)

                    decoded_indices[:, t] = pred_token
                    input_token = pred_token

                decoded_indices = decoded_indices.cpu().tolist()
                for seq in decoded_indices:
                    pred_str = self.tokenizer.decode(seq, remove_special_tokens=True)
                    predictions.append(pred_str)

        return predictions
