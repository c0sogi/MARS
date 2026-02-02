import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random
from library.config import Config


class Encoder(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(Encoder, self).__init__()
        # Create model without classifier and global pooling to keep spatial features
        self.cnn = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

    def forward(self, x):
        # x: (batch_size, 3, image_size, image_size)
        features = self.cnn(x)
        # features: (batch_size, encoder_dim, H/32, W/32) for efficientnet_b0

        # Permute to (batch_size, H*W, encoder_dim) for attention
        batch_size, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, h * w, c)
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        # Additive attention: tanh(W1*enc + W2*dec)
        # We unsqueeze att2 to broadcast over num_pixels
        att = self.full_att(
            self.relu(att1 + att2.unsqueeze(1))
        )  # (batch_size, num_pixels, 1)

        alpha = self.softmax(att)  # (batch_size, num_pixels, 1)

        # Calculate context vector
        context = torch.sum(alpha * encoder_out, dim=1)  # (batch_size, encoder_dim)

        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=1280,
        dropout=0.5,
    ):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def init_hidden_state(self, encoder_out):
        # Initialize hidden state and cell state using the mean of encoder features
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, x, hidden, encoder_out):
        # x: input token index (batch_size,)
        # hidden: (h, c) from previous step
        # encoder_out: spatial features from encoder

        embed = self.embedding(x)  # (batch_size, embed_dim)

        h, c = hidden
        context, alpha = self.attention(encoder_out, h)

        # Gating mechanism (optional in standard Show Attend Tell, but often useful)
        # gate = self.sigmoid(self.f_beta(h))
        # gated_context = gate * context

        # Concatenate embedding and context
        lstm_input = torch.cat((embed, context), dim=1)

        h, c = self.lstm(lstm_input, (h, c))

        output = self.fc(self.dropout_layer(h))

        return output, (h, c), alpha


class Seq2Seq(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_len = config.max_len
        self.device = config.device

        self.encoder = Encoder(model_name=config.encoder_name, pretrained=True)

        self.decoder = DecoderWithAttention(
            attention_dim=config.attention_dim,
            embed_dim=config.embed_dim,
            decoder_dim=config.decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=config.encoder_dim,
            dropout=config.dropout,
        )

    def forward(self, images, text=None, teacher_forcing_ratio=0.5):
        # images: (batch_size, 3, H, W)
        # text: (batch_size, max_seq_len) - Ground truth sequences (optional for inference)

        batch_size = images.size(0)

        # Encode images
        encoder_out = self.encoder(images)  # (batch_size, num_pixels, encoder_dim)

        # Initialize decoder hidden state
        hidden = self.decoder.init_hidden_state(encoder_out)

        # Prepare tensor for predictions
        # If training, we predict up to text length. If inference, up to max_len.
        target_len = text.size(1) if text is not None else self.max_len
        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(self.device)

        # First input is SOS token (assumed index 1 based on Tokenizer class)
        # We need to ensure we use the correct SOS index.
        # In Tokenizer: SOS_IDX = 1
        input_token = torch.tensor([1] * batch_size, device=self.device)

        for t in range(1, target_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_out)
            outputs[:, t, :] = output

            # Determine next input
            if text is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                input_token = text[:, t]
            else:
                # Use prediction
                input_token = output.argmax(1)

            # During inference, we could check for EOS here to break early,
            # but for batch processing usually we run fixed length or mask later.

        return outputs

    def predict(self, images):
        """
        Inference method for generating sequences greedily.
        """
        self.eval()
        with torch.no_grad():
            batch_size = images.size(0)
            encoder_out = self.encoder(images)
            hidden = self.decoder.init_hidden_state(encoder_out)

            # SOS token index = 1
            input_token = torch.tensor([1] * batch_size, device=self.device)

            predictions = []
            # We collect indices
            preds_tensor = torch.zeros(batch_size, self.max_len, dtype=torch.long).to(
                self.device
            )

            # Set SOS
            preds_tensor[:, 0] = 1

            for t in range(1, self.max_len):
                output, hidden, _ = self.decoder(input_token, hidden, encoder_out)
                input_token = output.argmax(1)
                preds_tensor[:, t] = input_token

                # Optimization: If all batches predicted EOS (index 2), we could break.
                # But handling mixed states is complex without masking.

        return preds_tensor
