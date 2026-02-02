import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class Encoder(nn.Module):
    """
    Asymmetric ResNet-18 Encoder + BiLSTM.
    Compresses image height to 1 while preserving width to create a sequence of features.
    """

    def __init__(self):
        super().__init__()
        # Load ResNet18 without pretrained weights (internet access restricted)
        resnet = models.resnet18(pretrained=False)

        # Modify strides to preserve width resolution (Scan-based approach)
        # Standard ResNet reduces H and W by 2 at the start of layer 2, 3, 4.
        # We change stride to (2, 1) to reduce H by 2 but keep W (stride 1).

        # Layer 2
        resnet.layer2[0].conv1.stride = (2, 1)
        if resnet.layer2[0].downsample is not None:
            resnet.layer2[0].downsample[0].stride = (2, 1)

        # Layer 3
        resnet.layer3[0].conv1.stride = (2, 1)
        if resnet.layer3[0].downsample is not None:
            resnet.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4
        resnet.layer4[0].conv1.stride = (2, 1)
        if resnet.layer4[0].downsample is not None:
            resnet.layer4[0].downsample[0].stride = (2, 1)

        # Extract layers up to layer4
        self.resnet_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Vertical Max Pooling to collapse height dimension to 1
        # Input to this will be (B, 512, H', W'). Output will be (B, 512, 1, W')
        self.vertical_pool = nn.AdaptiveMaxPool2d((1, None))

        # BiLSTM for sequence contextualization
        # Input size is 512 (ResNet18 output channels)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=Config.ENCODER_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, images):
        # images: (B, 3, 192, 512)

        # 1. CNN Feature Extraction
        features = self.resnet_layers(images)
        # Expected shape after asymmetric strides: (B, 512, 6, 128)

        # 2. Vertical Collapse
        pooled = self.vertical_pool(features)
        # Shape: (B, 512, 1, 128)

        # Squeeze height dimension
        pooled = pooled.squeeze(2)
        # Shape: (B, 512, 128)

        # Permute to (Batch, Seq_Len, Features) for LSTM
        pooled = pooled.permute(0, 2, 1)
        # Shape: (B, 128, 512)

        # 3. Sequence Modeling
        # output shape: (B, 128, 2 * ENCODER_HIDDEN_DIM)
        output, (hidden, cell) = self.lstm(pooled)

        return output


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention.
    """

    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        # Linear layer to transform concatenated hidden states
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, attn_dim)
        # Linear layer to score the energy
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (B, dec_hid_dim) - Decoder hidden state from previous step
        # encoder_outputs: (B, seq_len, enc_hid_dim * 2) - All encoder outputs

        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times to align with encoder outputs
        # (B, seq_len, dec_hid_dim)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate Energy
        # tanh(W * [hidden; encoder_outputs] + b)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # (B, seq_len, attn_dim)

        # Calculate Attention Scores
        attention = self.v(energy).squeeze(2)
        # (B, seq_len)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    LSTM Decoder with Attention.
    """

    def __init__(
        self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Input to LSTM is concatenation of Embedding and Context Vector (Weighted Encoder Outputs)
        self.rnn = nn.LSTMCell((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        # Output layer
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: (B,) - Index of previous character
        # hidden, cell: (B, dec_hid_dim) - Previous states
        # encoder_outputs: (B, seq_len, enc_hid_dim * 2)

        input_token = input_token.unsqueeze(1)  # (B, 1)

        # Embedding
        embedded = self.dropout(self.embedding(input_token)).squeeze(1)  # (B, emb_dim)

        # Calculate Attention Weights
        a = self.attention(hidden, encoder_outputs)  # (B, seq_len)
        a = a.unsqueeze(1)  # (B, 1, seq_len)

        # Apply Attention (Context Vector)
        # Weighted sum of encoder outputs
        weighted = torch.bmm(a, encoder_outputs).squeeze(1)  # (B, enc_hid_dim * 2)

        # LSTM Step
        rnn_input = torch.cat((embedded, weighted), dim=1)
        hidden, cell = self.rnn(rnn_input, (hidden, cell))

        # Prediction
        # Concatenate embedded input, context vector, and new hidden state
        prediction = self.fc_out(torch.cat((embedded, weighted, hidden), dim=1))

        return prediction, hidden, cell, a.squeeze(1)


class Seq2Seq(nn.Module):
    """
    Wrapper class for Encoder-Decoder architecture.
    Handles forward pass (Training/Teacher Forcing) and Inference (Greedy Decoding).
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.attention = Attention(
            Config.ENCODER_HIDDEN_DIM, Config.DECODER_HIDDEN_DIM, Config.ATTENTION_DIM
        )
        self.decoder = Decoder(
            vocab_size,
            Config.EMBEDDING_DIM,
            Config.ENCODER_HIDDEN_DIM,
            Config.DECODER_HIDDEN_DIM,
            Config.DROPOUT,
            self.attention,
        )
        self.vocab_size = vocab_size

        # SOS Token Index (Assumed to be 1 based on Tokenizer implementation)
        self.SOS_IDX = 1

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training or validation.

        Args:
            images: (B, C, H, W)
            targets: (B, max_len) - Ground truth sequences (indices). If None, runs inference.
            teacher_forcing_ratio: Probability of using ground truth as input for next step.

        Returns:
            outputs: (B, max_len, vocab_size) - Logits for each time step.
        """
        batch_size = images.shape[0]
        max_len = Config.MAX_LENGTH if targets is None else targets.shape[1]

        # Encode Images
        encoder_outputs = self.encoder(images)
        # encoder_outputs: (B, 128, 512)

        # Initialize Decoder State
        # Use mean of encoder outputs to initialize hidden state
        mean_encoder_out = torch.mean(encoder_outputs, dim=1)

        # Project if dimensions differed, but here they match (512)
        hidden = mean_encoder_out
        cell = torch.zeros_like(hidden)

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(Config.DEVICE)

        # First input is <sos>
        input_token = torch.tensor([self.SOS_IDX] * batch_size, dtype=torch.long).to(
            Config.DEVICE
        )

        # Decoding Loop
        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )

            # Store prediction
            outputs[:, t] = output

            # Decide next input
            top1 = output.argmax(1)

            # Teacher Forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = top1

        return outputs

    def predict(self, images):
        """
        Inference method for generating predictions.
        Uses greedy decoding (teacher_forcing_ratio=0.0).
        """
        self.eval()
        with torch.no_grad():
            # Run forward pass without targets
            outputs = self.forward(images, targets=None, teacher_forcing_ratio=0.0)

            # Get indices of highest probability
            predictions = outputs.argmax(dim=2)  # (B, max_len)

        return predictions
