# models/transformer_model.py
import torch
import torch.nn as nn
import config

class PulseEmbeddingTransformer(nn.Module):
    """
    Transformer-based model that embeds sequences of radar pulses into a metric space.
    """
    def __init__(self, d_model=config.D_MODEL, n_heads=config.NUM_HEADS, 
                 num_layers=config.NUM_LAYERS, dim_ff=config.DIM_FEEDFORWARD, 
                 dropout=config.DROPOUT, embedding_dim=config.EMBEDDING_DIM):
        super(PulseEmbeddingTransformer, self).__init__()
        # Input projection: 5 -> d_model
        self.input_linear = nn.Linear(5, d_model)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                  dim_feedforward=dim_ff, dropout=dropout,
                                                  batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output projection: d_model -> embedding_dim
        self.output_linear = nn.Linear(d_model, embedding_dim)
        # We can include a layer normalization on output if needed (optional)
        # self.output_norm = nn.functional.normalize  # (using F.normalize in forward instead)

    def forward(self, x, key_padding_mask=None):
        """
        Forward pass for the model.
        :param x: Tensor of shape (batch, seq_len, 5) containing input features.
        :param key_padding_mask: Boolean mask of shape (batch, seq_len) for padded positions (True for padded).
        :return: Tensor of shape (batch, seq_len, embedding_dim) with output embeddings.
        """
        # Project input features to d_model
        x_proj = self.input_linear(x)              # (B, L, d_model)
        # (No positional encoding added, as per methodology&#8203;:contentReference[oaicite:26]{index=26})
        # Pass through Transformer encoder
        # key_padding_mask tells the encoder which positions are padding (to ignore in self-attention)
        enc_out = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)
        # Project to embedding space
        embed_out = self.output_linear(enc_out)    # (B, L, embedding_dim)
        # Optionally, normalize embeddings to unit length (can stabilize metric learning)
        # embed_out = nn.functional.normalize(embed_out, p=2, dim=-1)
        return embed_out
