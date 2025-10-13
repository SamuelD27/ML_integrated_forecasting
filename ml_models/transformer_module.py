from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Learnable and sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000,
                learnable: bool = True, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of model
            max_len: Maximum sequence length
            learnable: Whether to use learnable embeddings
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.learnable = learnable
        self.dropout = nn.Dropout(dropout)

        if learnable:
            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                               (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Add batch dimension

            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        if self.learnable:
            seq_len = x.size(1)
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism with optional masking.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Dimension of model
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
               return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections in batch from d_model => h x d_k
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add head and query dimensions
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Attended values
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Final linear projection
        output = self.W_o(context)

        if return_attention:
            return output, attention.mean(dim=1)  # Average attention across heads
        else:
            return output, None


class DualAttention(nn.Module):
    """
    Dual attention mechanism with both masked and standard self-attention.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize dual attention.

        Args:
            d_model: Dimension of model
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.masked_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.standard_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None,
               padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through dual attention.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            causal_mask: Causal mask for masked attention
            padding_mask: Padding mask

        Returns:
            Output tensor with dual attention applied
        """
        # Masked self-attention (causal)
        masked_out, _ = self.masked_attention(x, causal_mask)

        # Standard self-attention
        standard_out, _ = self.standard_attention(x, padding_mask)

        # Gated combination
        gate_value = self.gate(torch.cat([masked_out, standard_out], dim=-1))
        combined = gate_value * masked_out + (1 - gate_value) * standard_out

        # Residual connection and layer norm
        output = self.layer_norm(x + combined)

        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor
        """
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    """

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048,
                dropout: float = 0.1, use_dual_attention: bool = False):
        """
        Initialize Transformer encoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_dual_attention: Whether to use dual attention
        """
        super().__init__()

        if use_dual_attention:
            self.attention = DualAttention(d_model, n_heads, dropout)
        else:
            self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_dual_attention = use_dual_attention

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        if self.use_dual_attention:
            attn_out = self.attention(x, padding_mask=mask)
        else:
            attn_out, _ = self.attention(x, mask)
            attn_out = self.layer_norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.feed_forward(attn_out)
        output = self.layer_norm2(attn_out + self.dropout(ff_out))

        return output


class TransformerEncoder(nn.Module):
    """
    Custom Transformer encoder for financial time series.

    Architecture:
    - Input projection
    - Learnable positional embeddings
    - Stack of Transformer encoder layers
    - Multi-head self-attention (8 heads)
    - Dual attention mechanism option
    - Feed-forward network with GELU
    """

    def __init__(self, input_dim: int, d_model: int = 512, n_heads: int = 8,
                num_layers: int = 4, d_ff: int = 2048, dropout: float = 0.1,
                max_seq_len: int = 1000, use_dual_attention: bool = True,
                output_dim: Optional[int] = None):
        """
        Initialize Transformer encoder.

        Args:
            input_dim: Number of input features
            d_model: Dimension of model
            n_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_dual_attention: Whether to use dual attention
            output_dim: Output dimension (default: d_model)
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_len, learnable=True, dropout=dropout
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, use_dual_attention)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_dim = output_dim or d_model
        if self.output_dim != d_model:
            self.output_projection = nn.Linear(d_model, self.output_dim)
        else:
            self.output_projection = nn.Identity()

        # Global pooling for sequence representation
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False)
        )

        logger.info(
            f"Initialized Transformer: layers={num_layers}, "
            f"d_model={d_model}, heads={n_heads}, "
            f"dual_attention={use_dual_attention}"
        )

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive attention.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask tensor
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
               return_all_layers: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer encoder.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Optional padding mask
            return_all_layers: Whether to return outputs from all layers

        Returns:
            Tuple of (output, attention_weights)
            - output: Encoded representation (batch_size, output_dim)
            - attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape

        # Input projection and scaling
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Store outputs from each layer if requested
        all_layer_outputs = [] if return_all_layers else None

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            if return_all_layers:
                all_layer_outputs.append(x)

        # Global pooling for sequence representation
        # Method 1: Weighted average using learned attention
        attention_scores = self.global_pool(x).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        weighted_output = (x * attention_weights).sum(dim=1)  # (batch_size, d_model)

        # Apply output projection
        output = self.output_projection(weighted_output)

        if return_all_layers:
            return output, all_layer_outputs
        else:
            return output, attention_weights.squeeze(-1)


class CrossAttentionTransformer(nn.Module):
    """
    Transformer with cross-attention for multi-stock modeling.
    """

    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 8,
                num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize cross-attention Transformer.

        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of encoder layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, learnable=True)

        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention layer for multi-stock relationships
        self.cross_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.output_dim = d_model

    def forward(self, x: torch.Tensor, other_stocks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional cross-stock attention.

        Args:
            x: Primary stock tensor (batch_size, seq_len, input_dim)
            other_stocks: Other stocks tensor (batch_size, n_stocks, seq_len, input_dim)

        Returns:
            Output tensor (batch_size, d_model)
        """
        # Project and encode primary stock
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Self-attention
        for layer in self.self_attention_layers:
            x = layer(x)

        # Cross-attention with other stocks if provided
        if other_stocks is not None:
            batch_size, n_stocks, seq_len, _ = other_stocks.shape

            # Process other stocks
            other_stocks = other_stocks.reshape(-1, seq_len, other_stocks.shape[-1])
            other_stocks = self.input_projection(other_stocks)
            other_stocks = self.positional_encoding(other_stocks)

            # Apply self-attention to other stocks
            for layer in self.self_attention_layers:
                other_stocks = layer(other_stocks)

            # Reshape back
            other_stocks = other_stocks.reshape(batch_size, n_stocks, seq_len, -1)

            # Pool other stocks
            other_stocks_pooled = other_stocks.mean(dim=2)  # (batch_size, n_stocks, d_model)

            # Cross-attention: x attends to other stocks
            x_pooled = x.mean(dim=1, keepdim=True)  # (batch_size, 1, d_model)
            cross_out, _ = self.cross_attention(
                torch.cat([x_pooled, other_stocks_pooled], dim=1)
            )
            x = self.layer_norm(x.mean(dim=1) + cross_out[:, 0, :])
        else:
            x = x.mean(dim=1)  # Simple pooling

        return x