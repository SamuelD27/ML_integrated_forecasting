# Architecture Templates for Financial ML

Complete, production-ready PyTorch implementations with proper initialization and validation.

---

## Template 1: Temporal Fusion Transformer (TFT)

### Overview
Multi-horizon forecasting with interpretability through variable selection and temporal attention. Outputs quantile predictions for uncertainty estimation.

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class VariableSelectionNetwork(nn.Module):
    """Learn which features matter most."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size

        # Feature-wise linear transformations
        self.feature_transform = nn.Linear(input_size, hidden_size)

        # Variable selection weights (learn importance)
        self.variable_weights = nn.Linear(hidden_size, input_size)

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.feature_transform.weight)
        nn.init.zeros_(self.feature_transform.bias)
        nn.init.xavier_uniform_(self.variable_weights.weight)
        nn.init.zeros_(self.variable_weights.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            selected: [batch, seq_len, hidden_size]
            importance: [batch, seq_len, input_size]
        """
        # Validate input
        assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"

        # Transform features
        transformed = self.feature_transform(x)  # [batch, seq, hidden]
        transformed = F.elu(transformed)
        transformed = self.dropout(transformed)

        # Compute importance weights
        importance = self.variable_weights(transformed)  # [batch, seq, input_size]
        importance = torch.softmax(importance, dim=-1)

        # Validate importance sums to 1
        assert torch.allclose(importance.sum(dim=-1), torch.ones_like(importance.sum(dim=-1)), atol=1e-5), \
            "Variable importance doesn't sum to 1"

        # Apply importance weights
        selected = torch.einsum('bsi,bsh->bsh', importance, transformed)

        return selected, importance


class TemporalFusionTransformer(nn.Module):
    """
    TFT for multi-horizon financial forecasting.

    Architecture:
    1. Variable selection (learn feature importance)
    2. LSTM encoder (capture temporal patterns)
    3. Multi-head attention (learn which past timesteps matter)
    4. Quantile regression heads (uncertainty estimation)
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Variable selection
        self.var_selection = VariableSelectionNetwork(input_size, hidden_dim, dropout)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm (stabilizes training)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Quantile regression heads (10%, 25%, 50%, 75%, 90%)
        self.quantile_heads = nn.ModuleDict({
            'q10': nn.Linear(hidden_dim, 1),
            'q25': nn.Linear(hidden_dim, 1),
            'q50': nn.Linear(hidden_dim, 1),
            'q75': nn.Linear(hidden_dim, 1),
            'q90': nn.Linear(hidden_dim, 1),
        })

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all weights for stable training."""

        # LSTM initialization
        for name, param in self.lstm_encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1.0
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # Quantile heads initialization
        for head in self.quantile_heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_size]

        Returns:
            Dict with quantile predictions and feature importance
        """
        # INPUT VALIDATION
        assert x.ndim == 3, f"Expected [batch, seq, features], got {x.ndim}D"
        assert x.shape[2] == self.input_size, \
            f"Expected {self.input_size} features, got {x.shape[2]}"
        assert torch.isfinite(x).all(), "Input contains NaN or Inf"

        batch_size, seq_len, _ = x.shape

        # 1. Variable selection
        selected, feature_importance = self.var_selection(x)

        # 2. LSTM encoding
        encoded, (h, c) = self.lstm_encoder(selected)

        # 3. Self-attention
        attended, attention_weights = self.attention(encoded, encoded, encoded)

        # Residual connection + layer norm
        attended = self.layer_norm(attended + encoded)

        # 4. Quantile predictions (use last timestep)
        last_hidden = attended[:, -1, :]  # [batch, hidden_dim]

        quantiles = {
            name: head(last_hidden).squeeze(-1)  # [batch]
            for name, head in self.quantile_heads.items()
        }

        # OUTPUT VALIDATION
        for name, pred in quantiles.items():
            assert pred.shape == (batch_size,), \
                f"Quantile {name} shape {pred.shape} != ({batch_size},)"
            assert torch.isfinite(pred).all(), \
                f"Quantile {name} contains NaN or Inf"

        # QUANTILE ORDERING VALIDATION (only during eval)
        if not self.training:
            assert (quantiles['q10'] <= quantiles['q25']).all(), "q10 > q25 violation"
            assert (quantiles['q25'] <= quantiles['q50']).all(), "q25 > q50 violation"
            assert (quantiles['q50'] <= quantiles['q75']).all(), "q50 > q75 violation"
            assert (quantiles['q75'] <= quantiles['q90']).all(), "q75 > q90 violation"

        return {
            'quantiles': quantiles,
            'feature_importance': feature_importance,
            'attention_weights': attention_weights
        }


# USAGE EXAMPLE
if __name__ == '__main__':
    # Initialize model
    model = TemporalFusionTransformer(
        input_size=20,      # 20 features
        hidden_dim=128,
        num_heads=4,
        num_lstm_layers=2,
        dropout=0.2
    )

    # Test forward pass
    x = torch.randn(32, 60, 20)  # [batch=32, seq=60, features=20]
    output = model(x)

    print("Quantile predictions:")
    for q, pred in output['quantiles'].items():
        print(f"  {q}: {pred.shape}")  # [32]

    print(f"Feature importance: {output['feature_importance'].shape}")  # [32, 60, 20]
```

---

## Template 2: N-BEATS

### Overview
Pure architecture for univariate time series forecasting using stacked blocks with residual connections.

### Complete Implementation

```python
class NBeatsBlock(nn.Module):
    """Single N-BEATS block with basis expansion."""

    def __init__(self, input_size: int, theta_size: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.input_size = input_size
        self.theta_size = theta_size

        # FC stack
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.fc_stack = nn.Sequential(*layers)

        # Basis function parameters
        self.theta_backcast = nn.Linear(hidden_size, theta_size)
        self.theta_forecast = nn.Linear(hidden_size, theta_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_size]
        Returns:
            backcast: [batch, input_size]
            forecast: [batch, theta_size]
        """
        assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
        assert x.shape[1] == self.input_size

        # Process through FC stack
        h = self.fc_stack(x)

        # Generate basis parameters
        theta_b = self.theta_backcast(h)
        theta_f = self.theta_forecast(h)

        # For simplicity, use linear basis (can be polynomial/Fourier)
        backcast = theta_b
        forecast = theta_f

        return backcast, forecast


class NBeats(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.

    Stacks multiple blocks, each producing backcast (residual) and forecast.
    Final prediction is sum of all block forecasts.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_size: int = 128,
        num_blocks: int = 4,
        num_layers_per_block: int = 4,
    ):
        super().__init__()

        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_blocks = num_blocks

        # Stack of blocks
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, forecast_size, hidden_size, num_layers_per_block)
            for _ in range(num_blocks)
        ])

        # Blocks already initialize their weights in __init__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size] - Historical values
        Returns:
            forecast: [batch, forecast_size] - Future predictions
        """
        # INPUT VALIDATION
        assert x.ndim == 2, f"Expected 2D [batch, seq], got {x.ndim}D"
        assert x.shape[1] == self.input_size, \
            f"Expected sequence length {self.input_size}, got {x.shape[1]}"
        assert torch.isfinite(x).all(), "Input contains NaN or Inf"

        residual = x
        forecast = torch.zeros(x.shape[0], self.forecast_size, device=x.device)

        # Process through stacked blocks
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residual)

            # Validate block outputs
            assert backcast.shape == residual.shape, \
                f"Block {i} backcast shape mismatch"
            assert block_forecast.shape[1] == self.forecast_size, \
                f"Block {i} forecast size mismatch"

            # Update residual and accumulate forecast
            residual = residual - backcast
            forecast = forecast + block_forecast

        # OUTPUT VALIDATION
        assert forecast.shape == (x.shape[0], self.forecast_size), \
            f"Output shape {forecast.shape} incorrect"
        assert torch.isfinite(forecast).all(), \
            "Forecast contains NaN or Inf"

        return forecast
```

---

## Template 3: Autoformer

### Overview
Transformer with series decomposition and autocorrelation attention for handling seasonality and trends separately.

### Complete Implementation

```python
class SeriesDecomposition(nn.Module):
    """Decompose time series into trend and seasonal components."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        # Moving average for trend extraction
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            trend: [batch, seq_len, features]
            seasonal: [batch, seq_len, features]
        """
        # Transpose for AvgPool1d: [batch, features, seq_len]
        x_perm = x.permute(0, 2, 1)
        trend = self.avg(x_perm).permute(0, 2, 1)

        # Handle boundary effects
        trend = trend[:, :x.shape[1], :]

        seasonal = x - trend

        # VALIDATION
        assert trend.shape == x.shape, "Trend shape mismatch"
        assert seasonal.shape == x.shape, "Seasonal shape mismatch"
        assert torch.isfinite(trend).all(), "Trend contains NaN/Inf"
        assert torch.isfinite(seasonal).all(), "Seasonal contains NaN/Inf"

        return trend, seasonal


class AutoCorrelation(nn.Module):
    """Autocorrelation-based attention mechanism."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: [batch, num_heads, seq, head_dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Autocorrelation: compute in frequency domain (FFT)
        # Simplified: use regular attention for this template
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # VALIDATE ATTENTION WEIGHTS
        assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5), \
            "Attention weights don't sum to 1"

        # Apply attention to values
        output = torch.matmul(attn, V)

        # Reshape back: [batch, seq, hidden_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


class Autoformer(nn.Module):
    """
    Autoformer with series decomposition and autocorrelation attention.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Input embedding
        self.embedding = nn.Linear(input_size, hidden_dim)

        # Series decomposition
        self.decomposition = SeriesDecomposition(kernel_size=25)

        # Attention layers with decomposition
        self.trend_attentions = nn.ModuleList([
            AutoCorrelation(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.seasonal_attentions = nn.ModuleList([
            AutoCorrelation(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Projection heads
        self.trend_projection = nn.Linear(seq_len, pred_len)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

        nn.init.xavier_uniform_(self.trend_projection.weight)
        nn.init.zeros_(self.trend_projection.bias)

        nn.init.xavier_uniform_(self.seasonal_projection.weight)
        nn.init.zeros_(self.seasonal_projection.bias)

        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            output: [batch, pred_len, input_size]
        """
        # INPUT VALIDATION
        assert x.shape[1:] == (self.seq_len, self.input_size), \
            f"Expected shape [batch, {self.seq_len}, {self.input_size}], got {x.shape}"
        assert torch.isfinite(x).all(), "Input contains NaN or Inf"

        # Embed input
        x = self.embedding(x)  # [batch, seq, hidden]

        # Decompose into trend and seasonal
        trend, seasonal = self.decomposition(x)

        # Process through attention layers
        for trend_attn, seasonal_attn in zip(self.trend_attentions, self.seasonal_attentions):
            trend = trend_attn(trend)
            seasonal = seasonal_attn(seasonal)

        # Project to prediction horizon
        # Transpose: [batch, hidden, seq] for linear projection over time
        trend = trend.transpose(1, 2)
        seasonal = seasonal.transpose(1, 2)

        trend_pred = self.trend_projection(trend).transpose(1, 2)  # [batch, pred_len, hidden]
        seasonal_pred = self.seasonal_projection(seasonal).transpose(1, 2)

        # Combine trend and seasonal
        combined = trend_pred + seasonal_pred

        # Project to output dimension
        output = self.output_projection(combined)  # [batch, pred_len, input_size]

        # OUTPUT VALIDATION
        expected_shape = (x.shape[0], self.pred_len, self.input_size)
        assert output.shape == expected_shape, \
            f"Output shape {output.shape} != expected {expected_shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

        return output
```

---

## Template 4: Dynamic Ensemble Weighter

### Overview
Learn to weight multiple trained models based on market regime for robust predictions.

### Complete Implementation

```python
class DynamicEnsembleWeighter(nn.Module):
    """
    Learn model weights conditioned on market regime.

    Weights each model differently depending on current market conditions
    (bull, bear, neutral, crisis) to leverage each model's strengths.
    """

    def __init__(self, num_models: int = 5, num_regimes: int = 4):
        super().__init__()

        self.num_models = num_models
        self.num_regimes = num_regimes

        # Learnable weight matrix: [num_regimes, num_models]
        # Each row: weights for one regime
        self.regime_weights = nn.Parameter(
            torch.randn(num_regimes, num_models)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize with softmax-friendly values."""
        # Initialize close to uniform (small random values)
        nn.init.normal_(self.regime_weights, mean=0.0, std=0.01)

    def forward(
        self,
        model_predictions: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted ensemble based on regime probabilities.

        Args:
            model_predictions: [batch_size, num_models]
            regime_probs: [batch_size, num_regimes] (softmax probabilities)

        Returns:
            ensemble_pred: [batch_size] - weighted predictions
        """
        # INPUT VALIDATION
        assert model_predictions.ndim == 2, "model_predictions must be 2D"
        assert regime_probs.ndim == 2, "regime_probs must be 2D"
        assert model_predictions.shape[1] == self.num_models, \
            f"Expected {self.num_models} models, got {model_predictions.shape[1]}"
        assert regime_probs.shape[1] == self.num_regimes, \
            f"Expected {self.num_regimes} regimes, got {regime_probs.shape[1]}"
        assert model_predictions.shape[0] == regime_probs.shape[0], \
            "Batch sizes must match"

        # VALIDATE REGIME PROBABILITIES
        assert torch.allclose(regime_probs.sum(dim=1), torch.ones(regime_probs.shape[0]), atol=1e-5), \
            f"Regime probs sum to {regime_probs.sum(dim=1).mean():.6f}, not 1.0"
        assert (regime_probs >= 0).all() and (regime_probs <= 1).all(), \
            "Regime probabilities outside [0, 1]"

        # Softmax over models for each regime (ensure weights sum to 1)
        normalized_weights = F.softmax(self.regime_weights, dim=1)  # [regimes, models]

        # Effective weights: regime_probs @ normalized_weights
        # [batch, regimes] @ [regimes, models] = [batch, models]
        effective_weights = torch.matmul(regime_probs, normalized_weights)

        # VALIDATE EFFECTIVE WEIGHTS
        assert torch.allclose(effective_weights.sum(dim=1), torch.ones(effective_weights.shape[0]), atol=1e-5), \
            "Effective weights don't sum to 1"
        assert (effective_weights >= 0).all() and (effective_weights <= 1).all(), \
            "Effective weights outside [0, 1]"

        # Weighted sum of model predictions
        ensemble_pred = (model_predictions * effective_weights).sum(dim=1)

        # OUTPUT VALIDATION
        assert ensemble_pred.shape == (model_predictions.shape[0],), \
            f"Output shape {ensemble_pred.shape} incorrect"
        assert torch.isfinite(ensemble_pred).all(), \
            "Ensemble prediction contains NaN or Inf"

        return ensemble_pred

    def get_regime_weights(self) -> torch.Tensor:
        """Return normalized weights for each regime."""
        return F.softmax(self.regime_weights, dim=1)


# USAGE EXAMPLE
if __name__ == '__main__':
    # 5 models, 4 regimes
    ensemble = DynamicEnsembleWeighter(num_models=5, num_regimes=4)

    # Batch of predictions and regime probabilities
    model_preds = torch.randn(32, 5)  # 32 samples, 5 model predictions
    regime_probs = torch.softmax(torch.randn(32, 4), dim=1)  # [32, 4] normalized

    # Get ensemble prediction
    ensemble_pred = ensemble(model_preds, regime_probs)
    print(f"Ensemble predictions: {ensemble_pred.shape}")  # [32]

    # Inspect learned weights per regime
    weights = ensemble.get_regime_weights()
    print("Learned weights per regime:")
    for i, regime in enumerate(['bull', 'bear', 'neutral', 'crisis']):
        print(f"  {regime}: {weights[i].detach().numpy()}")
```

---

## Common Implementation Patterns

### Pattern: Add Layer Normalization for Deep Networks

```python
class DeepNetwork(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(128, 128))
            layers.append(nn.LayerNorm(128))  # Stabilizes deep networks
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self._initialize_weights()
```

### Pattern: Residual Connections for Very Deep Networks

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        identity = x

        out = F.relu(self.fc1(x))
        out = self.fc2(out)

        # Residual connection
        out = out + identity
        out = self.layer_norm(out)
        out = F.relu(out)

        return out
```

---

## Training Configuration Template

```python
def get_training_config(model: nn.Module, device: str = 'cuda'):
    """Standard training configuration for financial models."""

    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Gradient clipping configuration
    max_grad_norm = 1.0

    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'max_grad_norm': max_grad_norm,
        'device': device
    }


def train_epoch(model, dataloader, config):
    """Training loop with proper gradient clipping."""
    model.train()
    total_loss = 0

    optimizer = config['optimizer']
    max_grad_norm = config['max_grad_norm']
    device = config['device']

    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = compute_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # GRADIENT CLIPPING (MANDATORY)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

---

## Quick Reference

### Initialization Rules

| Layer Type | Initialization | Reason |
|------------|----------------|--------|
| `nn.Linear` (general) | `xavier_uniform_` | Balanced for tanh/sigmoid |
| `nn.Linear` (with ReLU) | `kaiming_normal_` | Accounts for ReLU's 0-killing |
| `nn.Conv1d/2d` (with ReLU) | `kaiming_normal_` | Same as Linear with ReLU |
| `nn.LSTM` weight_ih | `xavier_uniform_` | Input-to-hidden connections |
| `nn.LSTM` weight_hh | `orthogonal_` | Prevents vanishing gradients |
| `nn.LSTM` bias | `zeros_` (except forget=1.0) | Standard LSTM init |
| `nn.Embedding` | `normal_(0, 0.01)` | Small random values |
| All biases | `zeros_` | Start from zero |

### Validation Checklist per Architecture

**Every architecture must validate:**
- [ ] Input shape matches expected dimensions
- [ ] All inputs are finite (no NaN/Inf)
- [ ] Output shape matches expected dimensions
- [ ] All outputs are finite
- [ ] Model-specific properties (see below)

**Architecture-specific validations:**
- **Classification:** Probabilities sum to 1, all in [0, 1]
- **TFT:** Quantile ordering (q10 < q25 < ... < q90)
- **Attention:** Attention weights sum to 1
- **Ensemble:** Effective weights sum to 1
- **Decomposition:** Trend + seasonal = original

---

**Use these templates as starting points. All include proper initialization and validation.**
