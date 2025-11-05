---
name: ml-architecture-builder
description: Use when implementing PyTorch neural network architectures for financial forecasting - enforces weight initialization, output validation, and gradient flow checks for Temporal Fusion Transformers, LSTMs, attention mechanisms, regime classifiers, and ensemble models to prevent training failures
---

# ML Architecture Builder

## Overview

**Architecture is structure + initialization + validation.** Every PyTorch model must include proper weight initialization, output shape/finite checks, and gradient flow configuration. Structure alone causes silent training failures.

**Core principle:** PyTorch defaults are insufficient. Always initialize explicitly and validate outputs.

## When to Use

Use this skill when implementing:
- Time series forecasting models (TFT, Autoformer, N-BEATS)
- Attention mechanisms (multi-head, autocorrelation)
- LSTM/GRU encoders
- Regime classifiers
- Ensemble models
- Any custom PyTorch architecture

**Don't skip validation because:**
- "PyTorch handles initialization automatically" (defaults cause poor convergence)
- "Architecture looks correct" (structure ≠ trainable)
- "Will add validation after it trains" (won't train without proper init)
- "Simple model doesn't need initialization" (all models need it)

## Implementation Checklist

Before training ANY PyTorch model:

- [ ] Initialize all weights explicitly (Xavier/He/orthogonal)
- [ ] Validate output shapes in forward pass
- [ ] Check outputs are finite (no NaN/Inf)
- [ ] Configure gradient clipping (max_norm=1.0)
- [ ] Verify model-specific properties (probs sum to 1, quantiles ordered, etc.)

## Weight Initialization Patterns

### Pattern 1: Feedforward Networks

```python
class RegimeClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=4, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, output_size)

        # WEIGHT INITIALIZATION (MANDATORY)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier for tanh/sigmoid, Kaiming for ReLU
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
```

### Pattern 2: LSTM/GRU Networks

```python
class LSTMEncoder(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )

        # LSTM-SPECIFIC INITIALIZATION (MANDATORY)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM weights for stable gradients."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden: Orthogonal (prevents vanishing gradients)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: zeros (except forget gate bias = 1.0)
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
```

### Pattern 3: Convolutional Networks

```python
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize conv weights with Kaiming (for ReLU)."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
```

## Output Validation Patterns

### Pattern 1: Shape and Finiteness Checks

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with validation."""

    # INPUT VALIDATION
    assert x.ndim == 3, f"Expected 3D input [batch, seq, features], got {x.ndim}D"
    assert x.shape[2] == self.input_size, \
        f"Expected {self.input_size} features, got {x.shape[2]}"

    # ... computation ...

    # OUTPUT VALIDATION
    expected_shape = (x.shape[0], self.seq_len, self.output_size)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"

    assert torch.isfinite(output).all(), \
        "Output contains NaN or Inf - check for numerical instability"

    return output
```

### Pattern 2: Probability Validation (Classification)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.classifier(x)

    # During inference, validate probabilities
    if not self.training:
        probs = F.softmax(logits, dim=1)

        # Probabilities must sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-6), \
            f"Probabilities sum to {probs.sum(dim=1).mean():.6f}, not 1.0"

        # All probabilities in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all(), \
            "Probabilities outside [0, 1] range"

    return logits
```

### Pattern 3: Quantile Validation (TFT)

```python
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # ... TFT computation ...

    quantiles = {
        'q10': self.q10_head(attended),
        'q25': self.q25_head(attended),
        'q50': self.q50_head(attended),
        'q75': self.q75_head(attended),
        'q90': self.q90_head(attended),
    }

    # QUANTILE ORDERING VALIDATION
    if not self.training:
        assert (quantiles['q10'] <= quantiles['q25']).all(), "q10 > q25 violation"
        assert (quantiles['q25'] <= quantiles['q50']).all(), "q25 > q50 violation"
        assert (quantiles['q50'] <= quantiles['q75']).all(), "q50 > q75 violation"
        assert (quantiles['q75'] <= quantiles['q90']).all(), "q75 > q90 violation"

    return quantiles
```

## Gradient Flow Configuration

### Training Loop with Gradient Clipping

```python
def train_epoch(model, dataloader, optimizer, max_grad_norm=1.0):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        output = model(batch['input'])
        loss = criterion(output, batch['target'])

        loss.backward()

        # GRADIENT CLIPPING (MANDATORY for deep networks)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Gradient Flow Verification

```python
def check_gradient_flow(model, loss):
    """Check if gradients are flowing properly."""
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            # Check for vanishing gradients
            if grad_norm < 1e-7:
                print(f"WARNING: Vanishing gradient in {name}: {grad_norm}")

            # Check for exploding gradients
            if grad_norm > 100:
                print(f"WARNING: Exploding gradient in {name}: {grad_norm}")

            # Check for NaN gradients
            if not torch.isfinite(param.grad).all():
                print(f"ERROR: NaN/Inf gradient in {name}")
```

## Common Mistakes

### Mistake 1: Relying on Default Initialization
❌ **Bad:**
```python
self.fc1 = nn.Linear(10, 64)
# No initialization - uses PyTorch default
```

✅ **Good:**
```python
self.fc1 = nn.Linear(10, 64)
nn.init.xavier_uniform_(self.fc1.weight)
nn.init.zeros_(self.fc1.bias)
```

### Mistake 2: No Output Validation
❌ **Bad:**
```python
def forward(self, x):
    output = self.model(x)
    return output  # Hope it's correct!
```

✅ **Good:**
```python
def forward(self, x):
    output = self.model(x)
    assert output.shape == expected_shape
    assert torch.isfinite(output).all()
    return output
```

### Mistake 3: Missing Gradient Clipping
❌ **Bad:**
```python
loss.backward()
optimizer.step()  # Gradients may explode
```

✅ **Good:**
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "PyTorch handles initialization" | Defaults cause poor convergence | Always initialize explicitly |
| "Architecture looks correct" | Structure ≠ trainable | Add initialization + validation |
| "Simple model doesn't need init" | All models need proper init | No exceptions, always initialize |
| "Will add validation later" | Later = never (pressure skips it) | Validate in forward pass now |
| "Gradients will be fine" | Deep networks need clipping | Configure max_norm=1.0 always |
| "Output shape will be correct" | Shape bugs are silent | Assert shape in forward pass |

## Architecture-Specific Patterns

For detailed templates, see [architecture-templates.md](architecture-templates.md):
- Temporal Fusion Transformer (quantile heads, variable selection, attention)
- Autoformer (series decomposition, autocorrelation attention)
- N-BEATS (stacked blocks, residual connections)
- Regime detectors (classification, softmax validation)
- Ensemble models (weight normalization, output aggregation)

## Real-World Impact

**Initialization prevents:**
- Model not training (loss plateaus from poor init)
- Slow convergence (10x-100x more epochs needed)
- Vanishing gradients (LSTM hidden states go to zero)
- Exploding gradients (NaN loss after 5 iterations)

**Validation prevents:**
- Silent shape mismatches (predictions off by one timestep)
- NaN propagation (one bad batch ruins entire model)
- Quantile violations (q10 > q90 produces nonsense forecasts)
- Probability errors (predictions sum to 0.87 not 1.0)

**Time investment:**
- Add initialization: 2 minutes per model
- Add validation: 1 minute per forward pass
- Debug training without init: 3-8 hours + frustration

## Bottom Line

**Good architecture requires three components:**
1. Structure (layers, activations)
2. Initialization (Xavier/He/orthogonal) ← AGENTS SKIP THIS
3. Validation (shapes, finite, properties) ← AGENTS SKIP THIS

Don't just design the architecture—initialize and validate it.

Every model gets explicit initialization. Every forward pass gets validation. No exceptions.
