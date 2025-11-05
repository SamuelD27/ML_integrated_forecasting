# Common Issues and Solutions

This document catalogs recurring problems in the ML trading system and their proven solutions.

## Model Training Issues

### Issue: NaN Losses During Training

**Symptoms**:
- Training loss becomes NaN after several epochs
- Gradients explode to infinity
- Model predictions all become identical

**Root Causes**:
1. Learning rate too high
2. Numerical instability in loss calculation
3. Extreme values in input features
4. Division by zero in custom loss functions

**Solutions**:
- Reduce learning rate by 10x (e.g., 1e-3 → 1e-4)
- Implement gradient clipping (threshold: 1.0-5.0)
- Add epsilon to denominators (1e-8) in loss calculations
- Verify feature normalization (check for outliers)
- Use mixed precision with loss scaling
- Add validation checks: `torch.isnan(loss).any()` before backward()

**Prevention**:
```python
# Add after loss calculation
if torch.isnan(loss):
    print(f"NaN loss at step {step}, skipping batch")
    continue

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue: Model Not Converging

**Symptoms**:
- Validation loss plateaus immediately
- Training loss decreases but validation loss doesn't improve
- Directional accuracy stuck at ~50% (random guessing)

**Root Causes**:
1. Insufficient model capacity for data complexity
2. Learning rate too low (under-fitting)
3. Data leakage or improper train/val split
4. Features not informative for prediction task
5. Class imbalance (if classification)

**Solutions**:
- Increase model size (more layers, wider dimensions)
- Learning rate finder: test range 1e-5 to 1e-1
- Verify temporal train/val split (no future data in training)
- Feature importance analysis: remove uninformative features
- Implement learning rate warmup (first 10% of training)
- Check data distribution: training vs validation

**Investigation Steps**:
1. Train on tiny dataset (10 samples) → should overfit quickly
2. If can't overfit, problem is model capacity or implementation bug
3. If overfits easily, problem is regularization or data quality

### Issue: Overfitting (Train Loss ↓, Val Loss ↑)

**Symptoms**:
- Training loss continues decreasing
- Validation loss increases or plateaus
- Large gap between train and validation accuracy

**Root Causes**:
1. Model too complex for available data
2. Insufficient regularization
3. Training too long without early stopping
4. Data augmentation not applied

**Solutions**:
- Add dropout (0.1-0.3) to all layers
- Implement early stopping (patience: 5-10 epochs)
- Reduce model size (fewer parameters)
- Apply L2 regularization (weight decay: 1e-4)
- Expand training data (more tickers, longer history)
- Data augmentation: add noise, temporal shifts

### Issue: GPU Out of Memory

**Symptoms**:
- RuntimeError: CUDA out of memory
- Process killed without error message
- Training crashes mid-epoch

**Root Causes**:
1. Batch size too large for GPU memory
2. Model size exceeds available VRAM
3. Gradient accumulation not cleared properly
4. Memory leak from cached tensors

**Solutions**:
- Reduce batch size by 50% and test
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- Clear cache periodically: `torch.cuda.empty_cache()`
- Use gradient accumulation for smaller batches:
  ```python
  accumulation_steps = 4
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```
- Monitor memory: `torch.cuda.memory_summary()`
- Mixed precision training (automatic with TF32 on B200)

## Data Pipeline Issues

### Issue: API Rate Limiting

**Symptoms**:
- HTTP 429 errors from APIs
- Missing data for specific tickers/dates
- Incomplete data downloads

**Root Causes**:
1. Too many requests per minute
2. No exponential backoff on failures
3. Concurrent requests exceeding limits

**Solutions**:
- Implement rate limiting:
  ```python
  import time
  from functools import wraps
  
  def rate_limit(calls_per_minute):
      min_interval = 60.0 / calls_per_minute
      last_call = [0.0]
      
      def decorator(func):
          @wraps(func)
          def wrapper(*args, **kwargs):
              elapsed = time.time() - last_call[0]
              if elapsed < min_interval:
                  time.sleep(min_interval - elapsed)
              result = func(*args, **kwargs)
              last_call[0] = time.time()
              return result
          return wrapper
      return decorator
  ```
- Add exponential backoff:
  ```python
  import requests
  from time import sleep
  
  def fetch_with_retry(url, max_retries=5):
      for attempt in range(max_retries):
          try:
              response = requests.get(url)
              response.raise_for_status()
              return response
          except requests.exceptions.HTTPError as e:
              if e.response.status_code == 429:
                  wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                  print(f"Rate limited, waiting {wait_time}s")
                  sleep(wait_time)
              else:
                  raise
      raise Exception("Max retries exceeded")
  ```
- Use API-specific rate limits (yfinance: 2000/hour, FMP: depends on plan)

### Issue: Missing or Stale Data

**Symptoms**:
- Missing values in dataset
- Data doesn't update with market
- Gaps in time series

**Root Causes**:
1. API failures not detected
2. No data validation after download
3. Caching old data without refresh
4. Ticker delisted or symbol changed

**Solutions**:
- Implement data validation:
  ```python
  def validate_data(df, ticker):
      # Check for missing dates
      expected_dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='B')
      missing_dates = expected_dates.difference(df.index)
      if len(missing_dates) > 100:
          raise ValueError(f"{ticker}: {len(missing_dates)} missing dates")
      
      # Check for NaN values
      nan_pct = df.isna().sum() / len(df)
      if (nan_pct > 0.05).any():
          raise ValueError(f"{ticker}: >5% NaN values in {nan_pct[nan_pct > 0.05].index}")
      
      # Check for staleness
      last_date = df.index[-1]
      if (pd.Timestamp.now() - last_date).days > 7:
          raise ValueError(f"{ticker}: Data stale (last: {last_date})")
  ```
- Add data freshness checks before training
- Implement fallback data sources
- Log data quality metrics to monitor

### Issue: Data Format Inconsistencies

**Symptoms**:
- Shape mismatches during training
- Type errors (string instead of float)
- Different column names across sources

**Root Causes**:
1. Multiple data sources with different schemas
2. No standardization layer
3. API response format changes

**Solutions**:
- Create data standardization pipeline:
  ```python
  def standardize_schema(df, source):
      """Standardize column names and types across sources"""
      schema_map = {
          'yfinance': {
              'Open': 'open', 'High': 'high', 'Low': 'low',
              'Close': 'close', 'Volume': 'volume'
          },
          'fmp': {
              'date': 'date', 'eps': 'earnings_per_share',
              'revenue': 'total_revenue'
          }
      }
      
      if source in schema_map:
          df = df.rename(columns=schema_map[source])
      
      # Ensure consistent types
      numeric_cols = ['open', 'high', 'low', 'close', 'volume']
      for col in numeric_cols:
          if col in df.columns:
              df[col] = pd.to_numeric(df[col], errors='coerce')
      
      return df
  ```
- Add schema validation tests
- Version control data schemas

## Infrastructure Issues

### Issue: RunPod Instance Crashes

**Symptoms**:
- Pod terminates unexpectedly
- "Out of space" errors
- Connection lost during training

**Root Causes**:
1. Storage full from excessive checkpoints
2. Memory leak in training loop
3. Unhandled exceptions causing crashes

**Solutions**:
- Implement checkpoint rotation:
  ```python
  import os
  from pathlib import Path
  
  def save_checkpoint_with_rotation(model, optimizer, epoch, val_loss, save_dir, keep_best=3):
      save_dir = Path(save_dir)
      save_dir.mkdir(exist_ok=True)
      
      checkpoint_path = save_dir / f"checkpoint_epoch{epoch}_loss{val_loss:.4f}.pt"
      torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'val_loss': val_loss
      }, checkpoint_path)
      
      # Keep only best N checkpoints
      checkpoints = sorted(save_dir.glob("checkpoint_*.pt"), 
                          key=lambda p: float(p.stem.split('loss')[1].split('.pt')[0]))
      for old_checkpoint in checkpoints[keep_best:]:
          old_checkpoint.unlink()
          print(f"Deleted old checkpoint: {old_checkpoint.name}")
  ```
- Monitor disk usage:
  ```bash
  df -h  # Check available space
  du -sh /path/to/checkpoints  # Check checkpoint size
  ```
- Add storage cleanup script:
  ```python
  # Clean up temporary files periodically
  import shutil
  for tmp_dir in ['/tmp', '/root/.cache']:
      if Path(tmp_dir).exists():
          shutil.rmtree(tmp_dir)
          Path(tmp_dir).mkdir()
  ```

### Issue: Slow Training Speed

**Symptoms**:
- Training taking >24 hours for single epoch
- GPU utilization <50%
- CPU bottleneck

**Root Causes**:
1. Data loading inefficient (CPU bottleneck)
2. Not using all available GPUs
3. Small batch size not utilizing GPU
4. Mixed precision not enabled

**Solutions**:
- Optimize DataLoader:
  ```python
  train_loader = DataLoader(
      dataset,
      batch_size=256,  # Larger batch for GPU efficiency
      num_workers=8,   # Parallel data loading
      pin_memory=True, # Faster CPU→GPU transfer
      prefetch_factor=2 # Prefetch batches
  )
  ```
- Enable distributed training on all GPUs:
  ```python
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel
  
  # Initialize process group
  dist.init_process_group(backend='nccl')
  
  # Wrap model
  model = DistributedDataParallel(model, device_ids=[local_rank])
  ```
- Monitor GPU utilization: `nvidia-smi dmon -s u`
- Profile training loop to find bottlenecks:
  ```python
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU, 
                  torch.profiler.ProfilerActivity.CUDA]
  ) as prof:
      # Training loop
      pass
  print(prof.key_averages().table())
  ```

## Integration Issues

### Issue: Model Predictions Not Reaching Dashboard

**Symptoms**:
- Dashboard shows "No predictions available"
- Analysis functions error: KeyError for prediction columns
- InvestmentDecisionEngine using only traditional signals

**Root Cause**:
Trained models and dashboard operate in separate workflows without connection

**Solution Implementation**:

1. **Create prediction pipeline**:
   ```python
   # inference/prediction_pipeline.py
   class PredictionPipeline:
       def __init__(self, model_checkpoint_path, output_dir):
           self.model = self.load_model(model_checkpoint_path)
           self.output_dir = Path(output_dir)
           
       def generate_predictions(self, tickers, start_date, end_date):
           predictions = []
           for ticker in tickers:
               data = self.load_ticker_data(ticker, start_date, end_date)
               pred = self.model.predict(data)
               predictions.append({
                   'ticker': ticker,
                   'date': end_date,
                   'horizon': [1, 5, 20, 60],
                   'prediction': pred,
                   'confidence': self.model.confidence(data)
               })
           
           # Save to Parquet for dashboard access
           df = pd.DataFrame(predictions)
           output_path = self.output_dir / f"predictions_{end_date}.parquet"
           df.to_parquet(output_path, compression='snappy')
           return output_path
   ```

2. **Update dashboard data loader**:
   ```python
   # dashboard/data_loaders/predictions.py
   def load_latest_predictions():
       pred_dir = Path('/data/predictions')
       latest_file = max(pred_dir.glob('predictions_*.parquet'), 
                        key=lambda p: p.stat().st_mtime)
       return pd.read_parquet(latest_file)
   ```

3. **Integrate with InvestmentDecisionEngine**:
   ```python
   # analysis/decision_engine/scoring.py
   def calculate_momentum_score(ticker, traditional_indicators, ml_predictions):
       # Traditional momentum (40%)
       trad_score = calculate_traditional_momentum(traditional_indicators)
       
       # ML prediction score (60%)
       pred_df = ml_predictions[ml_predictions['ticker'] == ticker]
       ml_score = (pred_df['prediction'].mean() * pred_df['confidence'].mean())
       
       # Combine
       return 0.4 * trad_score + 0.6 * ml_score
   ```

### Issue: Directional Accuracy Below Target

**Symptoms**:
- Accuracy stuck at 65% (target: 70-75%)
- Sharpe ratio improvement below 15%
- Out-of-sample performance poor

**Investigation**:
1. Check if issue is model capacity or data quality
2. Analyze error patterns (systematic biases?)
3. Review feature engineering
4. Test ensemble diversity

**Solutions**:
- **Expand features**:
  - Add FinBERT sentiment from news
  - Include macro regime indicators (FRED data)
  - Calculate factor exposures (value, momentum, quality)
  - Add options-implied volatility
  
- **Improve model architecture**:
  - Scale to 500M+ parameters (currently 100M)
  - Add attention on fundamental features
  - Implement regime-conditional predictions
  
- **Enhance ensemble**:
  - Train more diverse models (different architectures)
  - Optimize ensemble weights with validation data
  - Use confidence-weighted averaging
  
- **Refine training**:
  - Expand training data (15→25 years, more tickers)
  - Implement curriculum learning (easy→hard examples)
  - Add auxiliary tasks (predict volatility, volume)

## Production Issues

### Issue: Failing Institutional Standards

**Symptoms**:
- Feedback: "Analysis looks academic, not hedge fund quality"
- Missing risk metrics or sensitivity analysis
- Sources not cited properly

**Solutions**:

1. **Mandatory components** for every analysis:
   - Executive summary with key takeaways (2-3 bullets)
   - Methodology section (data sources, models, assumptions)
   - Multiple scenarios (base, optimistic, pessimistic)
   - Sensitivity analysis (key variable impacts)
   - Risk metrics (VaR, CVaR, max drawdown, Sharpe)
   - Benchmark comparison (S&P 500, sector index)
   - Limitations section (model assumptions, data quality)

2. **Proper sourcing**:
   - Primary sources: Company filings, Fed data, peer-reviewed papers
   - Avoid: Wikipedia, forums, blog posts
   - Format: "According to XYZ Corp Q4 2024 10-K filing..."

3. **Professional visualizations**:
   - Clean, minimal design
   - Clear axis labels and units
   - Source attribution at bottom
   - Color scheme consistent with brand
   - Export high-resolution for presentations

4. **Quantitative rigor**:
   - All claims backed by data
   - Include confidence intervals
   - Show probability distributions, not point estimates
   - Perform Monte Carlo simulations for uncertainty

## Debugging Checklist

When stuck on any issue:

1. [ ] Read the full error message and stack trace
2. [ ] Check logs for warnings before the error
3. [ ] Reproduce the error with minimal example
4. [ ] Verify input data format and values
5. [ ] Check for recent code or config changes
6. [ ] Review similar past issues in this document
7. [ ] Add logging at failure point
8. [ ] Test each component in isolation
9. [ ] Create unit test covering the failure case
10. [ ] Document the solution in this file
