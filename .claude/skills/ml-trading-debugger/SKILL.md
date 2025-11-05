---
name: ml-trading-debugger
description: Specialized debugging and problem-solving for ML-driven quantitative trading systems. Use when encountering issues with model training, data pipeline failures, GPU memory problems, portfolio optimization disconnects, storage limitations, or production deployment challenges in the ML_integrated_forecasting trading system. Handles PyTorch training issues, TFT/LSTM integration problems, RunPod infrastructure challenges, and institutional-grade analysis requirements.
---

# ML Trading System Debugger

Systematic debugging and problem-solving workflows for the ML_integrated_forecasting quantitative trading platform.

## Core Debugging Methodology

### Phase 1: Issue Identification & Triage

When encountering any problem:

1. **Classify the issue category**:
   - Model Training (convergence, loss patterns, gradient issues)
   - Data Pipeline (missing data, API failures, format issues)
   - Infrastructure (GPU memory, storage limits, pod failures)
   - Integration (model-dashboard disconnect, portfolio optimization)
   - Performance (accuracy below target, execution speed)
   - Production (reliability, institutional standards)

2. **Gather diagnostic context**:
   - Read error messages and stack traces completely
   - Check recent changes to code, data sources, or infrastructure
   - Identify which component failed (see references/architecture.md)
   - Review similar past issues in references/common-issues.md

3. **Run automated diagnostics** (if applicable):
   ```bash
   python scripts/diagnostic.py --component [training|data|infrastructure|integration]
   ```

### Phase 2: Systematic Investigation

For each issue category:

#### Model Training Issues

**Symptoms**: Loss not decreasing, NaN losses, gradient explosions, poor validation performance

**Investigation sequence**:
1. Check data quality and preprocessing
   - Verify no NaN/Inf values in features
   - Confirm proper normalization/standardization
   - Validate train/val/test split integrity
   
2. Examine model architecture
   - Review layer dimensions and parameter counts
   - Check for architecture mismatches (CNN-LSTM-Transformer hybrid)
   - Verify attention mechanisms and temporal fusion blocks
   
3. Analyze training hyperparameters
   - Learning rate (consider warmup and decay schedules)
   - Batch size vs GPU memory (8xB200 = 192GB VRAM available)
   - Mixed precision training settings (TF32 acceleration)
   - Gradient clipping thresholds
   
4. Inspect loss curves and metrics
   - Plot train/val loss over epochs
   - Check for overfitting (train loss ↓, val loss ↑)
   - Review directional accuracy trends (target: 70-75%)
   
5. Debug optimizer state
   - Check if gradients are flowing (use gradient norms)
   - Verify optimizer parameters (Adam, AdamW settings)
   - Consider learning rate finder experiments

**Common fixes**:
- Reduce learning rate or implement warmup
- Increase gradient clipping threshold
- Add dropout or regularization
- Reduce model size if GPU memory insufficient
- Fix data normalization issues
- Adjust loss function weighting

#### Data Pipeline Issues

**Symptoms**: Missing data, API rate limits, format mismatches, staleness

**Investigation sequence**:
1. Verify data source availability
   - yfinance API status
   - Financial Modeling Prep API quota
   - NewsAPI and Reddit scraping status
   - FRED API connectivity
   
2. Check data freshness and completeness
   - Confirm date ranges (target: 25 years)
   - Validate all 500 S&P tickers present
   - Verify fundamental data availability
   - Check sentiment data coverage
   
3. Inspect data formats and schemas
   - Parquet file structure and compression
   - DataFrame column names and types
   - Time series alignment across sources
   
4. Review caching and storage
   - Check /mnt/user-data/uploads for existing data
   - Verify Parquet compression effectiveness
   - Monitor disk usage on RunPod instance

**Common fixes**:
- Implement exponential backoff for API retries
- Add data validation checks before processing
- Create fallback data sources
- Optimize Parquet compression settings
- Implement incremental data updates
- Add data quality metrics logging

#### Infrastructure Issues (RunPod)

**Symptoms**: Pod crashes, OOM errors, storage full, checkpoint failures

**Investigation sequence**:
1. Check GPU memory utilization
   - Review current VRAM usage (192GB total)
   - Identify memory leaks (gradient accumulation issues)
   - Verify batch size appropriateness
   - Check for unnecessarily cached tensors
   
2. Analyze storage consumption
   - List checkpoint files and sizes
   - Review dataset storage (Parquet files)
   - Check temporary file accumulation
   - Identify large model files
   
3. Examine checkpoint strategy
   - Verify intelligent checkpoint saving (not every epoch)
   - Confirm cleanup of old checkpoints
   - Check checkpoint validation before saving
   
4. Review distributed training setup
   - Verify multi-GPU configuration (8xB200)
   - Check data parallelism implementation
   - Confirm gradient synchronization

**Common fixes**:
- Implement checkpoint rotation (keep only best 3)
- Enable gradient checkpointing for memory efficiency
- Reduce batch size or enable gradient accumulation
- Compress old datasets and checkpoints
- Clean up temporary files regularly
- Use mixed precision training (automatic TF32)
- Implement smart checkpoint frequency (val loss improvement only)

#### Integration Issues

**Symptoms**: Model outputs not reaching dashboard, portfolio optimization disconnected, analysis functions missing ML predictions

**Investigation sequence**:
1. Trace the data flow
   - Model training → checkpoint saving
   - Checkpoint loading → inference generation
   - Predictions → dashboard input format
   - Dashboard → portfolio optimization
   
2. Check interface contracts
   - Verify expected input/output formats
   - Confirm API endpoints or file paths
   - Validate data serialization (Parquet, pickle, JSON)
   
3. Review integration points
   - InvestmentDecisionEngine connection to ML models
   - TFT model outputs to factor scoring
   - Risk models receiving predictions
   - Dashboard analysis functions data sources
   
4. Test end-to-end pipeline
   - Run minimal integration test
   - Verify each stage produces expected output
   - Check for version mismatches

**Common fixes**:
- Create standardized prediction output format
- Implement prediction caching layer
- Add integration tests for ML → dashboard flow
- Build InvestmentDecisionEngine data adapters
- Document expected interfaces clearly
- Add logging at integration boundaries

#### Performance Issues

**Symptoms**: Directional accuracy below 70%, Sharpe ratio not improving, poor out-of-sample results

**Investigation sequence**:
1. Validate evaluation methodology
   - Confirm proper train/val/test splits
   - Check for data leakage (future info in past)
   - Verify walk-forward validation setup
   - Review backtest assumptions
   
2. Analyze prediction quality
   - Plot prediction vs actual returns
   - Calculate residual analysis
   - Check prediction confidence calibration
   - Review error distribution (systematic biases?)
   
3. Examine feature engineering
   - Feature importance analysis
   - Correlation with target variable
   - Temporal stability of features
   - Missing value handling
   
4. Review ensemble strategy
   - Individual model contributions
   - Ensemble weight optimization
   - Diversity of model predictions
   
5. Investigate alpha decay
   - Check performance over time
   - Test regime stability
   - Analyze different market conditions

**Common fixes**:
- Add more informative features (sentiment, fundamentals)
- Implement regime-aware models
- Tune ensemble weights with Optuna
- Add confidence-weighted predictions
- Implement adaptive learning rate schedules
- Expand training data (more history, more tickers)
- Add macro indicators for context

### Phase 3: Solution Implementation

1. **Create hypothesis**: Based on investigation, form specific hypothesis about root cause
2. **Design minimal test**: Create smallest possible test to validate hypothesis
3. **Implement fix**: Make targeted changes with clear rollback path
4. **Validate solution**: Confirm fix resolves issue without creating new problems
5. **Document learnings**: Update references/common-issues.md with pattern

### Phase 4: Prevention & Monitoring

After fixing an issue:

1. **Add monitoring**: Create alerts or checks to catch similar issues early
2. **Improve logging**: Add detailed logging at failure points
3. **Create tests**: Add unit/integration tests covering the failure case
4. **Update documentation**: Document the solution pattern
5. **Review related code**: Check for similar patterns elsewhere

## Institutional-Grade Standards Checklist

When preparing work for hedge fund presentations:

- [ ] All analysis includes multiple scenarios and sensitivity analysis
- [ ] Sources are rigorously cited and authoritative
- [ ] Quantitative claims have confidence intervals
- [ ] Risk metrics prominently featured (VaR, CVaR, max drawdown)
- [ ] Comparison to relevant benchmarks (S&P 500, sector indices)
- [ ] Professional visualizations with clear labels
- [ ] Executive summary with key takeaways up front
- [ ] Methodology section explaining approach
- [ ] Limitations and assumptions explicitly stated
- [ ] Forward-looking statements appropriately caveated

## Quick Reference Commands

```bash
# Run comprehensive diagnostics
python scripts/diagnostic.py --full

# Check specific component
python scripts/diagnostic.py --component training
python scripts/diagnostic.py --component data
python scripts/diagnostic.py --component infrastructure

# Validate data pipeline
python scripts/diagnostic.py --validate-data

# Check GPU memory and storage
python scripts/diagnostic.py --system-health
```

## Advanced Debugging Techniques

### Interactive Debugging
- Use pdb for Python debugging: `import pdb; pdb.set_trace()`
- Add breakpoints before error points
- Inspect tensor shapes and values interactively

### Profiling
- PyTorch profiler for training bottlenecks
- Memory profiler for allocation tracking
- cProfile for CPU-bound operations

### Logging Best Practices
- Log at decision boundaries
- Include context (batch #, epoch, ticker symbol)
- Use structured logging (JSON format)
- Separate debug vs production logging levels

## References

For detailed information:
- **Architecture overview**: See references/architecture.md
- **Common issue patterns**: See references/common-issues.md
- **Tech stack details**: See references/tech-stack.md
