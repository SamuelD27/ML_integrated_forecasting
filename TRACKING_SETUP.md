# Experiment Tracking Setup Guide

## ✅ Status: FULLY CONFIGURED

All three tracking systems are set up and ready to use!

---

## 📊 Three Tracking Systems

### 1. **Weights & Biases (W&B)** - Cloud-based ✓ READY
- **Status**: Logged in as `samsamd2704-nus`
- **Project**: `hybrid-trading-model`
- **Dashboard**: https://wandb.ai/samsamd2704-nus/hybrid-trading-model
- **API Key**: Configured in `.env`

**Test Results**:
```
✓ Login successful
✓ Test run created: https://wandb.ai/samsamd2704-nus/hybrid-trading-model/runs/7q0qxv4m
✓ Metrics logged successfully
```

### 2. **TensorBoard** - Local visualization ✓ READY
- **Status**: Enabled in config
- **Log Directory**: `runs/hybrid_model/`
- **Start Command**: `tensorboard --logdir runs/`
- **View at**: http://localhost:6006

### 3. **MLflow** - Local experiment tracking ✓ READY
- **Status**: Tested and working
- **Storage**: `mlruns/` directory
- **Start Command**: `mlflow ui --backend-store-uri mlruns/`
- **View at**: http://localhost:5000
- **Note**: Currently disabled in config (can enable if needed)

---

## 🚀 Quick Start

### Start Training with Tracking

```bash
# Navigate to training directory
cd training

# Run training (W&B and TensorBoard will log automatically)
python train_hybrid.py --config config.yaml
```

### Monitor Training in Real-Time

**Terminal 1 - TensorBoard**:
```bash
tensorboard --logdir runs/
# Open: http://localhost:6006
```

**Terminal 2 - MLflow** (optional):
```bash
mlflow ui --backend-store-uri mlruns/
# Open: http://localhost:5000
```

**Web Browser - W&B**:
```
https://wandb.ai/samsamd2704-nus/hybrid-trading-model
```

---

## 📈 What Gets Logged

Your `train_hybrid.py` script automatically logs:

### Metrics (every epoch):
- ✓ Loss (train/validation)
- ✓ RMSE (train/validation)
- ✓ Directional accuracy
- ✓ Sharpe ratio
- ✓ Max drawdown
- ✓ Learning rate

### Artifacts:
- ✓ Model checkpoints (best model)
- ✓ Training history (JSON)
- ✓ Configuration (YAML)

### TensorBoard Logs:
```python
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('RMSE/train', train_rmse, epoch)
writer.add_scalar('RMSE/val', val_rmse, epoch)
writer.add_scalar('Metrics/directional_accuracy', val_dir_acc, epoch)
writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
```

---

## ⚙️ Configuration

### Enable/Disable Tracking

Edit `training/config.yaml`:

```yaml
tracking:
  # Weights & Biases
  wandb:
    enabled: true  # ✓ Currently enabled
    project: "hybrid-trading-model"
    entity: null
    tags: ["hybrid", "deep-learning", "trading"]

  # TensorBoard
  tensorboard:
    enabled: true  # ✓ Currently enabled
    log_dir: "runs/hybrid_model"

  # MLflow
  mlflow:
    enabled: false  # Disabled (enable if needed)
    tracking_uri: "http://localhost:5000"
    experiment_name: "hybrid_trading"
```

### Environment Variables

Your `.env` file (already configured):
```bash
WANDB_API_KEY=4e6f46e0f12cb3edced9a0b4d94ba315c4be7b59
WANDB_PROJECT=hybrid-trading-model
```

---

## 🎯 W&B Features

### View Training Progress
1. Go to: https://wandb.ai/samsamd2704-nus/hybrid-trading-model
2. Click on any run to see:
   - Real-time training curves
   - System metrics (GPU, CPU, memory)
   - Hyperparameters
   - Model artifacts

### Compare Experiments
- Select multiple runs
- Compare metrics side-by-side
- Analyze hyperparameter impact

### Hyperparameter Sweeps (Advanced)
```bash
# Create sweep configuration
wandb sweep sweep_config.yaml

# Run sweep agent
wandb agent <sweep-id>
```

---

## 📊 TensorBoard Features

### Start TensorBoard
```bash
tensorboard --logdir runs/
```

### Available Visualizations
1. **Scalars**: Loss curves, metrics over time
2. **Graphs**: Model architecture visualization
3. **Distributions**: Weight and gradient distributions
4. **Histograms**: Parameter evolution
5. **Projector**: Embedding visualization

### Multiple Runs Comparison
```bash
# Compare multiple training runs
tensorboard --logdir runs/ --port 6006
```

---

## 🗂️ MLflow Features (Optional)

### Enable MLflow

Edit `training/config.yaml`:
```yaml
mlflow:
  enabled: true  # Change to true
```

### Start MLflow UI
```bash
mlflow ui --backend-store-uri mlruns/
```

### Features
- **Experiment Comparison**: Compare runs across different experiments
- **Model Registry**: Version and stage models (dev/staging/production)
- **Artifact Storage**: Store models, plots, data files
- **API Access**: Programmatic access to all logged data

---

## 🔍 Viewing Past Experiments

### W&B (Cloud Storage)
```bash
# All experiments are automatically saved to cloud
# Access anytime at: https://wandb.ai/samsamd2704-nus/hybrid-trading-model
```

### TensorBoard (Local)
```bash
# View all local runs
tensorboard --logdir runs/

# View specific run
tensorboard --logdir runs/hybrid_model/run_20231105
```

### MLflow (Local)
```bash
# View all experiments
mlflow ui --backend-store-uri mlruns/
```

---

## 🛠️ Advanced Usage

### Log Custom Metrics

Add to your training script:

```python
# W&B
import wandb
wandb.log({
    "custom_metric": value,
    "custom_chart": wandb.plot.line(...)
})

# TensorBoard
writer.add_scalar('Custom/metric', value, step)
writer.add_histogram('Weights/layer1', model.layer1.weight, step)

# MLflow
import mlflow
mlflow.log_metric("custom_metric", value, step=step)
```

### Log Images/Plots

```python
# W&B
wandb.log({"predictions": wandb.Image(img)})

# TensorBoard
writer.add_image('Predictions', img_tensor, step)
writer.add_figure('Results', matplotlib_fig, step)
```

### Log Artifacts

```python
# W&B
wandb.save("model.pt")
wandb.save("results.csv")

# MLflow
mlflow.log_artifact("model.pt")
mlflow.log_artifact("results.csv")
```

---

## 📝 Best Practices

1. **Tag Your Runs**: Use descriptive tags
   ```python
   wandb.init(tags=["experiment-1", "baseline", "cnn-lstm"])
   ```

2. **Log Hyperparameters**: Save all config
   ```python
   wandb.config.update(config)
   ```

3. **Regular Checkpointing**: Save models frequently
   ```python
   if epoch % 10 == 0:
       model.save_model(f"checkpoint_epoch_{epoch}.pt")
   ```

4. **Compare Runs**: Always compare against baseline

5. **Document Experiments**: Add notes to W&B runs

---

## 🐛 Troubleshooting

### W&B Issues

**Issue**: "Not logged in"
```bash
# Re-login
wandb login --relogin
```

**Issue**: "Run not syncing"
```bash
# Check internet connection
# Force sync
wandb sync <run-path>
```

### TensorBoard Issues

**Issue**: "Address already in use"
```bash
# Use different port
tensorboard --logdir runs/ --port 6007
```

**Issue**: "No logs found"
```bash
# Check log directory exists
ls -la runs/hybrid_model/
```

### MLflow Issues

**Issue**: "Cannot connect to tracking server"
```bash
# Ensure MLflow UI is running
mlflow ui --backend-store-uri mlruns/
```

---

## 📚 Resources

### Documentation
- **W&B**: https://docs.wandb.ai/
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **MLflow**: https://mlflow.org/docs/latest/

### Tutorials
- **W&B PyTorch**: https://docs.wandb.ai/guides/integrations/pytorch
- **TensorBoard with PyTorch**: https://pytorch.org/docs/stable/tensorboard.html
- **MLflow PyTorch**: https://mlflow.org/docs/latest/tracking.html

---

## ✅ Verification Checklist

Before starting training, verify:

- [x] W&B login successful
- [x] API key in `.env` file
- [x] `tracking.wandb.enabled: true` in config
- [x] `tracking.tensorboard.enabled: true` in config
- [x] `runs/` directory exists
- [x] `checkpoints/` directory exists

**All checks passed! You're ready to train!** 🚀

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Start training | `cd training && python train_hybrid.py --config config.yaml` |
| View TensorBoard | `tensorboard --logdir runs/` |
| View W&B dashboard | https://wandb.ai/samsamd2704-nus/hybrid-trading-model |
| Start MLflow | `mlflow ui --backend-store-uri mlruns/` |
| Check W&B status | `wandb status` |
| Re-login W&B | `wandb login --relogin` |

---

**Status**: ✅ READY TO TRAIN WITH FULL TRACKING!
