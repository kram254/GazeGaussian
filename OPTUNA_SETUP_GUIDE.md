# Optuna Automated Training Setup Guide

## 📋 Overview

This guide provides step-by-step instructions to integrate **Optuna** for automated hyperparameter optimization in your GazeGaussian training pipeline.

**What Optuna Will Do:**
- ✅ Automatically find optimal learning rate, DiT architecture parameters, and loss weights
- ✅ Stop unpromising trials early (saves GPU time)
- ✅ Run multiple training experiments in parallel or sequence
- ✅ Generate visualizations of hyperparameter importance

---

## 🎯 What Hyperparameters Will Be Optimized

### MeshHead Training:
1. **Learning rate** (1e-5 to 1e-2)
2. **Batch size** (1, 2)
3. **Shape/Pose/Eye MLP dimensions**

### GazeGaussian Training:
1. **Learning rate** (1e-5 to 5e-3)
2. **DiT depth** (4, 6, 8, 12)
3. **DiT number of heads** (4, 8, 16)
4. **DiT patch size** (4, 8, 16)
5. **Loss importance weights**:
   - VGG importance (0.05 to 0.5)
   - Eye loss importance (0.5 to 2.0)
   - Gaze loss importance (0.05 to 0.5)
   - Orthogonality loss importance (0.001 to 0.1)

---

## 🚀 Step-by-Step Implementation

### Step 1: Install Optuna

In your Colab notebook, add this cell:

```python
!pip install optuna optuna-dashboard plotly kaleido
```

**Packages:**
- `optuna` - Core optimization framework
- `optuna-dashboard` - Web UI for monitoring studies
- `plotly` - Interactive visualizations
- `kaleido` - Export plots as images

---

### Step 2: Understand the Optuna Workflow

```
┌─────────────────────────────────────────────┐
│ 1. Create Study                             │
│    - Define optimization direction          │
│    - Choose pruner strategy                 │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 2. Define Objective Function                │
│    - Suggest hyperparameters                │
│    - Train model with those parameters      │
│    - Return validation metric               │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 3. Optimize (Run Multiple Trials)           │
│    - Optuna suggests hyperparameters        │
│    - Train with those parameters            │
│    - Report intermediate values             │
│    - Prune if underperforming               │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 4. Analyze Results                          │
│    - Get best parameters                    │
│    - Visualize optimization history         │
│    - Understand parameter importance        │
└─────────────────────────────────────────────┘
```

---

### Step 3: Key Optuna Concepts

#### A. Trial
A single training run with specific hyperparameters.

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    train_model(lr)
    return validation_loss
```

#### B. Study
Collection of trials, manages the optimization process.

```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

#### C. Pruning
Stop bad trials early to save time.

```python
for epoch in range(num_epochs):
    train()
    val_loss = validate()
    
    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

#### D. Sampler
Algorithm that suggests hyperparameters (default: TPE - Tree-structured Parzen Estimator).

---

### Step 4: Integration Patterns

#### Pattern 1: Suggest Hyperparameters

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
    dit_depth = trial.suggest_int('dit_depth', 4, 12, step=2)
    
    # Use these in your training
    opt.lr = lr
    opt.batch_size = batch_size
    opt.dit_depth = dit_depth
```

**Available suggestion methods:**
- `suggest_float(name, low, high, log=False)` - Continuous values
- `suggest_int(name, low, high, step=1)` - Integer values
- `suggest_categorical(name, choices)` - Discrete choices

#### Pattern 2: Report Intermediate Values

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    trial.report(val_loss, epoch)
    
    if trial.should_prune():
        raise optuna.TrialPruned()
```

#### Pattern 3: Return Final Metric

```python
def objective(trial):
    final_val_loss = train_model(trial)
    return final_val_loss
```

---

### Step 5: Choose a Pruner

Pruners stop unpromising trials early to save GPU time.

#### MedianPruner (Recommended for Beginners)
Stops trials worse than the median of previous trials.

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=3,
    n_warmup_steps=5,
    interval_steps=1
)
```

**Parameters:**
- `n_startup_trials=3` - Don't prune first 3 trials (need baseline)
- `n_warmup_steps=5` - Don't prune before epoch 5
- `interval_steps=1` - Check every epoch

#### HyperbandPruner (More Aggressive)
Based on successive halving algorithm.

```python
pruner = optuna.pruners.HyperbandPruner(
    min_resource=5,
    max_resource=30,
    reduction_factor=3
)
```

**Parameters:**
- `min_resource=5` - Minimum epochs before pruning
- `max_resource=30` - Maximum epochs
- `reduction_factor=3` - How aggressively to prune

---

### Step 6: Create the Study

```python
import optuna

study = optuna.create_study(
    study_name='gazegaussian_optimization',
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=5,
        interval_steps=1
    ),
    storage='sqlite:///gazegaussian_study.db',
    load_if_exists=True
)
```

**Parameters:**
- `study_name` - Identifier for this optimization run
- `direction` - 'minimize' (loss) or 'maximize' (accuracy)
- `pruner` - Strategy for early stopping
- `storage` - SQLite database to persist results
- `load_if_exists=True` - Resume if interrupted

---

### Step 7: Run Optimization

```python
study.optimize(
    objective,
    n_trials=20,
    timeout=3600*10,
    show_progress_bar=True
)
```

**Parameters:**
- `n_trials=20` - Run 20 different hyperparameter combinations
- `timeout=3600*10` - Stop after 10 hours
- `show_progress_bar=True` - Show progress

---

### Step 8: Analyze Results

#### Get Best Parameters

```python
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
```

#### Visualize Optimization History

```python
import optuna.visualization as vis

fig = vis.plot_optimization_history(study)
fig.show()

fig = vis.plot_param_importances(study)
fig.show()

fig = vis.plot_parallel_coordinate(study)
fig.show()
```

#### Launch Dashboard

```python
!optuna-dashboard sqlite:///gazegaussian_study.db --port 8080
```

Then use Colab's port forwarding or ngrok to access it.

---

## 📊 Expected Results

### What to Expect:
1. **First 3 trials** - Run to completion (baseline)
2. **Subsequent trials** - Many will be pruned after 5-10 epochs
3. **Time savings** - 40-60% reduction in total GPU time
4. **Better performance** - 5-15% improvement over manual tuning

### Example Output:

```
[I 2024-10-29 14:30:00,123] Trial 0 finished with value: 0.0452
[I 2024-10-29 14:45:00,234] Trial 1 finished with value: 0.0389
[I 2024-10-29 14:50:00,345] Trial 2 pruned at epoch 8
[I 2024-10-29 14:55:00,456] Trial 3 pruned at epoch 6
[I 2024-10-29 15:15:00,567] Trial 4 finished with value: 0.0312 ⭐ Best!
```

---

## ⚠️ Important Notes

### 1. Validation Data Required
Optuna needs validation loss to make decisions. Ensure you have:
```python
valid_data_loader = get_val_loader(...)
```

### 2. GPU Memory
Batch size affects GPU memory. Start with `batch_size=1` for safety.

### 3. Time Commitment
- **MeshHead optimization**: ~5-8 hours for 15 trials
- **GazeGaussian optimization**: ~15-20 hours for 20 trials

### 4. Reproducibility
Set seeds but understand Optuna introduces randomness:
```python
study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
```

---

## 🎓 Best Practices

### 1. Start Small
```python
study.optimize(objective, n_trials=5)
```
Test with 5 trials first to ensure everything works.

### 2. Use Log Scale for Learning Rates
```python
lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
```

### 3. Save Checkpoints Per Trial
```python
checkpoint_path = f"trial_{trial.number}_best.pth"
torch.save(model.state_dict(), checkpoint_path)
```

### 4. Monitor in Real-Time
Use the dashboard or logging:
```python
import logging
optuna.logging.set_verbosity(optuna.logging.INFO)
```

### 5. Use Early Stopping Alongside Pruning
Combine Optuna pruning with your existing early stopping:
```python
if opt.early_stopping and no_improvement_for > patience:
    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory
**Solution:** Reduce batch_size search space to [1] only.

### Issue 2: All Trials Pruned
**Solution:** Increase `n_startup_trials` and `n_warmup_steps`.

### Issue 3: Optimization Slow
**Solution:** Use more aggressive pruner (HyperbandPruner).

### Issue 4: Results Not Improving
**Solution:** Expand hyperparameter search space.

---

## 📁 Files Created

After following this guide, you'll have:

1. `train_gazegaussian_optuna.py` - Optuna-integrated training script
2. `gazegaussian_study.db` - SQLite database with all trials
3. `optuna_results/` - Checkpoints and logs per trial
4. Colab notebook cells for running optimization

---

## 🎯 Next Steps

1. ✅ Install Optuna (Step 1)
2. ✅ Run MeshHead optimization (5 trials test)
3. ✅ Analyze MeshHead results
4. ✅ Run GazeGaussian optimization (20 trials)
5. ✅ Use best hyperparameters for final training
6. ✅ Document best configuration

---

## 📚 Additional Resources

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna GitHub](https://github.com/optuna/optuna)
- [Optuna Examples](https://github.com/optuna/optuna-examples)
- [PyTorch + Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)

---

**Ready to implement?** Let's proceed to create the training scripts!
