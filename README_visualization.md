# PlaNet Trajectory Visualization

## Overview

This repository now includes checkpoint saving during training and trajectory visualization capabilities.

## Checkpoint Saving

During training, the system automatically saves:

### **Best Model Checkpoint** (`checkpoints/best_model.pth`)
- Saved whenever evaluation performance improves
- Contains: model weights, optimizer state, best return, epoch info

### **Periodic Checkpoints** (`checkpoints/checkpoint_epoch_X.pth`)
- Saved every 100 epochs (every 2nd evaluation by default)
- Contains: full training state for resuming

### **Final Model** (`trained_rssm_dmc_walker.pth`)
- Saved at end of training
- Uploaded to Wandb as artifact

## Trajectory Visualization

### Quick Start

```bash
# Use default best model checkpoint
python visualize_trajectories.py

# Specify a specific checkpoint
python visualize_trajectories.py --checkpoint checkpoints/checkpoint_epoch_200.pth

# Run for longer trajectory
python visualize_trajectories.py --steps 200 --output-dir my_visualizations
```

### What Gets Visualized

1. **Observation Comparison**
   - Side-by-side real vs imagined observations at key timesteps
   - Shows how well the RSSM reconstructs visual observations

2. **Reward Analysis**
   - Real vs predicted rewards over time
   - Cumulative reward comparison
   - Reward prediction accuracy metrics

3. **Action and State Analysis**
   - Actions taken by the controller over time
   - State representation magnitude evolution

### Output Files

- `observations_comparison.png`: Visual comparison of real vs imagined frames
- `trajectory_metrics.png`: Plots of rewards, actions, and states
- Console output with summary statistics

### Example Usage Scenarios

```bash
# Evaluate your best model
python visualize_trajectories.py --checkpoint checkpoints/best_model.pth

# Compare different training checkpoints
python visualize_trajectories.py --checkpoint checkpoints/checkpoint_epoch_100.pth --output-dir epoch_100_viz
python visualize_trajectories.py --checkpoint checkpoints/checkpoint_epoch_500.pth --output-dir epoch_500_viz

# Generate longer trajectory for detailed analysis
python visualize_trajectories.py --steps 300 --output-dir long_trajectory

# Use GPU if available
python visualize_trajectories.py --device cuda
```

## Understanding the Visualizations

### **Good Model Indicators:**
- Imagined observations closely match real ones
- Predicted rewards track real rewards accurately
- Smooth, purposeful actions
- Stable state representations

### **Poor Model Indicators:**
- Blurry or unrealistic imagined observations
- Large reward prediction errors
- Erratic or unstable actions
- Rapidly changing state magnitudes

## Monitoring Training Progress

1. **Check Wandb Dashboard**: Monitor `eval_avg_return` for actual performance
2. **Checkpoint Files**: See which epochs produced the best models
3. **Visualization**: Run periodic visualizations to see qualitative improvements

## Tips

- Run visualizations every few hundred epochs to track learning progress
- Compare checkpoints from different stages of training
- Pay attention to reward prediction accuracy - it's crucial for good planning
- Use the visualization to debug why certain controllers perform poorly