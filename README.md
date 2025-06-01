# PhyCRNet: Physics-Constrained Recurrent Neural Network

PhyCRNet is a deep learning model that incorporates physical laws to simulate and predict natural convection phenomena. Based on Convolutional LSTM architecture, it integrates physical laws into the loss function to achieve more accurate and physically consistent predictions.

## Key Features

- **Physics-Informed Learning**: Integration of physical laws (continuity, momentum, and energy equations) into the loss function
- **Hybrid Architecture**: Combination of ConvLSTM and Residual Network
- **Multi-Phase Training**: Gradual progression from data-driven to physics-informed learning
- **MAC Grid Support**: Support for Staggered MAC (Marker-And-Cell) grid system

## Project Structure

```
.
├── train.py         # Training and validation code
├── data.py          # Data loader and preprocessing
├── losses.py        # Loss function implementation
├── models.py        # Neural network architecture
└── utils.py         # Utility functions
```

## Installation

1. Install required packages:
```bash
pip install torch numpy matplotlib scipy
```

2. Dataset preparation:
- Place `.mat` format data files in the `Rd` directory
- Data should include the following variables:
  - `ustore`: x-direction velocity (MAC staggered)
  - `vstore`: y-direction velocity (MAC staggered)
  - `pstore`: pressure (cell-centered)
  - `tstore`: temperature (cell-centered)
  - Physical parameters: `Ra`, `Ha`, `Pr`, `Da`, `Rd`, `Q`

## Execution

1. Start training:
```bash
python train.py
```

Default settings:
- Batch size: 16
- Learning rate: 1e-3
- Epochs: 1000
- Device: CUDA (if available)

2. Training phases:
- Phase 1 (0-30%): Data-driven learning
- Phase 2 (30-50%): Gradual introduction of physical constraints
- Phase 3 (50-100%): Full integration of physical laws

## Results

Generated files during training:
- `phycrnet_image/`: Validation images and animations
  - `validation_epoch_*.png`: Validation results for each epoch
  - `training_losses.png`: Training curves
  - `*_prediction.gif`: Prediction result animations

- `phycrnet_model/`: Model checkpoints
  - `phycrnet_best.pth`: Best performing model
  - `phycrnet_final.pth`: Final trained model

## Key Parameter Settings

Main configurations in `train.py`:
```python
config = {
    'matfile': "Rd/Ra_10^5_Rd_1.7.mat",  # Data file path
    'n_epoch': 1000,                      # Total epochs
    'batch_size': 16,                     # Batch size
    'learning_rate': 1e-3,                # Learning rate
    'physics_weight': 1.0,                # Physics loss weight
    'data_weight': 1.0,                   # Data loss weight
    'plot_interval': 50,                  # Visualization interval
}
```

## Notes

- GPU usage is recommended (reduces training time)
- Reduce batch size if experiencing memory issues
- Adjust learning rate if training becomes unstable
- Balance between physical laws and data is crucial (adjust physics_weight and data_weight)

## Physical Equations

The model incorporates the following governing equations:

1. Continuity equation:
```
∂U/∂X + ∂V/∂Y = 0
```

2. X-momentum equation:
```
∂U/∂t + U∂U/∂X + V∂U/∂Y = -∂P/∂X + Pr(∂²U/∂X² + ∂²U/∂Y²) - (Pr/Da)U
```

3. Y-momentum equation:
```
∂V/∂t + U∂V/∂X + V∂V/∂Y = -∂P/∂Y + Pr(∂²V/∂X² + ∂²V/∂Y²) + Ra Pr θ - Ha² Pr V - (Pr/Da)V
```

4. Energy equation:
```
∂θ/∂t + U∂θ/∂X + V∂θ/∂Y = (1+4Rd/3)(∂²θ/∂X² + ∂²θ/∂Y²) + Q θ
```

Where:
- `Ra`: Rayleigh number
- `Ha`: Hartmann number
- `Pr`: Prandtl number
- `Da`: Darcy number
- `Rd`: Radiation parameter
- `Q`: Heat source/sink parameter 
