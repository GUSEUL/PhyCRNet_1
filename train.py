"""
Training and visualization functions for PhyCRNet.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch.nn.functional as F

from utils import ensure_directory, enforce_bc
from losses import PhysicsLoss, DataLoss
from models import PhyCRNet
from data import MatDataset

def train(matfile, n_epoch=1000, bs=16, lr=1e-3, device='cuda', 
          physics_weight=1.0, data_weight=1.0, save_model=True, 
          plot_interval=10, weight_decay=1e-5):
    """Train PhyCRNet model.
    
    Args:
        matfile (str): Path to .mat data file
        n_epoch (int): Number of training epochs
        bs (int): Batch size (default: 16)
        lr (float): Learning rate (default: 1e-3)
        device (str): Device for training
        physics_weight (float): Final weight for physics loss
        data_weight (float): Weight for data loss
        save_model (bool): Whether to save model checkpoints
        plot_interval (int): Interval for plotting validation results
        weight_decay (float): L2 regularization strength
        
    Returns:
        nn.Module: Trained model
    """
    # Create directories
    image_dir = ensure_directory("phycrnet_image")
    model_dir = ensure_directory("phycrnet_model")
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Load dataset
    print(f"Loading data from {matfile}...")
    ds = MatDataset(matfile, device)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    
    # Get physical parameters
    params = ds.get_params()
    print(f"Physical parameters: Ra={params['Ra']}, Ha={params['Ha']}, " + 
          f"Pr={params['Pr']}, Da={params['Da']}, Rd={params['Rd']}, Q={params['Q']}")
    
    # Grid parameters
    Nx = ds.nx
    dx = 1.0/(Nx-1)
    half_idx = Nx//2
    
    # Setup model and losses
    model = PhyCRNet(hidden=192, dropout_rate=0.2).to(device)
    p_loss = PhysicsLoss(dx, params).to(device)
    d_loss = DataLoss(weight_u=1.0, weight_v=1.0, weight_p=0.1, weight_t=1.0).to(device)
    
    # Optimizer with modified settings
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler with modified settings
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        steps_per_epoch=len(dl),
        epochs=n_epoch,
        pct_start=0.2,        # Warm-up for 20% of training
        div_factor=10.0,      # Initial lr = max_lr/10
        final_div_factor=1e4, # Final lr = initial_lr/10000
        three_phase=True,     # Use three-phase schedule
        anneal_strategy='cos' # Use cosine annealing
    )
    
    # Modified physics loss weight schedule:
    # Phase 1 (0-30%): Focus on data loss (very small physics weight)
    # Phase 2 (30-50%): Gradually introduce physics loss
    # Phase 3 (50-100%): Continue increasing physics weight to final value
    physics_weight_schedule = np.zeros(n_epoch)
    
    # Phase 1: First 30% epochs - very small physics weight
    phase1_end = int(0.3 * n_epoch)
    physics_weight_schedule[:phase1_end] = physics_weight * 1e-6
    
    # Phase 2: Next 20% epochs - exponential increase
    phase2_end = int(0.5 * n_epoch)
    phase2_epochs = np.arange(phase1_end, phase2_end)
    physics_weight_schedule[phase1_end:phase2_end] = np.logspace(
        -6, -2, phase2_end - phase1_end
    ) * physics_weight
    
    # Phase 3: Remaining epochs - linear increase to final weight
    remaining_epochs = np.arange(phase2_end, n_epoch)
    physics_weight_schedule[phase2_end:] = np.linspace(
        physics_weight * 1e-2, physics_weight, len(remaining_epochs)
    )
    
    print("\nTraining schedule:")
    print(f"Phase 1 (epochs 0-{phase1_end-1}): Focus on data loss")
    print(f"Phase 2 (epochs {phase1_end}-{phase2_end-1}): Gradual physics introduction")
    print(f"Phase 3 (epochs {phase2_end}-{n_epoch-1}): Increase to full physics")
    
    # Gradient clipping value
    grad_clip_value = 1.0
    
    # Training loop
    best_loss = float('inf')
    loss_history = {
        'physics': [], 'data': [], 'total': [], 'lr': [],
        'u': [], 'v': [], 'p': [], 't': []  # Individual data losses
    }
    
    print(f"Starting training for {n_epoch} epochs...")
    for epoch in range(n_epoch):
        model.train()
        running_p = 0  # Physics loss
        running_d = 0  # Total data loss
        running_total = 0  # Total loss
        running_u = 0  # U velocity loss
        running_v = 0  # V velocity loss
        running_p = 0  # Pressure loss
        running_t = 0  # Temperature loss
        
        # Current physics loss weight
        curr_physics_weight = physics_weight_schedule[epoch]
        
        for f0, f1, gt in dl:
            optimizer.zero_grad()
            
            # Get timestep
            dt = params['dt']
            
            # Forward pass
            delta = model(f0)
            pred1 = f0 + dt*delta
            pred1 = enforce_bc(pred1, half_idx)
            
            # Compute losses
            loss_p = p_loss(f0, pred1, dt)
            loss_d = d_loss(pred1, gt)
            
            # Get individual data losses for monitoring
            U, V, T, P = torch.chunk(pred1, 4, 1)
            
            # U velocity loss
            u_staggered = d_loss.center_to_u(U)
            min_h = min(u_staggered.shape[2], gt['u'].shape[2])
            min_w = min(u_staggered.shape[3], gt['u'].shape[3])
            loss_u = F.mse_loss(u_staggered[:,:,:min_h,:min_w], 
                               gt['u'][:,:,:min_h,:min_w])
            
            # V velocity loss
            v_staggered = d_loss.center_to_v(V)
            min_h = min(v_staggered.shape[2], gt['v'].shape[2])
            min_w = min(v_staggered.shape[3], gt['v'].shape[3])
            loss_v = F.mse_loss(v_staggered[:,:,:min_h,:min_w], 
                               gt['v'][:,:,:min_h,:min_w])
            
            # Pressure loss
            min_h = min(P.shape[2], gt['p'].shape[2])
            min_w = min(P.shape[3], gt['p'].shape[3])
            loss_p_val = F.mse_loss(P[:,:,:min_h,:min_w], 
                                   gt['p'][:,:,:min_h,:min_w])
            
            # Temperature loss
            min_h = min(T.shape[2], gt['t'].shape[2])
            min_w = min(T.shape[3], gt['t'].shape[3])
            loss_t = F.mse_loss(T[:,:,:min_h,:min_w], 
                               gt['t'][:,:,:min_h,:min_w])
            
            # Combined loss with weights
            weighted_physics_loss = curr_physics_weight * loss_p
            weighted_data_loss = data_weight * loss_d
            loss = weighted_physics_loss + weighted_data_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses (store raw losses for monitoring)
            running_p += loss_p.item()
            running_d += loss_d.item()
            running_total += weighted_physics_loss.item() + weighted_data_loss.item()  # Store weighted sum
            running_u += loss_u.item()
            running_v += loss_v.item()
            running_p += loss_p_val.item()
            running_t += loss_t.item()
        
        # Average losses
        avg_p_loss = running_p / len(dl)
        avg_d_loss = running_d / len(dl)
        avg_total_loss = running_total / len(dl)
        avg_u_loss = running_u / len(dl)
        avg_v_loss = running_v / len(dl)
        avg_p_loss = running_p / len(dl)
        avg_t_loss = running_t / len(dl)
        
        # Record history
        loss_history['physics'].append(avg_p_loss)
        loss_history['data'].append(avg_d_loss)
        loss_history['total'].append(avg_total_loss)
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_history['u'].append(avg_u_loss)
        loss_history['v'].append(avg_v_loss)
        loss_history['p'].append(avg_p_loss)
        loss_history['t'].append(avg_t_loss)
        
        # Save best model
        if avg_total_loss < best_loss and save_model:
            best_loss = avg_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'params': params,
            }, os.path.join(model_dir, 'phycrnet_best.pth'))
            print(f"Saved best model with loss {best_loss:.2e}")
        
        # Print progress with detailed losses
        print(f"[{epoch:03d}] physics {avg_p_loss:.2e} (w={curr_physics_weight:.2f}) | " + 
              f"data {avg_d_loss:.2e} (U={avg_u_loss:.2e}, V={avg_v_loss:.2e}, " + 
              f"P={avg_p_loss:.2e}, θ={avg_t_loss:.2e}) | " + 
              f"total {avg_total_loss:.2e} | lr {optimizer.param_groups[0]['lr']:.1e}")
        
        # Periodic validation
        if epoch % plot_interval == 0 and epoch > 0:
            validate_and_visualize(model, ds, device, epoch, image_dir)
            
    # Save final model
    if save_model:
        torch.save({
            'epoch': n_epoch-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'params': params,
        }, os.path.join(model_dir, 'phycrnet_final.pth'))
        
    # Plot training curves with individual losses
    plot_training_curves(loss_history, image_dir)
    
    return model

def validate_and_visualize(model, dataset, device, epoch, image_dir='phycrnet_image'):
    """Validate model and create visualization.
    
    Args:
        model (nn.Module): PhyCRNet model
        dataset (MatDataset): Dataset instance
        device (str): Device to run on
        epoch (int): Current epoch number
        image_dir (str): Directory to save plots
    """
    model.eval()
    
    # Get first sample
    f0, f1_gt, gt = dataset[0]
    f0 = f0.unsqueeze(0).to(device)
    
    # Parameters
    params = dataset.get_params()
    dt = params['dt']
    
    # Forward pass
    with torch.no_grad():
        delta = model(f0)
        pred = f0 + dt*delta
        pred = enforce_bc(pred, dataset.p.shape[0]//2)
    
    # Convert to numpy
    u_pred = pred[0, 0].cpu().numpy()
    v_pred = pred[0, 1].cpu().numpy()
    t_pred = pred[0, 2].cpu().numpy()
    p_pred = pred[0, 3].cpu().numpy()
    
    u_gt = f1_gt[0].cpu().numpy()
    v_gt = f1_gt[1].cpu().numpy()
    t_gt = f1_gt[2].cpu().numpy()
    p_gt = f1_gt[3].cpu().numpy()
    
    # Create figure
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot predictions
    im1 = axs[0, 0].imshow(u_pred, cmap='viridis')
    axs[0, 0].set_title('U Prediction')
    plt.colorbar(im1, ax=axs[0, 0])
    
    im2 = axs[0, 1].imshow(v_pred, cmap='viridis')
    axs[0, 1].set_title('V Prediction')
    plt.colorbar(im2, ax=axs[0, 1])
    
    im3 = axs[0, 2].imshow(t_pred, cmap='hot')
    axs[0, 2].set_title('θ Prediction')
    plt.colorbar(im3, ax=axs[0, 2])
    
    im4 = axs[0, 3].imshow(p_pred, cmap='viridis')
    axs[0, 3].set_title('P Prediction')
    plt.colorbar(im4, ax=axs[0, 3])
    
    # Plot ground truth
    im5 = axs[1, 0].imshow(u_gt, cmap='viridis')
    axs[1, 0].set_title('U Ground Truth')
    plt.colorbar(im5, ax=axs[1, 0])
    
    im6 = axs[1, 1].imshow(v_gt, cmap='viridis')
    axs[1, 1].set_title('V Ground Truth')
    plt.colorbar(im6, ax=axs[1, 1])
    
    im7 = axs[1, 2].imshow(t_gt, cmap='hot')
    axs[1, 2].set_title('θ Ground Truth')
    plt.colorbar(im7, ax=axs[1, 2])
    
    im8 = axs[1, 3].imshow(p_gt, cmap='viridis')
    axs[1, 3].set_title('P Ground Truth')
    plt.colorbar(im8, ax=axs[1, 3])
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(image_dir, f'validation_epoch_{epoch:03d}.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Validation image saved to {save_path}")

def plot_training_curves(loss_history, image_dir):
    """Plot training loss curves with individual components."""
    plt.figure(figsize=(12, 12))
    
    # Plot main losses
    plt.subplot(3, 1, 1)
    plt.semilogy(loss_history['physics'], label='Physics Loss')
    plt.semilogy(loss_history['data'], label='Data Loss')
    plt.semilogy(loss_history['total'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True, which="both", ls="--")
    
    # Plot individual data losses
    plt.subplot(3, 1, 2)
    plt.semilogy(loss_history['u'], label='U Loss')
    plt.semilogy(loss_history['v'], label='V Loss')
    plt.semilogy(loss_history['p'], label='P Loss')
    plt.semilogy(loss_history['t'], label='θ Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Individual Data Losses')
    plt.grid(True, which="both", ls="--")
    
    # Plot learning rate
    plt.subplot(3, 1, 3)
    plt.plot(loss_history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'training_losses.png'))
    plt.close()

def predict_sequence(model, init_state, n_steps, dt, half_idx, device='cuda'):
    """Generate sequence of predictions.
    
    Args:
        model (nn.Module): PhyCRNet model
        init_state (torch.Tensor): Initial state [1×4×H×W]
        n_steps (int): Number of steps to predict
        dt (float): Time step size
        half_idx (int): Index for mid-height
        device (str): Device to run on
        
    Returns:
        list: Predicted states
    """
    model.eval()
    current_state = init_state.clone().to(device)
    predictions = [current_state.cpu().squeeze(0)]
    
    with torch.no_grad():
        for _ in range(n_steps):
            delta = model(current_state)
            next_state = current_state + dt*delta
            next_state = enforce_bc(next_state, half_idx)
            predictions.append(next_state.cpu().squeeze(0))
            current_state = next_state
    
    return predictions

def create_animation(predictions, variable_idx, filename, cmap='viridis', 
                    image_dir='phycrnet_image'):
    """Create animation of predictions.
    
    Args:
        predictions (list): List of predicted states
        variable_idx (int): Index of variable to animate (0=U, 1=V, 2=θ, 3=P)
        filename (str): Output filename
        cmap (str): Colormap to use
        image_dir (str): Directory to save animation
    """
    # Extract frames
    frames = [pred[variable_idx].numpy() for pred in predictions]
    
    # Value range for consistent colormap
    vmin = min(frame.min() for frame in frames)
    vmax = max(frame.max() for frame in frames)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize plot
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax)
    
    # Variable names
    var_names = ['U (x-velocity)', 'V (y-velocity)', 'θ (temperature)', 'P (pressure)']
    title = ax.set_title(f"{var_names[variable_idx]} - t=0")
    
    def update(i):
        title.set_text(f"{var_names[variable_idx]} - t={i*dt:.4f}")
        im.set_array(frames[i])
        return [im, title]
    
    # Create animation
    dt = 0.01
    ani = FuncAnimation(fig, update, frames=len(frames), 
                       blit=True, interval=200)
    
    # Save animation
    save_path = os.path.join(image_dir, filename)
    ani.save(save_path, writer='pillow', dpi=100)
    plt.close()
    
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    # Configuration
    config = {
        'matfile': "Rd/Ra_10^5_Rd_1.7.mat",
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'n_epoch': 1000,
        'batch_size': 16,      # Increased batch size
        'learning_rate': 1e-3, # Increased learning rate
        'physics_weight': 1.0,
        'data_weight': 1.0,
        'save_model': True,
        'plot_interval': 50,
    }
    
    print(f"Using device: {config['device']}")
    print(f"Training configuration: {config}")
    
    try:
        # Train model
        model = train(
            config['matfile'], 
            n_epoch=config['n_epoch'],
            bs=config['batch_size'],
            lr=config['learning_rate'],
            device=config['device'],
            physics_weight=config['physics_weight'],
            data_weight=config['data_weight'],
            plot_interval=config['plot_interval']
        )
        
        # Load dataset for prediction
        ds = MatDataset(config['matfile'], device='cpu')
        params = ds.get_params()
        dt = params['dt']
        half_idx = ds.nx//2
        
        print("Generating predictions...")
        
        # Get initial state
        init_state, _, _ = ds[0]
        init_state = init_state.unsqueeze(0).to(config['device'])
        
        # Predict sequence
        n_steps = 20
        predictions = predict_sequence(
            model, init_state, n_steps, dt, half_idx, config['device']
        )
        
        # Create animations
        print("Creating animations...")
        image_dir = ensure_directory("phycrnet_image")
        create_animation(predictions, 0, 'u_prediction.gif', cmap='viridis', image_dir=image_dir)
        create_animation(predictions, 1, 'v_prediction.gif', cmap='viridis', image_dir=image_dir)
        create_animation(predictions, 2, 'theta_prediction.gif', cmap='hot', image_dir=image_dir)
        create_animation(predictions, 3, 'p_prediction.gif', cmap='viridis', image_dir=image_dir)
        
        print(f"Training and prediction complete. Animations saved to {image_dir}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc() 