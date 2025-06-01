"""
Utility functions for PhyCRNet implementation.
Contains directory management and finite difference kernels.
"""

import os
import torch

def ensure_directory(directory):
    """Create directory if it doesn't exist.
    
    Args:
        directory (str): Path to directory
        
    Returns:
        str: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def fd_kernels(dx):
    """Create finite difference kernels for spatial derivatives.
    
    Args:
        dx (float): Grid spacing
        
    Returns:
        tuple: (d/dx, d/dy, Laplacian) kernels shaped (1,1,3,3)
    """
    # Central difference for x-derivative
    k_dx = torch.tensor([[0, 0, 0],
                        [-1, 0, 1],
                        [0, 0, 0]], dtype=torch.float32) / (2*dx)
    
    # Central difference for y-derivative (transpose of x)
    k_dy = k_dx.t().clone()
    
    # 5-point stencil for Laplacian
    k_lap = torch.tensor([[0, 1, 0],
                         [1,-4, 1],
                         [0, 1, 0]], dtype=torch.float32) / (dx**2)
    
    # Add batch and channel dimensions
    return k_dx[None,None], k_dy[None,None], k_lap[None,None]

def enforce_bc(fields, half_idx):
    """Enforce boundary conditions on the fields.
    
    Args:
        fields (torch.Tensor): Input fields [B×4×H×W] (U,V,θ,P)
        half_idx (int): Index for mid-height
        
    Returns:
        torch.Tensor: Fields with enforced boundary conditions
    """
    U, V, T, P = torch.chunk(fields, 4, 1)
    
    # Create new tensors instead of modifying in-place
    U_new = U.clone()
    V_new = V.clone()
    T_new = T.clone()
    P_new = P.clone()

    # No-slip boundary conditions for velocity
    # Bottom wall (Y=0): U = V = 0
    U_new[:,:,0,:] = 0
    V_new[:,:,0,:] = 0
    
    # Top wall (Y=1): U = V = 0
    U_new[:,:,-1,:] = 0
    V_new[:,:,-1,:] = 0
    
    # Left wall (X=0): U = V = 0
    U_new[:,:,:,0] = 0
    V_new[:,:,:,0] = 0
    
    # Right wall (X=1): U = V = 0
    U_new[:,:,:,-1] = 0
    V_new[:,:,:,-1] = 0

    # Temperature boundary conditions
    # Left wall (X=0): θ = 1 for Y ≤ 0.5, θ = 0 for Y > 0.5
    left_wall_mask = torch.zeros_like(T_new[:,:,:,0])
    left_wall_mask[:,:,:half_idx] = 1.0
    T_new[:,:,:,0] = left_wall_mask
    
    # Right wall (X=1): θ = 1 for Y ≤ 0.5, θ = 0 for Y > 0.5
    right_wall_mask = torch.zeros_like(T_new[:,:,:,-1])
    right_wall_mask[:,:,:half_idx] = 1.0
    T_new[:,:,:,-1] = right_wall_mask
    
    # Bottom wall (Y=0): θ = 0
    T_new[:,:,0,:] = 0.0
    
    # Top wall (Y=1): ∂θ/∂Y = 0 (Neumann BC)
    T_new[:,:,-1,:] = T_new[:,:,-2,:]

    return torch.cat([U_new, V_new, T_new, P_new], 1) 