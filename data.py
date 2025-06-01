"""
Dataset loader for PhyCRNet.
Handles loading and preprocessing of .mat data files.
"""

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

class MatDataset(Dataset):
    """Dataset class for loading .mat files with MAC grid data.
    
    The data is stored in a staggered MAC grid format:
    - u: staggered in x (41,42,300)
    - v: staggered in y (42,41,300)
    - p, θ: cell-centered (42,42,300)
    """
    
    def __init__(self, matfile, device='cpu'):
        """Initialize dataset from .mat file.
        
        Args:
            matfile (str): Path to .mat data file
            device (str): Device to store tensors on
        """
        d = sio.loadmat(matfile)
        
        # Extract data arrays
        u_raw = d['ustore']   # (41, 42, 300)  staggered in x
        v_raw = d['vstore']   # (42, 41, 300)  staggered in y
        p_raw = d['pstore']   # (42, 42, 300)  cell-centered
        t_raw = d['tstore']   # (42, 42, 300)  cell-centered
        
        # Transpose from (x,y,t) to (y,x,t) ordering
        u_raw = np.transpose(u_raw, (1, 0, 2))  # (41, 42, 300) → (42, 41, 300)
        v_raw = np.transpose(v_raw, (1, 0, 2))  # (42, 41, 300) → (41, 42, 300)
        p_raw = np.transpose(p_raw, (1, 0, 2))  # (42, 42, 300) → (42, 42, 300)
        t_raw = np.transpose(t_raw, (1, 0, 2))  # (42, 42, 300) → (42, 42, 300)
        
        # Calculate normalization factors
        self.u_mean, self.u_std = np.mean(u_raw), np.std(u_raw)
        self.v_mean, self.v_std = np.mean(v_raw), np.std(v_raw)
        self.p_mean, self.p_std = np.mean(p_raw), np.std(p_raw)
        self.t_mean, self.t_std = np.mean(t_raw), np.std(t_raw)
        
        # Normalize data
        u_norm = (u_raw - self.u_mean) / (self.u_std + 1e-8)
        v_norm = (v_raw - self.v_mean) / (self.v_std + 1e-8)
        p_norm = (p_raw - self.p_mean) / (self.p_std + 1e-8)
        t_norm = (t_raw - self.t_mean) / (self.t_std + 1e-8)
        
        # Convert to PyTorch tensors
        self.u = torch.tensor(u_norm, dtype=torch.float32, device=device)
        self.v = torch.tensor(v_norm, dtype=torch.float32, device=device)
        self.p = torch.tensor(p_norm, dtype=torch.float32, device=device)
        self.t = torch.tensor(t_norm, dtype=torch.float32, device=device)
        
        # Store normalization parameters as tensors
        self.norm_params = {
            'u': (torch.tensor(self.u_mean, device=device), torch.tensor(self.u_std, device=device)),
            'v': (torch.tensor(self.v_mean, device=device), torch.tensor(self.v_std, device=device)),
            'p': (torch.tensor(self.p_mean, device=device), torch.tensor(self.p_std, device=device)),
            't': (torch.tensor(self.t_mean, device=device), torch.tensor(self.t_std, device=device))
        }
        
        # Grid dimensions from pressure/temperature field
        self.ny, self.nx = self.p.shape[0], self.p.shape[1]  # 42, 42
        self.nt = self.p.shape[2]                            # 300
        
        # Number of usable time-steps (total - 1)
        self.T = self.nt - 1
        
        # Store physical parameters
        self.params = {
            'Ra': float(d['Ra'].squeeze()),  # Rayleigh number
            'Ha': float(d['Ha'].squeeze()),  # Hartmann number
            'Pr': float(d['Pr'].squeeze()),  # Prandtl number
            'Da': float(d['Da'].squeeze()),  # Darcy number
            'Rd': float(d['Rd'].squeeze()),  # Radiation parameter
            'Q': float(d['Q'].squeeze()),    # Heat source/sink parameter
            'dt': float(d['dt'].squeeze()) if 'dt' in d else 0.0001,  # Time step
        }
        
    def __len__(self):
        """Return number of time steps available for training."""
        return self.T
        
    def __getitem__(self, idx):
        """Get data for time steps idx and idx+1.
        
        Args:
            idx (int): Time index
            
        Returns:
            tuple: (Current state, Next state, Ground truth dict)
                - States are cell-centered [4×H×W] tensors
                - Ground truth is MAC grid format dictionary
        """
        # Convert staggered velocities to cell-centered at time idx
        u_c = 0.5*(self.u[:, :-1, idx] + self.u[:, 1:, idx])
        v_c = 0.5*(self.v[:-1, :, idx] + self.v[1:, :, idx])
        
        # Combine into cell-centered state at time idx
        f0 = torch.zeros(4, self.ny, self.nx, device=self.p.device)
        f0[0, :u_c.shape[0], :u_c.shape[1]] = u_c  # U
        f0[1, :v_c.shape[0], :v_c.shape[1]] = v_c  # V
        f0[2] = self.t[:,:,idx]  # θ
        f0[3] = self.p[:,:,idx]  # P
        
        # Same procedure for time idx+1
        u1_c = 0.5*(self.u[:, :-1, idx+1] + self.u[:, 1:, idx+1])
        v1_c = 0.5*(self.v[:-1, :, idx+1] + self.v[1:, :, idx+1])
        
        f1 = torch.zeros(4, self.ny, self.nx, device=self.p.device)
        f1[0, :u1_c.shape[0], :u1_c.shape[1]] = u1_c  # U
        f1[1, :v1_c.shape[0], :v1_c.shape[1]] = v1_c  # V
        f1[2] = self.t[:,:,idx+1]  # θ
        f1[3] = self.p[:,:,idx+1]  # P
        
        # Ground truth in original MAC staggered format
        gt = {
            'u': self.u[:,:,idx+1].unsqueeze(0),  # [1×H×W-1]
            'v': self.v[:,:,idx+1].unsqueeze(0),  # [1×H-1×W]
            'p': self.p[:,:,idx+1].unsqueeze(0),  # [1×H×W]
            't': self.t[:,:,idx+1].unsqueeze(0)   # [1×H×W]
        }
        
        return f0, f1, gt
    
    def get_params(self):
        """Return physical and normalization parameters"""
        return {
            **self.params,
            'norm_params': self.norm_params
        } 