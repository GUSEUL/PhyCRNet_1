"""
Loss functions for PhyCRNet.
Includes physics-based and data-driven losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fd_kernels

class PhysicsLoss(nn.Module):
    """Physics-based loss using PDE residuals.
    
    Implements the following equations:
    1. Continuity: ∂U/∂X + ∂V/∂Y = 0
    2. X-momentum: ∂U/∂t + U∂U/∂X + V∂U/∂Y = -∂P/∂X + Pr(∂²U/∂X² + ∂²U/∂Y²) - (Pr/Da)U
    3. Y-momentum: ∂V/∂t + U∂V/∂X + V∂V/∂Y = -∂P/∂Y + Pr(∂²V/∂X² + ∂²V/∂Y²) + Ra Pr θ - Ha² Pr V - (Pr/Da)V
    4. Energy: ∂θ/∂t + U∂θ/∂X + V∂θ/∂Y = (1+4Rd/3)(∂²θ/∂X² + ∂²θ/∂Y²) + Q θ
    """
    def __init__(self, dx, params):
        super().__init__()
        # Physical parameters
        self.Pr = params['Pr']  
        self.Ra = params['Ra']   
        self.Ha = params['Ha']  
        self.Da = params['Da']  
        self.Rd = params['Rd'] 
        self.Q = params['Q']
        
        # Normalization parameters
        norm_params = params['norm_params']
        self.u_mean, self.u_std = norm_params['u']
        self.v_mean, self.v_std = norm_params['v']
        self.p_mean, self.p_std = norm_params['p']
        self.t_mean, self.t_std = norm_params['t']
        
        # Register finite difference kernels
        kdx, kdy, klap = fd_kernels(dx)
        self.register_buffer('kdx', kdx)
        self.register_buffer('kdy', kdy)
        self.register_buffer('klap', klap)
        
        # Loss scaling factors (to balance different terms)
        self.momentum_scale = 1.0
        self.continuity_scale = 1.0
        self.energy_scale = 1.0

    def denormalize(self, U, V, T, P):
        """Denormalize variables to physical units"""
        U = U * self.u_std + self.u_mean
        V = V * self.v_std + self.v_mean
        T = T * self.t_std + self.t_mean
        P = P * self.p_std + self.p_mean
        return U, V, T, P

    def normalize(self, U, V, T, P):
        """Normalize variables back to normalized units"""
        U = (U - self.u_mean) / (self.u_std + 1e-8)
        V = (V - self.v_mean) / (self.v_std + 1e-8)
        T = (T - self.t_mean) / (self.t_std + 1e-8)
        P = (P - self.p_mean) / (self.p_std + 1e-8)
        return U, V, T, P

    def normalize_derivatives(self, dUdt, dVdt, dTdt, dPdx, dPdy):
        """Normalize derivatives back to normalized units"""
        dUdt = dUdt / (self.u_std + 1e-8)
        dVdt = dVdt / (self.v_std + 1e-8)
        dTdt = dTdt / (self.t_std + 1e-8)
        dPdx = dPdx / (self.p_std + 1e-8)
        dPdy = dPdy / (self.p_std + 1e-8)
        return dUdt, dVdt, dTdt, dPdx, dPdy

    def forward(self, f_now, f_next, dt):
        """Compute physics-based loss.
        
        Args:
            f_now (torch.Tensor): Current state [B×4×H×W]
            f_next (torch.Tensor): Next state [B×4×H×W]
            dt (float): Time step size
            
        Returns:
            torch.Tensor: Mean squared PDE residuals
        """
        # Split into components
        U0, V0, T0, P0 = torch.chunk(f_now, 4, 1)
        U1, V1, T1, P1 = torch.chunk(f_next, 4, 1)
        
        # Denormalize for physical calculations
        U0, V0, T0, P0 = self.denormalize(U0, V0, T0, P0)
        U1, V1, T1, P1 = self.denormalize(U1, V1, T1, P1)

        # Temporal derivatives
        dUdt = (U1-U0)/dt
        dVdt = (V1-V0)/dt
        dTdt = (T1-T0)/dt

        # Spatial derivatives
        dUdx = F.conv2d(U1, self.kdx, padding=1)
        dUdy = F.conv2d(U1, self.kdy, padding=1)
        dVdx = F.conv2d(V1, self.kdx, padding=1)
        dVdy = F.conv2d(V1, self.kdy, padding=1)
        dTdx = F.conv2d(T1, self.kdx, padding=1)
        dTdy = F.conv2d(T1, self.kdy, padding=1)
        dPdx = F.conv2d(P1, self.kdx, padding=1)
        dPdy = F.conv2d(P1, self.kdy, padding=1)
        
        # Laplacian terms
        lapU = F.conv2d(U1, self.klap, padding=1)
        lapV = F.conv2d(V1, self.klap, padding=1)
        lapT = F.conv2d(T1, self.klap, padding=1)

        # Normalize all terms back to normalized units
        dUdt, dVdt, dTdt, _, _ = self.normalize_derivatives(dUdt, dVdt, dTdt, dPdx, dPdy)
        U1, V1, T1, P1 = self.normalize(U1, V1, T1, P1)
        dUdx = dUdx / (self.u_std + 1e-8)
        dUdy = dUdy / (self.u_std + 1e-8)
        dVdx = dVdx / (self.v_std + 1e-8)
        dVdy = dVdy / (self.v_std + 1e-8)
        dTdx = dTdx / (self.t_std + 1e-8)
        dTdy = dTdy / (self.t_std + 1e-8)
        dPdx = dPdx / (self.p_std + 1e-8)
        dPdy = dPdy / (self.p_std + 1e-8)
        lapU = lapU / (self.u_std + 1e-8)
        lapV = lapV / (self.v_std + 1e-8)
        lapT = lapT / (self.t_std + 1e-8)

        # Continuity equation: ∂U/∂X + ∂V/∂Y = 0
        r1 = (dUdx + dVdy) * self.continuity_scale

        # X-momentum
        r2 = (dUdt + U1*dUdx + V1*dUdy + dPdx - self.Pr*lapU - (self.Pr/self.Da)*U1) * self.momentum_scale

        # Y-momentum
        r3 = (dVdt + U1*dVdx + V1*dVdy + dPdy - self.Pr*lapV - self.Ra*self.Pr*T1 + 
              (self.Ha**2)*self.Pr*V1 - (self.Pr/self.Da)*V1) * self.momentum_scale

        # Energy
        r4 = (dTdt + U1*dTdx + V1*dTdy - (1+4*self.Rd/3)*lapT - self.Q*T1) * self.energy_scale

        # Combine residuals with balanced weights
        res = torch.cat([r1, r2, r3, r4], 1)
        return torch.mean(res**2)

class DataLoss(nn.Module):
    """Data-driven loss comparing predictions with ground truth on MAC grid."""
    
    def __init__(self, weight_u=1.0, weight_v=1.0, weight_p=0.1, weight_t=1.0):
        super().__init__()
        self.wu = weight_u
        self.wv = weight_v
        self.wp = weight_p  # Reduced weight for pressure
        self.wt = weight_t

    @staticmethod
    def center_to_u(pred):
        """Convert cell-centered U to staggered grid (in x)."""
        return 0.5 * (pred[:,:,:,:-1] + pred[:,:,:,1:])

    @staticmethod
    def center_to_v(pred):
        """Convert cell-centered V to staggered grid (in y)."""
        return 0.5 * (pred[:,:,:-1,:] + pred[:,:,1:,:])

    def forward(self, pred, gt):
        """Compute data-driven loss.
        
        Args:
            pred (torch.Tensor): Predicted fields [B×4×H×W]
            gt (dict): Ground truth fields on MAC grid
            
        Returns:
            torch.Tensor: Weighted sum of MSE losses
        """
        U, V, T, P = torch.chunk(pred, 4, 1)
        total_loss = 0.0
        
        # U velocity loss
        if gt['u'] is not None:
            u_staggered = self.center_to_u(U)
            min_h = min(u_staggered.shape[2], gt['u'].shape[2])
            min_w = min(u_staggered.shape[3], gt['u'].shape[3])
            loss_u = F.mse_loss(u_staggered[:,:,:min_h,:min_w], 
                               gt['u'][:,:,:min_h,:min_w])
            total_loss += self.wu * loss_u
            
        # V velocity loss
        if gt['v'] is not None:
            v_staggered = self.center_to_v(V)
            min_h = min(v_staggered.shape[2], gt['v'].shape[2])
            min_w = min(v_staggered.shape[3], gt['v'].shape[3])
            loss_v = F.mse_loss(v_staggered[:,:,:min_h,:min_w], 
                               gt['v'][:,:,:min_h,:min_w])
            total_loss += self.wv * loss_v
            
        # Pressure loss (with reduced weight)
        if gt['p'] is not None:
            min_h = min(P.shape[2], gt['p'].shape[2])
            min_w = min(P.shape[3], gt['p'].shape[3])
            loss_p = F.mse_loss(P[:,:,:min_h,:min_w], 
                               gt['p'][:,:,:min_h,:min_w])
            total_loss += self.wp * loss_p
                
        # Temperature loss
        if gt['t'] is not None:
            min_h = min(T.shape[2], gt['t'].shape[2])
            min_w = min(T.shape[3], gt['t'].shape[3])
            loss_t = F.mse_loss(T[:,:,:min_h,:min_w], 
                               gt['t'][:,:,:min_h,:min_w])
            total_loss += self.wt * loss_t
        
        return total_loss 