"""
Neural network models for PhyCRNet.
Includes PhyCRNet and its components (ConvLSTM, ResBlock).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhyCRNet(nn.Module):
    """Physics-informed Convolutional-Recurrent Network."""
    
    def __init__(self, ch=4, hidden=192, upscale=1, dropout_rate=0.2):
        super().__init__()
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ConvLSTM layers
        self.conv_lstm = DeepConvLSTM(hidden, hidden, num_layers=3, kernel_size=5, padding=2)
        
        # Residual block
        self.residual_block = ResidualBlock(hidden, hidden)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(hidden, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout_rate/2),
            nn.Conv2d(64, ch*(upscale**2), 3, padding=1),
            nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()
        )
        
        self._initialize_weights()
        self.up = upscale

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor [B×C×H×W]
            
        Returns:
            torch.Tensor: Output tensor [B×C×H×W]
        """
        # Encoding
        z = self.enc(x)                           # B×hidden×H×W
        
        # ConvLSTM processing
        z = z.unsqueeze(1)                        # B×1×hidden×H×W
        z, _ = self.conv_lstm(z)                  # B×1×hidden×H×W
        z = z.squeeze(1)                          # B×hidden×H×W
        
        # Residual processing
        z = self.residual_block(z)                # B×hidden×H×W
        
        # Decoding
        out = self.dec(z)                         # B×C×H×W
        return out

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class DeepConvLSTM(nn.Module):
    """Multi-layer ConvLSTM with attention mechanism."""
    
    def __init__(self, in_channels, hidden_channels, num_layers=3, kernel_size=5, padding=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # ConvLSTM layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(ConvLSTM(in_channels, hidden_channels, kernel_size, padding))
            else:
                self.cells.append(ConvLSTM(hidden_channels, hidden_channels, kernel_size, padding))
        
        # Attention mechanism
        self.attention = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x, hidden_states=None):
        """Forward pass through all ConvLSTM layers.
        
        Args:
            x (torch.Tensor): Input tensor [B×T×C×H×W]
            hidden_states (list): Initial hidden states for each layer
            
        Returns:
            tuple: (Output tensor, New hidden states)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
            
        output = x
        new_hidden_states = []
        
        # Process through each layer
        for i, cell in enumerate(self.cells):
            output, state = cell(output, hidden_states[i])
            new_hidden_states.append(state)
        
        # Apply attention to last time step
        last_output = output[:, -1]
        attention_weights = torch.sigmoid(self.attention(last_output))
        output_attended = last_output * attention_weights
        
        new_output = output.clone()
        new_output[:, -1] = output_attended
            
        return new_output, new_hidden_states

class ConvLSTM(nn.Module):
    """Enhanced Convolutional LSTM cell with peephole connections."""
    
    def __init__(self, in_channels, hidden_channels, kernel_size=5, padding=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Combined gates computation
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 
            hidden_channels * 4,  # 4 gates
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm([hidden_channels, 42, 42])
        
        # Peephole connections
        self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'w_c' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, hidden_state=None):
        """Forward pass with peephole connections and layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor [B×T×C×H×W]
            hidden_state (tuple): Previous (h, c) state
            
        Returns:
            tuple: (Output tensor, New state)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h_t, c_t = hidden_state
            
        # Output container
        output = torch.zeros(batch_size, seq_len, self.hidden_channels, height, width, device=x.device)
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t]
            combined = torch.cat([x_t, h_t], dim=1)
            
            # Calculate gates
            gates = self.conv(combined)
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            
            # Apply peephole connections
            i = torch.sigmoid(i + self.w_ci * c_t)
            f = torch.sigmoid(f + self.w_cf * c_t)
            g = torch.tanh(g)
            
            # Update cell state
            c_t_new = f * c_t + i * g
            c_t_new = torch.clamp(c_t_new, -10, 10)  # Prevent gradient explosion
            
            # Output gate with peephole
            o = torch.sigmoid(o + self.w_co * c_t_new)
            
            # Calculate hidden state
            h_t_new = o * torch.tanh(c_t_new)
            
            # Apply layer normalization
            if height == 42 and width == 42:
                h_t_new = self.layer_norm(h_t_new)
            
            # Store output
            output[:, t] = h_t_new
            
            # Update states
            h_t = h_t_new
            c_t = c_t_new
        
        return output, (h_t, c_t) 