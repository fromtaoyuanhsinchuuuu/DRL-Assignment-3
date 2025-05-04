# dueling_qnet.py
import torch
import torch.nn as nn
import numpy as np

class DuelingMarioQNet(nn.Module):
    """
    Dueling CNN Q-Network for processing Mario game frames and predicting Q-values.
    Based on the Dueling DQN architecture from Wang et al. (2016).
    """
    def __init__(self, state_shape, num_actions):
        """
        Initialize the Dueling Q-Network.

        Args:
            state_shape: Shape of the input state (expected to be (4, 84, 84))
            num_actions: Number of possible actions (expected to be 12)
        """
        super(DuelingMarioQNet, self).__init__()

        # Convolutional layers (same as in the original network)
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the flattened features after convolution
        conv_out_size = self._get_conv_out(state_shape)

        # Dueling architecture: split into value and advantage streams
        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        # Value stream (estimates the state value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs a single value
        )

        # Advantage stream (estimates the advantage for each action A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)  # Outputs one value per action
        )

        # Initialize weights using Kaiming normal initialization
        self._initialize_weights()

    def _get_conv_out(self, shape):
        """
        Calculate the output size of the convolutional layers.

        Args:
            shape: Input shape from environment, could be:
                  - (frames, height, width, channels)
                  - (frames, height, width)
                  - (channels, height, width)

        Returns:
            Size of flattened features after convolution
        """
        # Create a dummy tensor with the correct shape for Conv2d
        # Conv2d expects (batch_size, channels, height, width)

        # Handle different input shapes more robustly
        if len(shape) == 4:  # (frames, height, width, channels)
            # Create tensor with shape (batch_size=1, channels=frames, height, width)
            dummy_input = torch.zeros(1, shape[0], shape[-3], shape[-2])
        elif len(shape) == 3:
            if shape[0] in {1,3,4}:     # frames/channels first
                dummy_input = torch.zeros(1, shape[0], shape[1], shape[2])
            elif shape[-1] in {1,3,4}:  # channels last
                dummy_input = torch.zeros(1, shape[-1], shape[0], shape[1])
            else:                       # fallback
                dummy_input = torch.zeros(1, *shape)
        else:
            # Fallback for other shapes
            dummy_input = torch.zeros(1, *shape)

        o = self.conv(dummy_input)
        conv_out_size = int(np.prod(o.size()))
        return conv_out_size

    def forward(self, x):
        """
        Forward pass through the Dueling network.

        Args:
            x: Input tensor which could be in various formats:
               - (batch_size, frames, height, width, channels) from environment
               - (batch_size, frames, height, width) if channels dimension is already removed
               - (batch_size, channels=frames, height, width) if already in PyTorch format
               - (batch_size, height, width, channels) for single frame with channels last

        Returns:
            Q-values for each action
        """
        # Handle different input formats
        if len(x.shape) == 5:  # (batch_size, frames, height, width, channels)
            # Remove the channels dimension (assuming it's 1) and permute to PyTorch format
            x = x.squeeze(-1)  # Remove channels dimension -> (batch_size, frames, height, width)
            # No need to permute as frames are already in the right position for channels
        elif len(x.shape) == 4 and x.shape[-1] <= 4:  # (batch_size, height, width, channels)
            # This is likely (batch, H, W, C) format with channels last
            x = x.permute(0, 3, 1, 2)  # Convert to (batch, C, H, W)

        # Note: Normalization is now handled in the agent's preprocessing
        # to avoid double normalization. Uncomment the line below if your
        # agent does NOT normalize the input.
        # x = x / 255.0

        # Pass through convolutional layers
        conv_out = self.conv(x)

        # Flatten the convolutional output
        features = conv_out.view(conv_out.size(0), -1)

        # Pass through the common feature layer
        features = self.feature_layer(features)

        # Split into value and advantage streams
        value = self.value_stream(features)  # Shape: (batch_size, 1)
        advantages = self.advantage_stream(features)  # Shape: (batch_size, num_actions)

        # Combine value and advantages using the dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This formula ensures that the advantages sum to zero for each state
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming normal initialization for better training stability.
        This is consistent with the original DQN implementation.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # 區分「值頭 / 優勢頭最後一層」與其他層
                last_adv_linear = self.advantage_stream[-1]
                last_val_linear = self.value_stream[-1]
                if m is last_adv_linear or m is last_val_linear:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear', a=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
