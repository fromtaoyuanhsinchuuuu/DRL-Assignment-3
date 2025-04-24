# mario_qnet.py
import torch
import torch.nn as nn
import numpy as np

class MarioQNet(nn.Module):
    """
    CNN Q-Network for processing Mario game frames and predicting Q-values.
    Based on the architecture from the Nature DQN paper.
    """
    def __init__(self, state_shape, num_actions):
        """
        Initialize the Q-Network.

        Args:
            state_shape: Shape of the input state (expected to be (4, 84, 84))
            num_actions: Number of possible actions (expected to be 12)
        """
        super(MarioQNet, self).__init__()

        # print(f'state_shape {state_shape}, num_actions {num_actions}')
        # Convolutional layers
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

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        """
        Calculate the output size of the convolutional layers.

        Args:
            shape: Input shape from environment (frames, height, width, channels)

        Returns:
            Size of flattened features after convolution
        """
        # Create a dummy tensor with the correct shape for Conv2d
        # Conv2d expects (batch_size, channels, height, width)
        # For stacked frames, we treat frames as channels
        # shape is (frames, height, width, channels) from environment
        # We need to create a tensor of shape (batch_size=1, channels=frames, height, width)

        # Check if shape has 4 dimensions (frames, height, width, channels)
        if len(shape) == 4:
            # Create tensor with shape (batch_size=1, channels=frames, height, width)
            dummy_input = torch.zeros(1, shape[0], shape[1], shape[2])
        else:
            # If shape is (frames, height, width) without channels dimension
            dummy_input = torch.zeros(1, *shape)

        # print(f"Dummy input shape for conv out calculation: {dummy_input.shape}")
        o = self.conv(dummy_input)
        conv_out_size = int(np.prod(o.size()))
        # print(f"Calculated conv output size: {conv_out_size}")
        return conv_out_size

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor which could be in various formats:
               - (batch_size, frames, height, width, channels) from environment
               - (batch_size, frames, height, width) if channels dimension is already removed
               - (batch_size, channels=frames, height, width) if already in PyTorch format

        Returns:
            Q-values for each action
        """
        # Print input shape for debugging
        original_shape = x.shape

        # Handle different input formats
        if len(x.shape) == 5:  # (batch_size, frames, height, width, channels)
            # Remove the channels dimension (assuming it's 1) and permute to PyTorch format
            x = x.squeeze(-1)  # Remove channels dimension -> (batch_size, frames, height, width)
            # No need to permute as frames are already in the right position for channels

        # Normalize input (after shape adjustments)
        x = x / 255.0

        # Print transformed shape for debugging
        transformed_shape = x.shape
        if original_shape != transformed_shape:
            print(f"Transformed input shape from {original_shape} to {transformed_shape}")

        # Pass through convolutional layers
        conv_out = self.conv(x)
        # print(f'conv_out_shape {conv_out.shape}')

        # Flatten and pass through fully connected layers
        predict_q_value_arr = self.fc(conv_out.view(x.size(0), -1))
        # print(f'predict_q {predict_q_value_arr}')
        return predict_q_value_arr