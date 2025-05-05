# dueling_qnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration in Deep RL.

    Implements the Noisy Networks for Exploration technique from the paper:
    "Noisy Networks for Exploration" (https://arxiv.org/abs/1706.10295)

    This layer replaces the standard linear layer with a version that adds
    parametric noise to the weights, which is learned with gradient descent.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        std_init: float = 1.2,  # Increased from 0.8 to 1.2 for more exploration
    ):
        super(NoisyLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init
        self.bias_flag = bias  # Store whether to use bias

        # Mean weights
        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # Standard deviation of weights
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # Register epsilon buffer for weights
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )

        if bias:
            # Mean bias
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                )
            )

            # Standard deviation of bias
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                )
            )

            # Register epsilon buffer for bias
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            # If not using bias, set all bias-related parameters to None
            self.bias_mu = None
            self.bias_sigma = None
            self.register_buffer("bias_epsilon", None)

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize the parameters of the layer."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        if self.bias_flag:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Reset the noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))

        if self.bias_flag:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise according to the factorized Gaussian noise approach."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise if in training mode."""
        if self.training:
            # During training, use noisy weights
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = None
            if self.bias_flag:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only the mean weights
            weight = self.weight_mu
            bias = self.bias_mu if self.bias_flag else None

        return F.linear(input, weight, bias)

class DuelingMarioQNet(nn.Module):
    """
    Dueling CNN Q-Network for processing Mario game frames and predicting Q-values.
    Based on the Dueling DQN architecture from Wang et al. (2016).

    Can use Noisy Linear layers for exploration instead of epsilon-greedy.
    """
    def __init__(self, state_shape, num_actions, use_noisy_net=False):
        """
        Initialize the Dueling Q-Network.

        Args:
            state_shape: Shape of the input state (expected to be (4, 84, 84))
            num_actions: Number of possible actions (expected to be 12)
            use_noisy_net: Whether to use Noisy Linear layers for exploration
        """
        super(DuelingMarioQNet, self).__init__()

        self.use_noisy_net = use_noisy_net

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
        # Common feature layer - use standard Linear layer here
        self.feature_layer = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        # Value stream (estimates the state value V(s))
        if use_noisy_net:
            self.value_stream = nn.Sequential(
                NoisyLinear(512, 256),
                nn.ReLU(),
                NoisyLinear(256, 1)  # Outputs a single value
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # Outputs a single value
            )

        # Advantage stream (estimates the advantage for each action A(s,a))
        if use_noisy_net:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(512, 256),
                nn.ReLU(),
                NoisyLinear(256, num_actions)  # Outputs one value per action
            )
        else:
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

        NoisyLinear layers have their own initialization in their reset_parameters method,
        so we only need to initialize the standard layers here.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # NoisyLinear is now a direct subclass of nn.Module, not nn.Linear,
                # so we don't need to check for it here

                # 區分「值頭 / 優勢頭最後一層」與其他層
                last_adv_linear = None
                last_val_linear = None

                if not self.use_noisy_net:
                    last_adv_linear = self.advantage_stream[-1]
                    last_val_linear = self.value_stream[-1]

                if m is last_adv_linear or m is last_val_linear:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear', a=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_noise(self):
        """
        Reset noise for all NoisyLinear layers in the network.
        This should be called at the beginning of each episode or step
        to sample a new set of noisy weights.
        """
        if not self.use_noisy_net:
            return

        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
