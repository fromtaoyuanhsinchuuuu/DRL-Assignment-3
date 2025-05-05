# dueling_ddqn_per_agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from dueling_qnet import DuelingMarioQNet
from per_buffer import PrioritizedReplayBuffer
import config

class Dueling_DDQN_PER_Agent:
    """
    Dueling Double DQN agent with Prioritized Experience Replay for Super Mario Bros.
    """
    def __init__(self, state_shape, action_size, seed=42,
                 buffer_capacity=50000, batch_size=32, gamma=0.99,
                 lr=5e-5, tau=1e-3, per_alpha=0.6, per_beta_start=0.4,
                 per_beta_frames=1000000, epsilon_start=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9999,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Dueling DDQN+PER agent.

        Args:
            state_shape: Shape of the state (expected to be (4, 84, 84))
            action_size: Number of possible actions
            seed: Random seed for reproducibility
            buffer_capacity: Capacity of the replay buffer
            batch_size: Size of training batches
            gamma: Discount factor
            lr: Learning rate
            tau: Soft update parameter
            per_alpha: PER alpha parameter (prioritization amount)
            per_beta_start: Initial value of beta for importance sampling
            per_beta_frames: Number of frames over which to anneal beta to 1.0
            epsilon_start: Initial epsilon for exploration
            epsilon_min: Minimum epsilon value
            epsilon_decay: Epsilon decay rate
            device: Device to run the model on (cuda or cpu)
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.device = device

        # Epsilon parameters for exploration
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q networks with Dueling architecture and Noisy Networks if enabled
        self.use_noisy_net = config.USE_NOISY_NET if hasattr(config, 'USE_NOISY_NET') else False
        self.q_net = DuelingMarioQNet(state_shape, action_size, use_noisy_net=self.use_noisy_net).to(device)
        self.target_net = DuelingMarioQNet(state_shape, action_size, use_noisy_net=self.use_noisy_net).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Initialize replay buffer with PER
        beta_increment = (1.0 - per_beta_start) / per_beta_frames
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, seed)
        self.replay_buffer.alpha = per_alpha
        self.replay_buffer.beta = per_beta_start
        self.replay_buffer.beta_increment_per_sampling = beta_increment

        # Initialize time step counter
        self.t_step = 0

        print(f"Dueling DDQN+PER Agent initialized on device: {device}")
        if self.use_noisy_net:
            print("Using Noisy Networks for exploration (epsilon-greedy disabled)")

    def get_action(self, state, use_epsilon=False):
        """
        Select an action using epsilon-greedy policy or Noisy Networks.

        Args:
            state: Current state (numpy array) with shape (frames, height, width, channels)
            use_epsilon: Whether to use epsilon-greedy policy (only applies if not using Noisy Networks)

        Returns:
            Selected action
        """
        # If using Noisy Networks, we don't need epsilon-greedy exploration
        # as the noise in the network parameters provides exploration
        if not self.use_noisy_net:
            # Epsilon-greedy exploration only if not using Noisy Networks
            if use_epsilon and random.random() < self.epsilon:
                return random.randrange(self.action_size)

        # Convert state to tensor
        # Handle LazyFrames from FrameStack wrapper
        if hasattr(state, '__array__'):
            state = np.array(state)
        state = torch.from_numpy(state).float()

        # Reshape from (frames, height, width, channels) to (batch_size, frames, height, width)
        # First, remove the channel dimension (which is 1)
        state = state.squeeze(-1)
        # Then add batch dimension
        state = state.unsqueeze(0)
        # Normalize
        state = state / 255.0
        # Move to device
        state = state.to(self.device)

        # Set network to evaluation mode for action selection
        self.q_net.eval()

        # Get Q values from the network
        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax().item()

    def update_target_network(self, update_type='soft'):
        """
        Update the target network.

        Args:
            update_type: 'soft' for soft update, 'hard' for hard update
        """
        if update_type == 'soft':
            for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:  # hard update
            self.target_net.load_state_dict(self.q_net.state_dict())

    def step(self, state, action, reward, next_state, done, train_freq=4):
        """
        Store experience in replay buffer and trigger training if needed.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            train_freq: How often to train (in steps)
        """
        # Store experience in replay buffer
        # Note: The PER buffer already handles normalization during sampling
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Increment time step
        self.t_step += 1

        # Decay epsilon (only if not using Noisy Networks)
        if not self.use_noisy_net and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Train if enough samples and it's time to train
        if len(self.replay_buffer) > self.batch_size and self.t_step % train_freq == 0:
            self.train()

    def train(self):
        """
        Train the agent using a batch of experiences from the replay buffer.
        """
        # Set network to training mode
        self.q_net.train()

        # Reset noise for Noisy Networks if enabled
        if self.use_noisy_net:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        # Sample experiences from replay buffer
        experiences, indices, is_weights = self.replay_buffer.sample(self.batch_size)

        # Check if we have valid experiences
        if experiences is None:
            print("Skipping training step due to insufficient valid experiences.")
            return

        states, actions, rewards, next_states, dones = experiences

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)  # Already a PyTorch tensor from the buffer

        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Get best actions from online network
            best_actions_next = self.q_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate those actions using target network
            Q_targets_next = self.target_net(next_states).gather(1, best_actions_next)
            # Compute Q targets for current states
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from online network
        Q_expected = self.q_net(states).gather(1, actions)

        # Compute TD errors for updating priorities
        td_errors = torch.abs(Q_targets - Q_expected).detach().cpu().numpy()

        # Compute weighted loss using Huber Loss (smooth_l1_loss) instead of MSE
        # Huber Loss is more robust to outliers than MSE
        loss = (F.smooth_l1_loss(Q_expected, Q_targets, reduction='none') * is_weights).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in the replay buffer
        for idx, error in zip(indices, td_errors):
            self.replay_buffer.update_priority(idx, error)

    def save(self, path):
        """
        Save the model.

        Args:
            path: Path to save the model
        """
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        """
        Load the model.

        Args:
            path: Path to load the model from
        """
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
