# dueling_nstep_agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from dueling_qnet import DuelingMarioQNet
from per_buffer import PrioritizedReplayBuffer
import config

class Dueling_NSTEP_DDQN_Agent:
    """
    Dueling Double DQN agent with Prioritized Experience Replay and N-step bootstrapping for Super Mario Bros.
    This agent combines multiple advanced techniques:
    1. Dueling Network Architecture - Separate value and advantage streams
    2. Double DQN - Decoupling action selection and evaluation
    3. Prioritized Experience Replay - Sample important transitions more frequently
    4. N-step Bootstrapping - Learn from multi-step returns
    """
    def __init__(self, state_shape, action_size, seed=42,
                 buffer_capacity=50000, batch_size=32, gamma=0.99,
                 lr=5e-5, tau=1e-3, per_alpha=0.6, per_beta_start=0.4,
                 per_beta_frames=1000000, epsilon_start=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9999,
                 n_step=3, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Dueling DDQN+PER agent with N-step bootstrapping.

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
            n_step: Number of steps for N-step bootstrapping
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
        self.n_step = n_step

        # Epsilon parameters for exploration
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Dueling Q networks with Noisy Networks if enabled
        self.q_net = DuelingMarioQNet(state_shape, action_size, use_noisy_net=config.USE_NOISY_NET).to(device)
        self.target_net = DuelingMarioQNet(state_shape, action_size, use_noisy_net=config.USE_NOISY_NET).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Initialize replay buffer with PER and N-step
        beta_increment = (1.0 - per_beta_start) / per_beta_frames
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            seed=seed,
            n_step=n_step,
            gamma=gamma
        )
        self.replay_buffer.alpha = per_alpha
        self.replay_buffer.beta = per_beta_start
        self.replay_buffer.beta_increment_per_sampling = beta_increment

        # Initialize time step counter
        self.t_step = 0

        # 移除不必要的打印
        # print(f"Dueling DDQN+PER Agent with {n_step}-step bootstrapping initialized on device: {device}")

    def get_action(self, state, use_epsilon=True):
        """
        Select an action using either epsilon-greedy policy or Noisy Networks.
        Standard Noisy Net usage: Action selection is deterministic based on mean weights.
        Exploration is driven by learning the noise parameters during training.

        Args:
            state: Current state (numpy array) with shape (frames, height, width, channels)
                  Expected to be in format (4, 84, 84, 1) for stacked frames
            use_epsilon: Whether to use epsilon-greedy policy (only applies if USE_NOISY_NET is False)

        Returns:
            Selected action
        """
        # If NOT using Noisy Networks, use epsilon-greedy
        if not config.USE_NOISY_NET:
            if use_epsilon and random.random() < self.epsilon:
                return random.randrange(self.action_size)
        # Even with Noisy Networks, occasionally force random actions to break out of local optima
        # This is a hybrid approach that combines Noisy Networks with occasional epsilon-greedy
        elif use_epsilon and random.random() < 0.01:  # 1% chance of random action
            random_action = random.randrange(self.action_size)
            print(f"Forcing random exploration: action {random_action}")
            return random_action

        # Remember current network mode
        was_training = self.q_net.training

        # Set network to evaluation mode for deterministic action selection
        self.q_net.eval()

        try:
            # Convert state to tensor
            # Handle LazyFrames from FrameStack wrapper
            if hasattr(state, '__array__'):
                state = np.array(state)

            # Validate state shape
            if len(state.shape) == 4:
                # Expected shape: (frames, height, width, channels)
                assert state.shape[0] == 4, f"Expected 4 stacked frames, got {state.shape[0]}"
                assert state.shape[3] == 1, f"Expected channel dimension to be 1, got {state.shape[3]}"

            state = torch.from_numpy(state).float()

            # Reshape from (frames, height, width, channels) to (batch_size, frames, height, width)
            # First, remove the channel dimension (which is 1)
            state = state.squeeze(-1)
            # Then add batch dimension
            state = state.unsqueeze(0)
            # Normalize
            state = state / 255.0  # Normalization is done here in the agent, not in the network
            # Move to device
            state = state.to(self.device)

            # Get Q values from the network (in eval mode, uses mean weights)
            with torch.no_grad():
                q_values = self.q_net(state)

                # 打印 Q 值分布信息 (每 100 步打印一次)
                if self.t_step % 100 == 0:
                    q_numpy = q_values.cpu().numpy()[0]  # 转换为 numpy 数组
                    max_q_idx = np.argmax(q_numpy)

                    print(f"\nQ-values distribution at step {self.t_step}:")
                    print(f"Min Q: {q_numpy.min():.4f}, Max Q: {q_numpy.max():.4f}, Mean Q: {q_numpy.mean():.4f}")
                    print(f"Q-values variance: {q_numpy.var():.4f}, std: {q_numpy.std():.4f}")
                    print(f"Selected action: {max_q_idx}, with Q-value: {q_numpy[max_q_idx]:.4f}")

                    # 打印所有动作的 Q 值
                    action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B",
                                   "A", "LEFT", "LEFT+A", "LEFT+B", "LEFT+A+B",
                                   "DOWN", "UP"]
                    print("All Q-values:")
                    for i, (action, q) in enumerate(zip(action_names, q_numpy)):
                        marker = " *" if i == max_q_idx else ""
                        print(f"  {action:10s}: {q:.4f}{marker}")
                    print()  # 空行分隔

            return q_values.argmax().item()
        finally:
            # Restore network mode
            if was_training:
                self.q_net.train()
            # If it was in eval mode, it stays in eval mode

    def update_target_network(self, update_type='soft'):
        """
        Update the target network.

        Args:
            update_type: 'soft' for soft update, 'hard' for hard update, or 'none' for no update
        """
        if update_type == 'soft':
            for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        elif update_type == 'hard':  # hard update
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            raise ValueError(f"Invalid update type: {update_type}")

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
        # Note: We don't need to normalize states here because the PER buffer
        # handles normalization during sampling (states = states / 255.0)
        # The buffer stores the raw uint8 states to save memory
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Increment time step
        self.t_step += 1

        # Train if enough samples, we've reached the learning start step, and it's time to train
        if (len(self.replay_buffer) > self.batch_size and
            self.t_step >= config.LEARNING_START_STEP and
            self.t_step % train_freq == 0):

            # Print a message when we start training for the first time
            if self.t_step == config.LEARNING_START_STEP and (self.t_step - config.LEARNING_START_STEP) % train_freq == 0:
                print(f"Starting training at step {self.t_step} (learning_start_step={config.LEARNING_START_STEP})")

            self.train()

    def train(self):
        """
        Train the agent using a batch of experiences from the replay buffer.
        Uses N-step returns for more efficient learning.
        """
        # Set Q-network to training mode
        self.q_net.train()

        # Reset noise for Noisy Networks if enabled
        if config.USE_NOISY_NET:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        # Sample experiences from replay buffer
        experiences, indices, is_weights = self.replay_buffer.sample(self.batch_size)

        # Check if we have valid experiences
        if experiences is None:
            print("No valid experiences for training")
            return

        states, actions, rewards, next_states, dones, discounts = experiences

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        discounts = discounts.to(self.device)  # Actual discount factors for each experience
        is_weights = is_weights.to(self.device)  # Already a PyTorch tensor from the buffer

        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Get best actions from online network
            best_actions_next = self.q_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate those actions using target network
            Q_targets_next = self.target_net(next_states).gather(1, best_actions_next)

            # Compute Q targets for current states
            # Use the actual discount factors from the buffer
            # The rewards already include the discounted sum of N-step rewards
            # Note: For terminal states (dones=1), discounts will be 0.0
            Q_targets = rewards + discounts * Q_targets_next      # discounts 已含終止資訊

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

        # Check gradients (example for the first conv layer)
        if self.t_step % 1000 == 0:  # Print every 1000 steps
            grad_mean = self.q_net.conv[0].weight.grad.abs().mean().item()
            q_mean = Q_expected.mean().item()
            q_min = Q_expected.min().item()
            q_max = Q_expected.max().item()
            print(f"Step {self.t_step}: Mean abs grad (conv0): {grad_mean:.6f}")
            print(f"Step {self.t_step}: Q values - Mean: {q_mean:.4f}, Min: {q_min:.4f}, Max: {q_max:.4f}")
            if grad_mean < 1e-6:
                print("Warning: Gradients are very small or zero!")

            # Check weights before and after update
            conv0_weight_before = self.q_net.conv[0].weight.data.clone()

        # Gradient clipping to prevent exploding gradients
        # Increased from 1.0 to 10.0 for Atari/Mario environments as recommended
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Check if weights actually changed after optimizer step
        if self.t_step % 1000 == 0:  # Same frequency as gradient check
            if 'conv0_weight_before' in locals():
                conv0_weight_after = self.q_net.conv[0].weight.data
                weight_diff = (conv0_weight_after - conv0_weight_before).abs().mean().item()
                print(f"Step {self.t_step}: Weight change after update: {weight_diff:.8f}")
                if weight_diff < 1e-8:
                    print("Warning: Weights barely changed after update!")

        # Print detailed training information every 100 steps
        if self.t_step % 100 == 0:
            # Calculate statistics for Q values and TD errors
            q_mean = Q_expected.mean().item()
            q_min = Q_expected.min().item()
            q_max = Q_expected.max().item()
            q_std = Q_expected.std().item()

            td_mean = np.mean(td_errors)
            td_max = np.max(td_errors)

            # Print detailed training information
            print(f"\n===== Training Statistics at Step {self.t_step} =====")
            print(f"Loss: {loss.item():.6f}")
            print(f"Q-values - Mean: {q_mean:.4f}, Min: {q_min:.4f}, Max: {q_max:.4f}, Std: {q_std:.4f}")
            print(f"TD Errors - Mean: {td_mean:.4f}, Max: {td_max:.4f}")

            # Print weight update information if available
            if 'weight_diff' in locals():
                print(f"Weight Change: {weight_diff:.8f}")

            # Print beta value (importance sampling)
            print(f"PER Beta: {self.replay_buffer.beta:.4f}")
            print("================================================\n")

        # Update priorities in the replay buffer
        # Add small constant epsilon to prevent priorities from becoming zero
        for idx, error in zip(indices, td_errors):
            self.replay_buffer.update_priority(idx, float(error) + 1e-6)

    def save(self, path):
        """
        Save the agent's state including replay buffer.

        Args:
            path: Path to save the model
        """
        # Save model, optimizer, and training state
        model_state = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step,
            'replay_buffer_beta': self.replay_buffer.beta,
        }

        # Save the main checkpoint
        torch.save(model_state, path)

        # Save replay buffer separately (it can be large)
        buffer_path = path.replace('.pth', '_buffer.pth')
        try:
            # Export replay buffer state
            buffer_state = {
                'memory': self.replay_buffer.memory,
                'tree': self.replay_buffer.tree.tree,
                'size': self.replay_buffer.size,
                'position': self.replay_buffer.tree.write_pos,
                'max_priority': self.replay_buffer.max_priority,
                'discounts': self.replay_buffer.discounts if hasattr(self.replay_buffer, 'discounts') else {},
                'n_step_buffer': self.replay_buffer.n_step_buffer,
            }
            torch.save(buffer_state, buffer_path)
            # 移除不必要的打印
        except Exception:
            # 移除不必要的打印
            pass

    def load(self, path):
        """
        Load the agent's state including replay buffer if available.

        Args:
            path: Path to load the model from
        """
        # Load model checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load model and optimizer state
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load training state
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.t_step = checkpoint.get('t_step', 0)

        # Load replay buffer beta if available
        if 'replay_buffer_beta' in checkpoint:
            self.replay_buffer.beta = checkpoint['replay_buffer_beta']

        # Try to load replay buffer
        buffer_path = path.replace('.pth', '_buffer.pth')
        if os.path.exists(buffer_path):
            try:
                buffer_state = torch.load(buffer_path, map_location=self.device)

                # Restore replay buffer state
                self.replay_buffer.memory = buffer_state['memory']
                self.replay_buffer.tree.tree = buffer_state['tree']
                self.replay_buffer.size = buffer_state['size']
                self.replay_buffer.tree.write_pos = buffer_state['position']
                self.replay_buffer.max_priority = buffer_state['max_priority']

                # Restore discounts dictionary if available
                if 'discounts' in buffer_state:
                    self.replay_buffer.discounts = buffer_state['discounts']

                # Restore n-step buffer if available
                if 'n_step_buffer' in buffer_state:
                    self.replay_buffer.n_step_buffer = buffer_state['n_step_buffer']

                # 移除不必要的打印
                pass
            except Exception:
                # 移除不必要的打印
                pass
