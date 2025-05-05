import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from dueling_qnet import DuelingMarioQNet
import config

# Debug flag - set to True to enable debug messages, False to disable
DEBUG = False

# Define the COMPLEX_MOVEMENT action space for Super Mario Bros
# This is the same as gym_super_mario_bros.actions.COMPLEX_MOVEMENT
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]



# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that uses trained Dueling DQN model to select actions."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = "cpu"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEBUG:
            print(f'device {self.device}')

        # Define state shape for Mario environment (4 stacked frames of 84x84 grayscale images)
        self.state_shape = (4, 84, 84, 1)

        # Flag to track if we're using noisy networks
        self.use_noisy_net = False

        # Find the model path first to determine if it's a noisy network model
        self.model_path = self._find_latest_model()

        # Check if the model path contains "noisy" to determine if we should use Noisy Networks
        if self.model_path and "noisy" in self.model_path.lower():
            print("Detected Noisy Network model. Initializing network with Noisy layers...")
            self.use_noisy_net = True

        # Initialize Dueling Q-network with appropriate noisy network setting
        self.q_net = DuelingMarioQNet(self.state_shape, self.action_space.n, use_noisy_net=self.use_noisy_net).to(self.device)

        # Try to load the model weights
        self.model_loaded = False
        if self.model_path:
            try:
                # Load the model weights
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)

                # Load the model with strict=False to allow for architecture differences
                self.q_net.load_state_dict(state_dict, strict=False)
                self.model_loaded = True
                print(f"Loaded model weights from {self.model_path}")
                print(f"Using Noisy Networks: {self.use_noisy_net}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No model weights found. Using random actions.")

        # Counter for tracking actions
        self.action_count = 0

    def _find_latest_model(self):
        """Find the latest model weights file in the checkpoints directory."""
        # First, check for the noisy network model file (highest priority)
        if os.path.exists("mario_dueling_nstep_noisy_qnet.pth"):
            if DEBUG:
                print("Found noisy network model file: mario_dueling_nstep_noisy_qnet.pth")
            return "mario_dueling_nstep_noisy_qnet.pth"

        # Then, check for the regular n-step model file
        if os.path.exists("mario_dueling_nstep_qnet.pth"):
            if DEBUG:
                print("Found n-step model file: mario_dueling_nstep_qnet.pth")
            return "mario_dueling_nstep_qnet.pth"

        # Check for n-step checkpoint files
        if os.path.exists("dueling_nstep_checkpoints"):
            if DEBUG:
                print("Checking dueling_nstep_checkpoints directory...")
            # Find all Q-network weight files
            qnet_files = [f for f in os.listdir("dueling_nstep_checkpoints") if f.startswith("mario_dueling_nstep_qnet_ep") and f.endswith(".pth")]

            if qnet_files:
                # Extract episode numbers and find the latest one
                latest_file = None
                latest_episode = -1

                for file in qnet_files:
                    try:
                        # Extract episode number from filename
                        episode = int(file.split("_ep")[1].split(".")[0])
                        if episode > latest_episode:
                            latest_episode = episode
                            latest_file = file
                    except:
                        continue

                if latest_file:
                    if DEBUG:
                        print(f"Found latest n-step model: {latest_file}")
                    return os.path.join("dueling_nstep_checkpoints", latest_file)

            # Also check for full checkpoint files which contain the state dict
            checkpoint_files = [f for f in os.listdir("dueling_nstep_checkpoints") if f.startswith("mario_dueling_nstep_ep") and f.endswith(".pth")]

            if checkpoint_files:
                if DEBUG:
                    print("Found full n-step checkpoint files. Will try to extract weights.")
                # Sort by episode number to find the latest
                latest_file = None
                latest_episode = -1

                for file in checkpoint_files:
                    try:
                        # Extract episode number from filename
                        episode = int(file.split("ep")[1].split(".")[0])
                        if episode > latest_episode:
                            latest_episode = episode
                            latest_file = file
                    except:
                        continue

                if latest_file:
                    full_path = os.path.join("dueling_nstep_checkpoints", latest_file)
                    if DEBUG:
                        print(f"Found latest n-step checkpoint: {full_path}")

                    # Try to extract q_net_state_dict from the checkpoint
                    try:
                        checkpoint = torch.load(full_path, map_location=self.device)
                        if 'q_net_state_dict' in checkpoint:
                            if DEBUG:
                                print("Extracting q_net_state_dict from checkpoint...")
                            # Save the extracted state dict to a temporary file
                            temp_path = "temp_extracted_qnet.pth"
                            torch.save(checkpoint['q_net_state_dict'], temp_path)
                            return temp_path
                    except Exception as e:
                        if DEBUG:
                            print(f"Error extracting state dict from checkpoint: {e}")

        # Fall back to regular dueling model if n-step not found
        if DEBUG:
            print("No n-step model found. Falling back to regular dueling model.")
        if os.path.exists("mario_dueling_ddqn_per_qnet.pth"):
            return "mario_dueling_ddqn_per_qnet.pth"

        # Check for regular dueling checkpoint files
        if os.path.exists("dueling_checkpoints"):
            # Find all Q-network weight files
            qnet_files = [f for f in os.listdir("dueling_checkpoints") if f.startswith("mario_dueling_qnet_ep") and f.endswith(".pth")]

            if qnet_files:
                # Extract episode numbers and find the latest one
                latest_file = None
                latest_episode = -1

                for file in qnet_files:
                    try:
                        # Extract episode number from filename (format: mario_dueling_qnet_ep{episode}.pth)
                        episode = int(file.split("_ep")[1].split(".")[0])
                        if episode > latest_episode:
                            latest_episode = episode
                            latest_file = file
                    except:
                        continue

                if latest_file:
                    return os.path.join("dueling_checkpoints", latest_file)

        # If no dueling model found, try to load regular model
        if os.path.exists("mario_ddqn_per_qnet.pth"):
            if DEBUG:
                print("No dueling model found. Trying to load regular model.")
            return "mario_ddqn_per_qnet.pth"

        # Check regular checkpoints
        if os.path.exists("checkpoints"):
            # Find all Q-network weight files
            qnet_files = [f for f in os.listdir("checkpoints") if f.startswith("mario_qnet_ep") and f.endswith(".pth")]

            if qnet_files:
                # Extract episode numbers and find the latest one
                latest_file = None
                latest_episode = -1

                for file in qnet_files:
                    try:
                        # Extract episode number from filename (format: mario_qnet_ep{episode}.pth)
                        episode = int(file.split("_ep")[1].split(".")[0])
                        if episode > latest_episode:
                            latest_episode = episode
                            latest_file = file
                    except:
                        continue

                if latest_file:
                    if DEBUG:
                        print("Loading regular model into dueling architecture.")
                    return os.path.join("checkpoints", latest_file)

        return None

    def get_action(self, state, use_epsilon=False):
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
        # If no model is loaded, use random actions
        if not self.model_loaded:
            return random.randrange(self.action_space.n)

        # If NOT using Noisy Networks, use epsilon-greedy
        if not self.use_noisy_net:
            if use_epsilon and random.random() < 0.01:  # Small chance of random action
                return random.randrange(self.action_space.n)

        # Remember current network mode
        was_training = self.q_net.training

        # Set network to evaluation mode for deterministic action selection
        self.q_net.eval()

        try:
            # Convert state to tensor
            # Handle LazyFrames from FrameStack wrapper
            if hasattr(state, '__array__'):
                state = np.array(state)

            # Store original shape for debugging
            orig_shape = state.shape

            # Handle raw RGB frames from evaluation environment (240, 256, 3)
            if len(state.shape) == 3 and state.shape[0] == 240 and state.shape[1] == 256 and state.shape[2] == 3:
                # Only print the warning once every 100 steps to reduce spam
                if DEBUG and self.action_count % 100 == 0:
                    print(f"Handling raw RGB frame from evaluation environment: {state.shape}")

                # Create a tensor with the correct shape for PyTorch (batch, channels, height, width)
                processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                # Convert to tensor
                rgb_tensor = torch.from_numpy(state).float()

                # Convert RGB to PyTorch format (channels first) and add batch dimension
                rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 240, 256]

                # Resize to [1, 3, 84, 84]
                rgb_tensor = torch.nn.functional.interpolate(rgb_tensor, size=(84, 84), mode='bilinear', align_corners=False)

                # Convert RGB to grayscale: 0.299 * R + 0.587 * G + 0.114 * B
                gray_tensor = 0.299 * rgb_tensor[:, 0:1] + 0.587 * rgb_tensor[:, 1:2] + 0.114 * rgb_tensor[:, 2:3]

                # Fill all 4 channels with the same grayscale image (frame stacking)
                for i in range(4):
                    processed_state[:, i] = gray_tensor.squeeze(1)

                # Normalize
                processed_state = processed_state / 255.0

                # Use the processed state
                state_tensor = processed_state

            # Handle the standard training environment format (4, 84, 84, 1)
            elif len(state.shape) == 4 and state.shape[0] == 4 and state.shape[3] == 1:
                # Validate state shape
                if DEBUG and self.action_count % 100 == 0:
                    print(f"Processing standard training format: {state.shape}")

                # Reshape from (frames, height, width, channels) to (batch_size, frames, height, width)
                # First, remove the channel dimension (which is 1)
                state = torch.from_numpy(state).float()
                state = state.squeeze(-1)
                # Then add batch dimension
                state = state.unsqueeze(0)
                # Normalize
                state = state / 255.0

                # Use the processed state
                state_tensor = state

            # Handle other formats with a warning
            else:
                if DEBUG and self.action_count % 100 == 0:
                    print(f"Warning: Unexpected observation shape: {orig_shape}")

                # Try to convert to tensor and normalize
                try:
                    state_tensor = torch.from_numpy(state).float()

                    # If it's a single RGB frame
                    if len(state.shape) == 3 and state.shape[-1] == 3:
                        # Create tensor with correct shape
                        processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                        # Convert RGB to grayscale and resize
                        rgb_state = state_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                        if rgb_state.shape[2] != 84 or rgb_state.shape[3] != 84:
                            rgb_state = torch.nn.functional.interpolate(rgb_state, size=(84, 84), mode='bilinear', align_corners=False)

                        # Convert to grayscale
                        gray_state = 0.299 * rgb_state[:, 0:1] + 0.587 * rgb_state[:, 1:2] + 0.114 * rgb_state[:, 2:3]

                        # Fill all 4 channels with the same grayscale image
                        for i in range(4):
                            processed_state[:, i] = gray_state.squeeze(1)

                        state_tensor = processed_state / 255.0
                    else:
                        # Create a blank state as fallback
                        state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)
                except:
                    # Create a blank state as fallback
                    state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)

            # Move to device
            state_tensor = state_tensor.to(self.device)

            # Get Q values from the network (in eval mode, uses mean weights)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)

                # Print Q-value information occasionally
                if DEBUG and self.action_count % 100 == 0:
                    q_numpy = q_values.cpu().numpy()[0]
                    max_q_idx = np.argmax(q_numpy)

                    model_type = "Noisy Network" if self.use_noisy_net else "Standard"
                    print(f"Using {model_type} model. Max Q-value: {q_numpy[max_q_idx]:.4f}")

                    # Get action description
                    try:
                        action_desc = '+'.join(COMPLEX_MOVEMENT[max_q_idx])
                        print(f"Selected action: {max_q_idx} ({action_desc})")
                    except:
                        print(f"Selected action: {max_q_idx}")

            return q_values.argmax().item()
        finally:
            # Restore network mode
            if was_training:
                self.q_net.train()

    def act(self, observation):
        """
        Select an action based on the current observation.
        This is the required interface method that uses get_action internally.

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action
        """
        # Increment action counter
        self.action_count += 1

        # Use the get_action method to select an action
        # For evaluation, we set use_epsilon=False to ensure deterministic behavior
        return self.get_action(observation, use_epsilon=False)
