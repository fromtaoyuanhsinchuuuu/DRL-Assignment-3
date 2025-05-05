import gym
import torch
import numpy as np
import os
from dueling_qnet import DuelingMarioQNet



# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that uses trained Dueling DQN model to select actions."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = "cpu"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            print("Found noisy network model file: mario_dueling_nstep_noisy_qnet.pth")
            return "mario_dueling_nstep_noisy_qnet.pth"

        # Then, check for the regular n-step model file
        if os.path.exists("mario_dueling_nstep_qnet.pth"):
            print("Found n-step model file: mario_dueling_nstep_qnet.pth")
            return "mario_dueling_nstep_qnet.pth"

        # Check for n-step checkpoint files
        if os.path.exists("dueling_nstep_checkpoints"):
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
                    print(f"Found latest n-step model: {latest_file}")
                    return os.path.join("dueling_nstep_checkpoints", latest_file)

            # Also check for full checkpoint files which contain the state dict
            checkpoint_files = [f for f in os.listdir("dueling_nstep_checkpoints") if f.startswith("mario_dueling_nstep_ep") and f.endswith(".pth")]

            if checkpoint_files:
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
                    print(f"Found latest n-step checkpoint: {full_path}")

                    # Try to extract q_net_state_dict from the checkpoint
                    try:
                        checkpoint = torch.load(full_path, map_location=self.device)
                        if 'q_net_state_dict' in checkpoint:
                            print("Extracting q_net_state_dict from checkpoint...")
                            # Save the extracted state dict to a temporary file
                            temp_path = "temp_extracted_qnet.pth"
                            torch.save(checkpoint['q_net_state_dict'], temp_path)
                            return temp_path
                    except Exception as e:
                        print(f"Error extracting state dict from checkpoint: {e}")

        # Fall back to regular dueling model if n-step not found
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
                    print("Loading regular model into dueling architecture.")
                    return os.path.join("checkpoints", latest_file)

        return None

    def act(self, observation):
        # Increment action counter
        self.action_count += 1

        # If no model is loaded, use random actions
        if not self.model_loaded:
            return self.action_space.sample()

        # Process the observation (convert to tensor, reshape, normalize)
        with torch.no_grad():
            # Handle LazyFrames from FrameStack wrapper
            if hasattr(observation, '__array__'):
                observation = np.array(observation)

            # Convert to tensor
            state = torch.from_numpy(observation).float()

            # Handle the standard training environment format (4, 84, 84, 1)
            if len(state.shape) == 4 and state.shape[0] == 4 and state.shape[3] == 1:
                # Create a new tensor with the correct shape for PyTorch (batch, channels, height, width)
                processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                # Remove channel dimension and add batch dimension
                state_no_channel = state.squeeze(-1)  # Shape becomes (4, 84, 84)

                # Copy each frame to the processed state
                for i in range(4):
                    processed_state[0, i] = state_no_channel[i]

                # Replace the original state with the processed one
                state = processed_state

            # If already in correct format [1, 4, 84, 84], no need to reshape
            elif len(state.shape) == 4 and state.shape[0] == 1 and state.shape[1] == 4 and state.shape[2] == 84 and state.shape[3] == 84:
                # Already in the correct format
                pass
            else:
                # For any other format, print a warning and try to adapt
                print(f"Warning: Unexpected observation shape: {state.shape}. The evaluation environment should use the same wrappers as training.")

                # Try to handle common cases
                if len(state.shape) == 3 and state.shape[-1] == 3:  # Single RGB frame
                    print("Detected single RGB frame. Converting to 4-stacked grayscale frames.")
                    # Create tensor with correct shape
                    processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                    # Convert RGB to grayscale and resize
                    rgb_state = state.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                    if rgb_state.shape[2] != 84 or rgb_state.shape[3] != 84:
                        rgb_state = torch.nn.functional.interpolate(rgb_state, size=(84, 84), mode='bilinear', align_corners=False)

                    # Convert to grayscale
                    gray_state = 0.299 * rgb_state[:, 0:1] + 0.587 * rgb_state[:, 1:2] + 0.114 * rgb_state[:, 2:3]

                    # Fill all 4 channels with the same grayscale image
                    for i in range(4):
                        processed_state[:, i] = gray_state.squeeze(1)

                    state = processed_state
                elif len(state.shape) == 4 and state.shape[0] == 1 and state.shape[-1] == 3:  # Batch of RGB frames
                    print("Detected batch of RGB frames. Converting to 4-stacked grayscale frames.")
                    # Create tensor with correct shape
                    processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                    # Convert RGB to grayscale and resize
                    rgb_state = state.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    if rgb_state.shape[2] != 84 or rgb_state.shape[3] != 84:
                        rgb_state = torch.nn.functional.interpolate(rgb_state, size=(84, 84), mode='bilinear', align_corners=False)

                    # Convert to grayscale
                    gray_state = 0.299 * rgb_state[:, 0:1] + 0.587 * rgb_state[:, 1:2] + 0.114 * rgb_state[:, 2:3]

                    # Fill all 4 channels with the same grayscale image
                    for i in range(4):
                        processed_state[:, i] = gray_state.squeeze(1)

                    state = processed_state

            # Normalize (if not already done)
            if state.max() > 1.0:
                state = state / 255.0

            # Set network to evaluation mode
            self.q_net.eval()

            # Get Q-values and select best action
            q_values = self.q_net(state)
            action = q_values.argmax().item()

            # Print information about the state and action occasionally
            if self.action_count % 20 == 0:
                max_q = q_values.max().item()
                model_type = "Noisy Network" if self.use_noisy_net else "Standard"
                print(f"Using {model_type} model. Max Q-value: {max_q:.4f}, Selected action: {action}")

            return action
