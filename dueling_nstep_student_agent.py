import gym
import torch
import numpy as np
import os
from dueling_qnet import DuelingMarioQNet

# Action mapping for Super Mario Bros
# 'A' button represents jump
COMPLEX_MOVEMENT = [
    ['NOOP'],       # 0
    ['right'],      # 1
    ['right', 'A'], # 2 - right + jump
    ['right', 'B'], # 3
    ['right', 'A', 'B'], # 4 - right + jump + run
    ['A'],          # 5 - jump
    ['left'],       # 6
    ['left', 'A'],  # 7 - left + jump
    ['left', 'B'],  # 8
    ['left', 'A', 'B'], # 9 - left + jump + run
    ['down'],       # 10
    ['up'],         # 11
]

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that uses trained Dueling DQN model with N-step bootstrapping to select actions."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = "cpu"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device {self.device}')

        # Define state shape for Mario environment (4 stacked frames of 84x84 grayscale images)
        self.state_shape = (4, 84, 84, 1)

        # Initialize Dueling Q-network with Noisy Networks
        # We'll try to detect if the model was trained with Noisy Networks
        # by checking if the model path contains "noisy"
        self.use_noisy_net = False

        # Initialize Dueling Q-network
        self.q_net = DuelingMarioQNet(self.state_shape, self.action_space.n, use_noisy_net=self.use_noisy_net).to(self.device)

        # Try to load the latest model weights
        self.model_loaded = False
        self.model_path = self._find_latest_model()
        if self.model_path:
            try:
                # Check if the model path contains "noisy" to determine if we should use Noisy Networks
                if "noisy" in self.model_path.lower():
                    print("Detected Noisy Network model. Reinitializing network with Noisy layers...")
                    self.use_noisy_net = True
                    # Reinitialize the network with Noisy layers
                    self.q_net = DuelingMarioQNet(self.state_shape, self.action_space.n, use_noisy_net=True).to(self.device)

                # Load the model weights
                state_dict = torch.load(self.model_path, map_location=self.device)

                # Load the model with strict=False to allow for architecture differences
                self.q_net.load_state_dict(state_dict, strict=False)
                self.model_loaded = True
                print(f"Loaded model weights from {self.model_path}")
                print(f"Using Noisy Networks: {self.use_noisy_net}")

                # If using Noisy Networks, make sure to reset noise before evaluation
                if self.use_noisy_net:
                    self.q_net.reset_noise()
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No model weights found. Using random actions.")

        # Counter for tracking actions
        self.action_count = 0

    def _find_latest_model(self):
        """Find the latest model weights file in the checkpoints directory."""
        print("Searching for model weights...")

        # Check for the main model file first (this is the latest weights saved during training)
        # First check for noisy network model
        if os.path.exists("mario_dueling_nstep_noisy_qnet.pth"):
            print("Found main noisy model file: mario_dueling_nstep_noisy_qnet.pth")
            return "mario_dueling_nstep_noisy_qnet.pth"

        # Then check for regular model
        if os.path.exists("mario_dueling_nstep_qnet.pth"):
            print("Found main model file: mario_dueling_nstep_qnet.pth")
            return "mario_dueling_nstep_qnet.pth"

        # Check for noisy checkpoint files first
        if os.path.exists("noisy_checkpoints"):
            print("Checking noisy_checkpoints directory...")
            # Find all Q-network weight files
            qnet_files = [f for f in os.listdir("noisy_checkpoints") if f.startswith("mario_dueling_nstep_qnet_ep") and f.endswith(".pth")]

            # Also check for full checkpoint files which contain the state dict
            checkpoint_files = [f for f in os.listdir("noisy_checkpoints") if f.startswith("mario_dueling_nstep_ep") and f.endswith(".pth")]

            if checkpoint_files and not qnet_files:
                print("Found full checkpoint files but no direct qnet files. Will try to extract weights from checkpoints.")
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
                    full_path = os.path.join("noisy_checkpoints", latest_file)
                    print(f"Found latest checkpoint: {full_path}")

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

            # Continue with direct qnet files if available
            if qnet_files:
                # Extract episode numbers and find the latest one
                latest_file = None
                latest_episode = -1

                for file in qnet_files:
                    try:
                        # Extract episode number from filename (format: mario_dueling_nstep_qnet_ep{episode}.pth)
                        episode = int(file.split("_ep")[1].split(".")[0])
                        if episode > latest_episode:
                            latest_episode = episode
                            latest_file = file
                    except:
                        continue

                if latest_file:
                    return os.path.join("noisy_checkpoints", latest_file)

        # Check for regular n-step checkpoint files
        if os.path.exists("dueling_nstep_checkpoints"):
            print("Checking dueling_nstep_checkpoints directory...")
            # Find all Q-network weight files
            qnet_files = [f for f in os.listdir("dueling_nstep_checkpoints") if f.startswith("mario_dueling_nstep_qnet_ep") and f.endswith(".pth")]

            # Also check for full checkpoint files which contain the state dict
            checkpoint_files = [f for f in os.listdir("dueling_nstep_checkpoints") if f.startswith("mario_dueling_nstep_ep") and f.endswith(".pth")]

            if checkpoint_files and not qnet_files:
                print("Found full checkpoint files but no direct qnet files. Will try to extract weights from checkpoints.")
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
                    print(f"Found latest checkpoint: {full_path}")

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

            # Continue with direct qnet files if available

            if qnet_files:
                # Extract episode numbers and find the latest one
                latest_file = None
                latest_episode = -1

                for file in qnet_files:
                    try:
                        # Extract episode number from filename (format: mario_dueling_nstep_qnet_ep{episode}.pth)
                        episode = int(file.split("_ep")[1].split(".")[0])
                        if episode > latest_episode:
                            latest_episode = episode
                            latest_file = file
                    except:
                        continue

                if latest_file:
                    return os.path.join("dueling_nstep_checkpoints", latest_file)

        # If no dueling n-step model found, try to load dueling model
        if os.path.exists("mario_dueling_ddqn_per_qnet.pth"):
            print("No dueling n-step model found. Trying to load dueling model.")
            return "mario_dueling_ddqn_per_qnet.pth"

        # Check dueling checkpoints
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
                    print("Loading dueling model.")
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
        # Print action count every 100 actions
        self.action_count += 1
        if self.action_count % 100 == 0:
            print(f"Action count: {self.action_count}")

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

            # For the evaluation environment, we need to handle a specific format
            # The error shows input shape [1, 240, 256, 3] which is [batch, height, width, channels]

            # First, check if this is the format from the evaluation environment
            if len(state.shape) == 4 and state.shape[1] == 240 and state.shape[2] == 256 and state.shape[3] == 3:
                # This is the format from eval.py
                # We need to create a tensor with exactly the same shape that was used during training
                # The model expects input of shape [batch_size, 4, 84, 84]

                # Create a new tensor with the correct shape
                processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                # Extract features from the original state to fill the new tensor
                # First, permute to [1, 3, 240, 256] to get channels first
                rgb_state = state.permute(0, 3, 1, 2)

                # Resize to [1, 3, 84, 84]
                rgb_state = torch.nn.functional.interpolate(rgb_state, size=(84, 84), mode='bilinear', align_corners=False)

                # Convert RGB to grayscale: 0.299 * R + 0.587 * G + 0.114 * B
                gray_state = 0.299 * rgb_state[:, 0:1] + 0.587 * rgb_state[:, 1:2] + 0.114 * rgb_state[:, 2:3]

                # Fill all 4 channels with the same grayscale image
                processed_state[:, 0] = gray_state.squeeze(1)
                processed_state[:, 1] = gray_state.squeeze(1)
                processed_state[:, 2] = gray_state.squeeze(1)
                processed_state[:, 3] = gray_state.squeeze(1)

                # Replace the original state with the processed one
                state = processed_state

            # Handle our training environment format
            elif len(state.shape) == 4 and state.shape[0] == 4:  # (4, 84, 84, 1) - stacked frames
                # Create a new tensor with the correct shape
                processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                # Remove channel dimension and add batch dimension
                state_no_channel = state.squeeze(-1)  # Shape becomes (4, 84, 84)

                # Copy each frame to the processed state
                for i in range(4):
                    processed_state[0, i] = state_no_channel[i]

                # Replace the original state with the processed one
                state = processed_state

            # Handle other potential formats
            elif len(state.shape) == 3 and state.shape[-1] == 3:  # (height, width, 3) - RGB image
                # Create a new tensor with the correct shape
                processed_state = torch.zeros(1, 4, 84, 84, device=self.device)

                # Convert to grayscale
                rgb_state = state.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

                # Resize if needed
                if rgb_state.shape[2] != 84 or rgb_state.shape[3] != 84:
                    rgb_state = torch.nn.functional.interpolate(rgb_state, size=(84, 84), mode='bilinear', align_corners=False)

                # Convert to grayscale
                gray_state = 0.299 * rgb_state[:, 0:1] + 0.587 * rgb_state[:, 1:2] + 0.114 * rgb_state[:, 2:3]  # [1, 1, 84, 84]

                # Fill all 4 channels with the same grayscale image
                processed_state[:, 0] = gray_state.squeeze(1)
                processed_state[:, 1] = gray_state.squeeze(1)
                processed_state[:, 2] = gray_state.squeeze(1)
                processed_state[:, 3] = gray_state.squeeze(1)

                # Replace the original state with the processed one
                state = processed_state

            # If already in correct format [1, 4, 84, 84], no need to reshape
            elif len(state.shape) == 4 and state.shape[0] == 1 and state.shape[1] == 4 and state.shape[2] == 84 and state.shape[3] == 84:
                # Already in the correct format
                pass

            # Normalize (if not already done)
            # This normalization is done here in the agent, not in the network
            if state.max() > 1.0:
                state = state / 255.0

            # Get Q-values and select best action
            self.q_net.eval()

            # If using Noisy Networks, reset noise before evaluation
            if self.use_noisy_net:
                self.q_net.reset_noise()

            q_values = self.q_net(state)
            action = q_values.argmax().item()

            # Print information about the state and action
            if self.action_count % 20 == 0:  # Print every 20 actions to avoid too much output
                max_q = q_values.max().item()
                # Get the action description from COMPLEX_MOVEMENT
                action_desc = '+'.join(COMPLEX_MOVEMENT[action])
                print(f"Action: {action} ({action_desc}), Q-value: {max_q:.4f}")

            return action
