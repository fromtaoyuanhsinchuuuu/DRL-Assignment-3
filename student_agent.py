import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from dueling_qnet import DuelingMarioQNet
import config
from frame_processor import create_frame_processor, reset_frame_processor

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
        self.state_shape = (4, 84, 84, 1) # This is the shape *after* preprocessing

        # Flag to track if we're using noisy networks
        self.use_noisy_net = False

        # --- 使用 frame_processor 模組初始化幀處理器 ---
        self.skip_frames = 4 # 與 MaxAndSkipEnv 中的 skip 值一致
        self.process_frame, self.obs_buffer, self.frame_stack = create_frame_processor(
            device=self.device,
            skip_frames=self.skip_frames
        )
        self._current_action = 0 # 存儲在幀跳過期間重複的動作
        self._last_processed_state = None # 存儲上次生成的堆疊狀態

        # Find the model path first to determine if it's a noisy network model
        self.model_path = self._find_latest_model()

        # Check if the model path contains "noisy" to determine if we should use Noisy Networks
        # if self.model_path and "noisy" in self.model_path.lower():
        #     print("Detected Noisy Network model. Initializing network with Noisy layers...")
        #     self.use_noisy_net = True

        # Initialize Dueling Q-network with appropriate noisy network setting
        # Note: The input shape to the network is (4, 84, 84) after preprocessing and stacking
        self.q_net = DuelingMarioQNet((4, 84, 84), self.action_space.n, use_noisy_net=self.use_noisy_net).to(self.device)

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
        # If no model is loaded, use random actions
        if not self.model_loaded:
            return random.randrange(self.action_space.n)

        # If NOT using Noisy Networks, use epsilon-greedy with 1% exploration rate
        if not self.use_noisy_net:
            if use_epsilon and random.random() < 0.01:  # 1% chance of random action
                random_action = random.randrange(self.action_space.n)
                if DEBUG:
                    print(f"Using epsilon-greedy exploration: random action {random_action}")
                return random_action

        # Remember current network mode
        was_training = self.q_net.training

        # Set network to evaluation mode for deterministic action selection
        self.q_net.eval()

        try:
            # --- 統一將輸入轉換為 PyTorch 張量並移動到設備 ---
            if isinstance(state, np.ndarray):
                # Handle the standard training environment format (4, 84, 84, 1)
                if len(state.shape) == 4 and state.shape[0] == 4 and state.shape[3] == 1:
                    # Convert to tensor
                    state_tensor = torch.from_numpy(state).float()
                    # Remove the channel dimension (which is 1)
                    state_tensor = state_tensor.squeeze(-1)  # Shape: (4, 84, 84)
                    # Add batch dimension
                    state_tensor = state_tensor.unsqueeze(0)  # Shape: (1, 4, 84, 84)
                    # Normalize for inference
                    # This ensures the state is in the same range [0, 1] as during training
                    state_tensor = state_tensor / 255.0
                else:
                    # For other formats, use the last processed state if available
                    if self._last_processed_state is not None:
                        state_tensor = self._last_processed_state
                    else:
                        # Create a blank state as fallback
                        state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)
            elif isinstance(state, torch.Tensor):
                # If it's already a tensor, ensure it has the right shape
                if state.dim() == 3 and state.shape[0] == 4 and state.shape[1] == 84 and state.shape[2] == 84:
                    # Add batch dimension if needed
                    state_tensor = state.unsqueeze(0)
                elif state.dim() == 4 and state.shape[0] == 1 and state.shape[1] == 4:
                    # Already in the right format
                    state_tensor = state
                else:
                    # For other formats, use the last processed state if available
                    if self._last_processed_state is not None:
                        state_tensor = self._last_processed_state
                    else:
                        # Create a blank state as fallback
                        state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)
            elif hasattr(state, '__array__'):
                # Check if the state is a PyTorch tensor and on CUDA
                if isinstance(state, torch.Tensor) and state.device.type == 'cuda':
                    # Already a CUDA tensor, just ensure it's in the right format
                    if state.dim() == 3 and state.shape[0] == 4:
                        state_tensor = state.unsqueeze(0)  # Add batch dimension
                    elif state.dim() == 4 and state.shape[0] == 1 and state.shape[1] == 4:
                        state_tensor = state  # Already in the right format
                    else:
                        # Use last processed state as fallback
                        if self._last_processed_state is not None:
                            state_tensor = self._last_processed_state
                        else:
                            state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)
                else:
                    # Convert to NumPy and then to tensor
                    try:
                        state_np = np.array(state)
                        if len(state_np.shape) == 4 and state_np.shape[0] == 4 and state_np.shape[3] == 1:
                            # Standard format (4, 84, 84, 1)
                            state_tensor = torch.from_numpy(state_np).float()
                            state_tensor = state_tensor.squeeze(-1)  # Remove channel dimension
                            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                            # Normalize for inference
                            # This ensures the state is in the same range [0, 1] as during training
                            state_tensor = state_tensor / 255.0
                        else:
                            # Use last processed state as fallback
                            if self._last_processed_state is not None:
                                state_tensor = self._last_processed_state
                            else:
                                state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)
                    except Exception as e:
                        if DEBUG:
                            print(f"Error converting state to NumPy: {e}")
                        # Use last processed state as fallback
                        if self._last_processed_state is not None:
                            state_tensor = self._last_processed_state
                        else:
                            state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)
            else:
                # For other types, use the last processed state if available
                if self._last_processed_state is not None:
                    state_tensor = self._last_processed_state
                else:
                    # Create a blank state as fallback
                    state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)

            # Ensure the tensor is on the correct device
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
        except Exception as e:
            if DEBUG:
                print(f"Error in get_action: {e}")
            # Return a random action as fallback
            return random.randrange(self.action_space.n)
        finally:
            # Restore network mode
            if was_training:
                self.q_net.train()

    def act(self, observation):
        """
        Select an action based on the current observation, simulating environment preprocessing.
        Args:
            observation: Current raw observation from the environment (e.g., RGB 240x256x3).
        Returns:
            Selected action.
        """
        # If no model is loaded, use random actions
        if not self.model_loaded:
            return random.randrange(self.action_space.n)

        # Increment action counter (optional, for debugging)
        self.action_count += 1

        # Detect episode reset by checking if observation is a new episode start
        # This is a heuristic - we assume a new episode starts when:
        # 1. The observation is a full RGB frame (240, 256, 3)
        # 2. The action count is 1 or the observation is significantly different from previous frames
        if hasattr(observation, '__array__'):
            # Handle CUDA tensors properly
            if isinstance(observation, torch.Tensor) and observation.device.type == 'cuda':
                # Move to CPU before converting to NumPy
                obs_array = observation.cpu().numpy()
            else:
                # Otherwise, convert to NumPy directly
                obs_array = np.array(observation)

            if len(obs_array.shape) == 3 and obs_array.shape[0] == 240 and obs_array.shape[1] == 256 and obs_array.shape[2] == 3:
                # Check if this is the first action or if the observation is very different
                if self.action_count == 1 or (len(self.obs_buffer) > 0 and
                   np.mean(np.abs(obs_array - np.array(self.obs_buffer[-1]))) > 50):  # Threshold for difference
                    # Reset buffers for new episode
                    if DEBUG:
                        print("Detected new episode, resetting buffers")
                    # Use the reset_frame_processor function to reset buffers
                    reset_frame_processor(self.obs_buffer, self.frame_stack)

        # Process the current raw observation frame using our frame processor
        # This function will add to obs_buffer and frame_stack internally
        # It returns a new stacked state only when 'skip' frames have been processed
        new_stacked_state = self.process_frame(observation)

        # If a new stacked state was generated (i.e., after 'skip' raw frames)
        if new_stacked_state is not None:
            self._last_processed_state = new_stacked_state # Store the new state

            try:
                # Use get_action method for consistency
                # This will handle epsilon-greedy exploration and network inference
                self._current_action = self.get_action(self._last_processed_state, use_epsilon=True)

                # Print action information occasionally
                if DEBUG and self.action_count % (self.skip_frames * 10) == 0: # Print less often
                    try:
                        action_desc = '+'.join(COMPLEX_MOVEMENT[self._current_action])
                        print(f"Selected action: {self._current_action} ({action_desc})")
                    except:
                        print(f"Selected action: {self._current_action}")
            except Exception as e:
                if DEBUG:
                    print(f"Error in act() when calling get_action(): {e}")
                # Fallback to random action if get_action fails
                self._current_action = random.randrange(self.action_space.n)

        # Return the current action. This action will be repeated for 'skip' raw frames
        # until a new stacked state is generated and a new action is selected.
        return self._current_action
