import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from dueling_qnet import DuelingMarioQNet
import config
from collections import deque

# Debug flag - set to True to enable debug messages, False to disable
DEBUG = False

# --- 新增的預處理函數，包含幀跳過和取最大值模擬 ---
def preprocess_frame(raw_frame, obs_buffer, frame_stack, device, skip=4):
    """
    Preprocess a single raw observation frame, simulating frame skipping and max pooling.
    This function is called for each raw frame received from the environment.
    Args:
        raw_frame: The raw observation frame from the environment (e.g., RGB 240x256x3).
        obs_buffer: Deque to store the last 2 raw frames for max pooling.
        frame_stack: Deque to store the last 4 processed (84x84 grayscale) frames.
        device: Device to move tensors to.
        skip: The frame skipping value (default: 4).
    Returns:
        torch.Tensor or None: Processed and stacked observation tensor (1, 4, 84, 84)
                               if a new stacked frame is generated, otherwise None.
    """
    # Convert raw_frame to numpy array if needed
    if hasattr(raw_frame, '__array__'):
        raw_frame = np.array(raw_frame)

    # Add the current raw frame to the observation buffer
    obs_buffer.append(raw_frame)

    # Only process and stack frames every 'skip' steps
    if len(obs_buffer) == skip:
        # Take the maximum over the last 2 frames in the buffer
        # Even though we're storing up to 'skip' frames, we only take max over the last 2
        # This matches the behavior of MaxAndSkipEnv in the training environment
        if len(obs_buffer) >= 2:
            max_frame = np.maximum(obs_buffer[-2], obs_buffer[-1])
        else:
            # Should not happen if skip >= 2, but handle defensively
            max_frame = obs_buffer[-1]

        # Convert the max_frame (RGB) to tensor
        rgb_tensor = torch.from_numpy(max_frame).float()

        # Convert RGB to PyTorch format (channels first) and add batch dimension
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 240, 256]

        # Resize to [1, 3, 84, 84]
        rgb_tensor = torch.nn.functional.interpolate(rgb_tensor, size=(84, 84), mode='bilinear', align_corners=False)

        # Convert RGB to grayscale: 0.299 * R + 0.587 * G + 0.114 * B
        gray_tensor = 0.299 * rgb_tensor[:, 0:1] + 0.587 * rgb_tensor[:, 1:2] + 0.114 * rgb_tensor[:, 2:3]

        # Convert to numpy array and remove batch and channel dimensions
        current_processed_frame = gray_tensor.squeeze().cpu().numpy()  # Shape: (84, 84)

        # Add the current processed frame to the frame stack
        frame_stack.append(current_processed_frame)

        # Clear the observation buffer after processing
        obs_buffer.clear()

        # Create a tensor with the correct shape for PyTorch (batch, channels, height, width)
        stacked_state = torch.zeros(1, 4, 84, 84, device=device)

        # Fill the tensor with the frames from the frame stack
        for i, frame in enumerate(frame_stack):
            stacked_state[0, i] = torch.from_numpy(frame).float()

        # Normalize
        stacked_state = stacked_state / 255.0

        return stacked_state.to(device)
    else:
        # If not enough frames for skipping, return None
        return None

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

        # --- 新增：用於模擬幀跳過和取最大值的緩衝區和計數器 ---
        self.skip_frames = 4 # 與 MaxAndSkipEnv 中的 skip 值一致
        self.obs_buffer = deque(maxlen=self.skip_frames) # 存儲最近 skip_frames 幀原始觀察用於取最大值
        self.frame_stack = deque(maxlen=4) # 存儲最近 4 幀處理後的灰度圖 (84x84)
        self._current_action = 0 # 存儲在幀跳過期間重複的動作
        self._last_processed_state = None # 存儲上次生成的堆疊狀態

        # Initialize frame stack with blank frames
        blank_frame_84x84 = np.zeros((84, 84), dtype=np.float32)
        for _ in range(4):
            self.frame_stack.append(blank_frame_84x84)

        # Find the model path first to determine if it's a noisy network model
        self.model_path = self._find_latest_model()

        # Check if the model path contains "noisy" to determine if we should use Noisy Networks
        if self.model_path and "noisy" in self.model_path.lower():
            print("Detected Noisy Network model. Initializing network with Noisy layers...")
            self.use_noisy_net = True

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
            # For get_action, we assume the state is already preprocessed
            # Convert to tensor if it's a numpy array
            if isinstance(state, np.ndarray):
                # Handle the standard training environment format (4, 84, 84, 1)
                if len(state.shape) == 4 and state.shape[0] == 4 and state.shape[3] == 1:
                    # Convert to tensor
                    state_tensor = torch.from_numpy(state).float()
                    # Remove the channel dimension (which is 1)
                    state_tensor = state_tensor.squeeze(-1)  # Shape: (4, 84, 84)
                    # Add batch dimension
                    state_tensor = state_tensor.unsqueeze(0)  # Shape: (1, 4, 84, 84)
                    # Normalize
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
            else:
                # For other types, use the last processed state if available
                if self._last_processed_state is not None:
                    state_tensor = self._last_processed_state
                else:
                    # Create a blank state as fallback
                    state_tensor = torch.zeros(1, 4, 84, 84, device=self.device)

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
            obs_array = np.array(observation)
            if len(obs_array.shape) == 3 and obs_array.shape[0] == 240 and obs_array.shape[1] == 256 and obs_array.shape[2] == 3:
                # Check if this is the first action or if the observation is very different
                if self.action_count == 1 or (len(self.obs_buffer) > 0 and
                   np.mean(np.abs(obs_array - self.obs_buffer[-1])) > 50):  # Threshold for difference
                    # Reset buffers for new episode
                    if DEBUG:
                        print("Detected new episode, resetting buffers")
                    self.obs_buffer.clear()
                    # Reset frame stack with blank frames
                    self.frame_stack.clear()
                    blank_frame = np.zeros((84, 84), dtype=np.float32)
                    for _ in range(4):
                        self.frame_stack.append(blank_frame)

        # Process the current raw observation frame
        # This function will add to obs_buffer and frame_stack internally
        # It returns a new stacked state only when 'skip' frames have been processed
        new_stacked_state = preprocess_frame(
            observation,
            self.obs_buffer,
            self.frame_stack,
            self.device,
            skip=self.skip_frames
        )

        # If a new stacked state was generated (i.e., after 'skip' raw frames)
        if new_stacked_state is not None:
            self._last_processed_state = new_stacked_state # Store the new state
            # Set network to evaluation mode for deterministic action selection
            was_training = self.q_net.training
            self.q_net.eval()

            try:
                # Get Q values from the network
                with torch.no_grad():
                    q_values = self.q_net(self._last_processed_state)
                    # Select the action with the highest Q value
                    self._current_action = q_values.argmax().item()

                    # Print Q-value information occasionally
                    if DEBUG and self.action_count % (self.skip_frames * 10) == 0: # Print less often
                        q_numpy = q_values.cpu().numpy()[0]
                        max_q_idx = np.argmax(q_numpy)
                        model_type = "Noisy Network" if self.use_noisy_net else "Standard"
                        print(f"Using {model_type} model. Max Q-value: {q_numpy[max_q_idx]:.4f}")
                        try:
                            action_desc = '+'.join(COMPLEX_MOVEMENT[max_q_idx])
                            print(f"Selected action: {max_q_idx} ({action_desc})")
                        except:
                            print(f"Selected action: {max_q_idx}")
            except Exception as e:
                if DEBUG:
                    print(f"Error getting action from model: {e}")
                # Fallback to random action if model inference fails
                self._current_action = random.randrange(self.action_space.n)
            finally:
                # Restore network mode
                if was_training:
                    self.q_net.train()

        # Return the current action. This action will be repeated for 'skip' raw frames
        # until a new stacked state is generated and a new action is selected.
        return self._current_action
