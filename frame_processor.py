# frame_processor.py
"""
This module provides frame processing functions for the Super Mario Bros environment.
It includes functions for preprocessing raw frames, simulating frame skipping,
and stacking frames for use with deep reinforcement learning agents.
"""

import numpy as np
import torch
from collections import deque

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
        # Handle CUDA tensors properly
        if isinstance(raw_frame, torch.Tensor) and raw_frame.device.type == 'cuda':
            # Move to CPU before converting to NumPy
            raw_frame = raw_frame.cpu().numpy()
        else:
            # Otherwise, convert to NumPy directly
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

        # No normalization here
        # Normalization will be done in PER's sample method for training
        # and in Agent's get_action method for inference

        return stacked_state.to(device)
    else:
        # If not enough frames for skipping, return None
        return None

def create_frame_processor(device="cpu", skip_frames=4):
    """
    Create and return a frame processor with initialized buffers.

    Args:
        device: The device to use for tensor operations (default: "cpu")
        skip_frames: Number of frames to skip (default: 4)

    Returns:
        tuple: (process_frame_func, obs_buffer, frame_stack)
            - process_frame_func: A function that takes a raw frame and returns a processed state
            - obs_buffer: The observation buffer for raw frames
            - frame_stack: The frame stack for processed frames
    """
    # Initialize buffers
    obs_buffer = deque(maxlen=skip_frames)
    frame_stack = deque(maxlen=4)

    # Initialize frame stack with blank frames
    blank_frame = np.zeros((84, 84), dtype=np.float32)
    for _ in range(4):
        frame_stack.append(blank_frame)

    # Create a closure that captures the buffers
    def process_frame(raw_frame):
        return preprocess_frame(raw_frame, obs_buffer, frame_stack, device, skip=skip_frames)

    return process_frame, obs_buffer, frame_stack

def reset_frame_processor(obs_buffer, frame_stack):
    """
    Reset the frame processor buffers.

    Args:
        obs_buffer: The observation buffer to reset
        frame_stack: The frame stack to reset
    """
    # Clear buffers
    obs_buffer.clear()
    frame_stack.clear()

    # Initialize frame stack with blank frames
    blank_frame = np.zeros((84, 84), dtype=np.float32)
    for _ in range(4):
        frame_stack.append(blank_frame)
