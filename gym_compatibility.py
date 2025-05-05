"""
Compatibility layer for handling different Gym API versions.
This module provides wrappers to ensure compatibility between old and new Gym APIs.
"""
import gym
import numpy as np

# This class is no longer used as we're handling API differences directly in each wrapper
# Keeping the class definition as a placeholder to avoid breaking imports
class GymAPICompatibilityWrapper:
    """
    This class is deprecated. API differences are now handled directly in each wrapper.
    """
    pass

class CompatibleGrayScaleObservation(gym.ObservationWrapper):
    """
    Convert the image observation from RGB to grayscale.
    Compatible with both old and new Gym APIs.
    """
    def __init__(self, env, keep_dim=False):
        super(CompatibleGrayScaleObservation, self).__init__(env)
        self.keep_dim = keep_dim

        # Update the observation space
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] == 3:
            if self.keep_dim:
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
                )
            else:
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(obs_shape[0], obs_shape[1]), dtype=np.uint8
                )

    def observation(self, observation):
        # Handle case where observation might be a tuple (obs, info)
        if isinstance(observation, tuple):
            # 移除不必要的打印
            # Extract the actual observation from the tuple
            if len(observation) >= 1:
                observation = observation[0]
            else:
                raise ValueError("Empty tuple received as observation")

        try:
            # Convert RGB to grayscale
            grayscale = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

            if self.keep_dim:
                grayscale = grayscale[..., np.newaxis]

            return grayscale
        except Exception as e:
            # 移除不必要的打印
            raise

class CompatibleResizeObservation(gym.ObservationWrapper):
    """
    Resize the image observation to a given shape.
    Compatible with both old and new Gym APIs.
    """
    def __init__(self, env, shape):
        super(CompatibleResizeObservation, self).__init__(env)

        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # Update the observation space
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.shape[0], self.shape[1], obs_shape[2]), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.shape, dtype=np.uint8
            )

    def observation(self, observation):
        # Handle case where observation might be a tuple (obs, info)
        if isinstance(observation, tuple):
            # 移除不必要的打印
            # Extract the actual observation from the tuple
            if len(observation) >= 1:
                observation = observation[0]
            else:
                raise ValueError("Empty tuple received as observation")

        try:
            # Import here to avoid dependency issues
            try:
                import cv2

                def resize(image, output_shape, preserve_range=True, anti_aliasing=False):
                    """Simple resize function using OpenCV"""
                    # OpenCV expects (height, width) order
                    return cv2.resize(image, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                # 移除不必要的打印
                pass

                def resize(image, output_shape, preserve_range=True, anti_aliasing=False):
                    """Simple resize function using numpy"""
                    # Very basic nearest-neighbor resizing
                    h_in, w_in = image.shape[:2]
                    h_out, w_out = output_shape

                    h_indices = np.floor(np.arange(h_out) * h_in / h_out).astype(int)
                    w_indices = np.floor(np.arange(w_out) * w_in / w_out).astype(int)

                    if len(image.shape) == 3:
                        return image[h_indices[:, np.newaxis], w_indices, :]
                    else:
                        return image[h_indices[:, np.newaxis], w_indices]

            # Resize the observation
            if len(observation.shape) == 3 and observation.shape[2] == 1:
                # For grayscale images with channel dimension
                resized = resize(observation[:, :, 0], self.shape, preserve_range=True, anti_aliasing=False)
                return resized[..., np.newaxis].astype(np.uint8)
            elif len(observation.shape) == 3:
                # For RGB images
                resized = resize(observation, (*self.shape, observation.shape[2]), preserve_range=True, anti_aliasing=False)
                return resized.astype(np.uint8)
            else:
                # For grayscale images without channel dimension
                resized = resize(observation, self.shape, preserve_range=True, anti_aliasing=False)
                return resized.astype(np.uint8)
        except Exception as e:
            # 移除不必要的打印
            raise

class CompatibleFrameStack(gym.Wrapper):
    """
    Stack k last frames.
    Compatible with both old and new Gym APIs.
    """
    def __init__(self, env, num_stack):
        super(CompatibleFrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.frames = []

        # Update the observation space
        obs_shape = env.observation_space.shape

        # Handle different observation shapes
        if len(obs_shape) == 3:  # (H, W, C)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(num_stack, obs_shape[0], obs_shape[1], obs_shape[2]),
                dtype=env.observation_space.dtype
            )
            self.stack_axis = 0
        elif len(obs_shape) == 2:  # (H, W)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(num_stack, obs_shape[0], obs_shape[1]),
                dtype=env.observation_space.dtype
            )
            self.stack_axis = 0

    def reset(self, **kwargs):
        """
        Reset the environment and stack initial frames.
        Always returns just the observation (old Gym API format).
        """
        reset_result = self.env.reset(**kwargs)

        # Handle different return formats
        if isinstance(reset_result, tuple) and len(reset_result) >= 1:
            # New Gym API format (obs, info)
            obs = reset_result[0]
        else:
            # Old Gym API format (just obs)
            obs = reset_result

        # Clear the frame buffer
        self.frames = []

        # Stack the initial observation num_stack times
        for _ in range(self.num_stack):
            self.frames.append(obs)

        stacked_frames = np.stack(self.frames, axis=self.stack_axis)
        return stacked_frames

    def step(self, action):
        """Step the environment and update the frame stack."""
        step_result = self.env.step(action)

        # Handle different return formats
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            print(f"Warning: Unexpected step result length: {len(step_result)}")
            # Try to extract what we can
            obs = step_result[0] if len(step_result) >= 1 else None
            reward = step_result[1] if len(step_result) >= 2 else 0.0
            done = step_result[2] if len(step_result) >= 3 else False
            info = step_result[3] if len(step_result) >= 4 else {}

        # Update the frame stack
        self.frames.pop(0)
        self.frames.append(obs)

        stacked_frames = np.stack(self.frames, axis=self.stack_axis)

        # Always return 4 values (old Gym API format)
        return stacked_frames, reward, done, info
