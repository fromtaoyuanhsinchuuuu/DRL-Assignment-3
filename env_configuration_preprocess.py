# env_configuration_preprocess.py
import config
import numpy as np
from gym_compatibility import (
    CompatibleGrayScaleObservation,
    CompatibleResizeObservation,
    CompatibleFrameStack
)

# 導入 gym
import gym
# 移除不必要的打印
# print("Using custom compatible wrappers for gym")

# 定義 Frame Skipping 包裝器
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max over the last 2 frames.
    Useful for games like Atari and Mario where there's flickering.
    """
    def __init__(self, env, skip=4):
        """
        Initialize MaxAndSkipEnv wrapper.

        Args:
            env: The environment to wrap
            skip: Number of frames to skip (default: 4)
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)

    def step(self, action):
        """
        Step the environment with the given action.
        Repeat action, sum reward, and max over last observations.
        Always returns 4 values (old Gym API format).
        """
        total_reward = 0.0
        done = False
        info = {}

        # Repeat the action `skip` times
        for i in range(self._skip):
            try:
                # Call the environment's step method
                step_result = self.env.step(action)

                # Handle different return formats
                if len(step_result) == 5:
                    # New Gym API format
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # Old Gym API format
                    obs, reward, done, info = step_result
                else:
                    print(f"Warning: Unexpected step result length: {len(step_result)}")
                    # Try to extract what we can
                    obs = step_result[0] if len(step_result) >= 1 else None
                    reward = step_result[1] if len(step_result) >= 2 else 0.0
                    done = step_result[2] if len(step_result) >= 3 else False
                    info = step_result[3] if len(step_result) >= 4 else {}

                if i == self._skip - 2:
                    self._obs_buffer[0] = obs
                if i == self._skip - 1:
                    self._obs_buffer[1] = obs
                total_reward += reward
                if done:
                    break
            except Exception as e:
                # Log the error for debugging
                print(f"Error in MaxAndSkipEnv.step: {e}")
                # Return safe defaults
                if hasattr(self, '_obs_buffer') and self._obs_buffer.shape[0] > 0:
                    max_frame = self._obs_buffer.max(axis=0)
                    return max_frame, 0.0, True, {}
                else:
                    return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), 0.0, True, {}

        # Max over the last 2 frames to deal with flickering
        max_frame = self._obs_buffer.max(axis=0)

        # Always return 4 values (old Gym API format)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        Always returns just the observation (old Gym API format).
        """
        try:
            # Call the environment's reset method
            reset_result = self.env.reset(**kwargs)

            # Handle different return formats
            if isinstance(reset_result, tuple) and len(reset_result) >= 1:
                # New Gym API format (obs, info)
                obs = reset_result[0]
            else:
                # Old Gym API format (just obs)
                obs = reset_result

            return obs
        except Exception as e:
            print(f"Error in MaxAndSkipEnv.reset: {e}")
            # Return a blank observation
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

# 導入 Super Mario Bros 環境
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# Add a final wrapper to ensure consistent API with old Gym format
class OldAPIWrapper(gym.Wrapper):
    """
    A final wrapper to ensure the environment always returns 4 values from step().
    This helps maintain consistency with the old Gym API format.
    """
    def __init__(self, env):
        super(OldAPIWrapper, self).__init__(env)
        self.done = False

    def step(self, action):
        """Ensure step always returns 4 values (old Gym API format)."""
        # Check if the environment is already done
        if self.done:
            # Return the last observation with a terminal state
            if hasattr(self, 'last_obs'):
                return self.last_obs, 0.0, True, {}
            else:
                # Create a blank observation
                blank_shape = self.observation_space.shape
                blank_obs = np.zeros(blank_shape, dtype=np.uint8)
                return blank_obs, 0.0, True, {}

        try:
            result = self.env.step(action)

            # Convert to old Gym API format (4 values)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                obs, reward, done, info = result
            else:
                print(f"Warning: Unexpected step result length: {len(result)}")
                # Try to extract what we can and fill in defaults
                obs = result[0] if len(result) >= 1 else None
                reward = result[1] if len(result) >= 2 else 0.0
                done = result[2] if len(result) >= 3 else False
                info = result[3] if len(result) >= 4 else {}

            # Store the done state and last observation
            self.done = done
            self.last_obs = obs

            # Return 4 values (old Gym API format)
            return obs, reward, done, info

        except Exception as e:
            print(f"Error in step: {e}")
            # Return safe defaults
            if hasattr(self, 'last_obs'):
                return self.last_obs, -100.0, True, {}
            else:
                blank_shape = self.observation_space.shape
                blank_obs = np.zeros(blank_shape, dtype=np.uint8)
                return blank_obs, -100.0, True, {}

    def reset(self, **kwargs):
        """Reset the environment and return just the observation (old Gym API format)."""
        self.done = False

        try:
            result = self.env.reset(**kwargs)

            # Convert to old Gym API format
            if isinstance(result, tuple) and len(result) >= 1:
                obs = result[0]
            else:
                obs = result

            self.last_obs = obs
            return obs

        except Exception as e:
            print(f"Error in reset: {e}")
            # Return a blank observation
            blank_shape = self.observation_space.shape
            blank_obs = np.zeros(blank_shape, dtype=np.uint8)
            self.last_obs = blank_obs
            return blank_obs

def create_mario_env(render_mode=None):
    """
    Create and configure the Super Mario Bros environment with necessary preprocessing.

    Args:
        render_mode: Optional render mode (e.g., 'human' for visualization)

    Returns:
        env: The preprocessed environment ready for training.
    """
    try:
        # Create the base environment
        # SuperMarioBros-v0 provides 240x256x3 RGB image observation space
        if render_mode:
            env = gym_super_mario_bros.make(config.ENV, render_mode=render_mode)
        else:
            env = gym_super_mario_bros.make(config.ENV)

        # Wrap the environment to use the COMPLEX_MOVEMENT action space (12 actions)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        # Apply preprocessing wrappers sequentially:
        # 1. Apply frame skipping (4 frames) - speeds up training and reduces flickering
        env = MaxAndSkipEnv(env, skip=4)

        # 2. Convert the RGB observation to grayscale using our compatible wrapper
        env = CompatibleGrayScaleObservation(env, keep_dim=True) # Output shape (H, W, 1)

        # 3. Resize the observation to 84x84 using our compatible wrapper
        env = CompatibleResizeObservation(env, shape=(84, 84)) # Output shape (84, 84, 1)

        # 4. Stack 4 consecutive frames to capture temporal information using our compatible wrapper
        # FrameStack typically changes the shape to (num_stack, H, W) or (num_stack, H, W, C)
        # For grayscale with keep_dim=True, it should be (num_stack, H, W, 1)
        # However, PyTorch Conv2d expects (Batch, Channels, H, W), so we'll treat num_stack as Channels later
        env = CompatibleFrameStack(env, num_stack=4) # Output shape (4, 84, 84, 1)

        # 5. Add a final wrapper to ensure consistent API with old Gym format
        env = OldAPIWrapper(env)

        # Print environment information for verification
        print(f"Creating Mario environment with old Gym API format")
        print(f"Final Observation Space: {env.observation_space}")

        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise
