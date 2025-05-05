"""
Custom wrapper for Super Mario Bros environment to handle API differences.
This wrapper directly handles the NES-Py environment to avoid compatibility issues.
"""
import gym
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import config

class MarioEnvWrapper:
    """
    A custom wrapper for the Super Mario Bros environment that handles API differences.
    This wrapper directly handles the NES-Py environment to avoid compatibility issues.
    """
    def __init__(self, env_id="SuperMarioBros-v0", render_mode=None):
        """
        Initialize the Mario environment wrapper.

        Args:
            env_id: The environment ID to use
            render_mode: Optional render mode (e.g., 'human' for visualization)
        """
        # Create the base environment
        if render_mode:
            self.env = gym_super_mario_bros.make(env_id, render_mode=render_mode)
        else:
            self.env = gym_super_mario_bros.make(env_id)

        # Wrap the environment to use the COMPLEX_MOVEMENT action space (12 actions)
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)

        # Store the action and observation spaces
        self.action_space = self.env.action_space

        # Define the observation space for 4 stacked 84x84 grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84, 1), dtype=np.uint8
        )

        # Initialize frame buffer for frame stacking
        self.frames = []
        self.num_stack = 4

    def reset(self):
        """Reset the environment and preprocess the initial observation."""
        # Reset the done state
        self._done = False

        # Reset the environment
        try:
            # Try newer Gym API (returns obs, info)
            obs, info = self.env.reset()
        except ValueError:
            # Fall back to older Gym API (returns just obs)
            obs = self.env.reset()
            info = {}
        except Exception as e:
            print(f"Error in environment reset: {e}")
            # Create a blank observation as fallback
            obs = np.zeros((240, 256, 3), dtype=np.uint8)
            info = {}

        # Preprocess the observation
        processed_obs = self._preprocess_observation(obs)

        # Clear the frame buffer
        self.frames = []

        # Stack the initial observation num_stack times
        for _ in range(self.num_stack):
            self.frames.append(processed_obs)

        # Stack the frames
        stacked_frames = np.stack(self.frames, axis=0)

        return stacked_frames, info

    def step(self, action):
        """
        Take a step in the environment and preprocess the observation.

        Args:
            action: The action to take

        Returns:
            tuple: (stacked_frames, reward, terminated, truncated, info)
        """
        # Check if the environment is already done
        if hasattr(self, '_done') and self._done:
            print("Warning: Attempting to step in a done environment. Returning terminal state.")
            # Return the last observation with a terminal state
            if len(self.frames) > 0:
                stacked_frames = np.stack(self.frames, axis=0)
                return stacked_frames, 0.0, True, False, {}
            else:
                # If we don't have frames, create a blank observation
                blank_obs = np.zeros((4, 84, 84, 1), dtype=np.uint8)
                return blank_obs, 0.0, True, False, {}

        # Take 4 steps with the same action (frame skipping)
        total_reward = 0
        terminated = False
        truncated = False
        info = {}

        for i in range(4):  # Skip 4 frames
            try:
                # Try newer Gym API (returns obs, reward, terminated, truncated, info)
                obs, reward, term, trunc, info = self.env.step(action)
                terminated = term
                truncated = trunc
            except ValueError:
                # Fall back to older Gym API (returns obs, reward, done, info)
                try:
                    obs, reward, done, info = self.env.step(action)
                    terminated = done
                    truncated = False
                except Exception as e:
                    print(f"Error in environment step: {e}")
                    # Return safe defaults
                    if len(self.frames) > 0:
                        obs = np.zeros_like(self.frames[0])
                    else:
                        obs = np.zeros((84, 84, 1), dtype=np.uint8)
                    reward = -100
                    terminated = True
                    truncated = False
                    info = {}

            total_reward += reward
            if terminated or truncated:
                # Store the done state for future reference
                self._done = True
                break

        # Preprocess the observation
        processed_obs = self._preprocess_observation(obs)

        # Update the frame buffer
        self.frames.pop(0)
        self.frames.append(processed_obs)

        # Stack the frames
        stacked_frames = np.stack(self.frames, axis=0)

        # Always return 5 values (new Gym API format)
        return stacked_frames, total_reward, terminated, truncated, info

    def _preprocess_observation(self, observation):
        """
        Preprocess the observation (grayscale, resize, etc.).

        Args:
            observation: The raw observation from the environment

        Returns:
            np.ndarray: The preprocessed observation
        """
        try:
            # Check if the observation is already grayscale
            if len(observation.shape) == 2 or (len(observation.shape) == 3 and observation.shape[2] == 1):
                # Already grayscale, just resize
                if len(observation.shape) == 3:
                    # Remove the channel dimension for resizing
                    observation = observation[:, :, 0]

                # Resize to 84x84
                resized = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
            else:
                # Convert RGB to grayscale
                gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

                # Resize to 84x84
                resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

            # Add channel dimension
            processed = resized.reshape(84, 84, 1)

            return processed
        except Exception as e:
            print(f"Error in preprocessing observation: {e}")
            print(f"Observation shape: {observation.shape}")
            # Return a blank observation as fallback
            return np.zeros((84, 84, 1), dtype=np.uint8)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()

def create_mario_env(env_id=None, render_mode=None):
    """
    Create a new Mario environment with the custom wrapper.

    Args:
        env_id: The environment ID to use (defaults to config.ENV)
        render_mode: Optional render mode (e.g., 'human' for visualization)

    Returns:
        MarioEnvWrapper: The wrapped environment
    """
    if env_id is None:
        env_id = config.ENV

    print(f"Creating Mario environment with env_id={env_id}, render_mode={render_mode}")
    return MarioEnvWrapper(env_id=env_id, render_mode=render_mode)
