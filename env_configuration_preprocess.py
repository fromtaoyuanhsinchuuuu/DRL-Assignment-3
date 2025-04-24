# env_configuration_preprocess.py
import config

# 嘗試從 gymnasium 導入 (新版)
try:
    import gymnasium as gym
    # 嘗試從 transform_observation 子模塊導入
    try:
        from gymnasium.wrappers.transform_observation import GrayScaleObservation, ResizeObservation
        from gymnasium.wrappers.stack_observation import FrameStack  # FrameStack 可能在 stack_observation
        print("Using gymnasium with specific submodule imports")
    except ImportError:
        # 如果上面不行，再嘗試從其他可能的位置導入
        try:
            from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack  # 嘗試直接導入 (舊方式)
            print("Using gymnasium with direct wrapper imports")
        except ImportError:
            print("Error: Could not import necessary wrappers from gymnasium. Falling back to gym.")
            raise ImportError("Failed to import from gymnasium")
except ImportError:
    # 如果 gymnasium 不可用，回退到舊版 gym
    print("Gymnasium not available, falling back to gym")
    import gym
    try:
        from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
        print("Using gym with direct wrapper imports")
    except ImportError:
        print("Error: Could not import necessary wrappers from gym. Please check your installation.")
        import sys
        sys.exit(1)

# 導入 Super Mario Bros 環境
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
def create_mario_env():
    """
    Create and configure the Super Mario Bros environment with necessary preprocessing.
    Returns:
        env: The preprocessed environment ready for training.
    """
    # Create the base environment
    # SuperMarioBros-v0 provides 240x256x3 RGB image observation space
    env = gym_super_mario_bros.make(config.ENV)
    # Wrap the environment to use the COMPLEX_MOVEMENT action space (12 actions)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # Apply preprocessing wrappers sequentially:
    # 1. Convert the RGB observation to grayscale
    env = GrayScaleObservation(env, keep_dim=True) # Output shape (H, W, 1)
    # 2. Resize the observation to 84x84
    env = ResizeObservation(env, shape=(84, 84)) # Output shape (84, 84, 1)
    # 3. Stack 4 consecutive frames to capture temporal information
    # FrameStack typically changes the shape to (num_stack, H, W) or (num_stack, H, W, C)
    # For grayscale with keep_dim=True, it should be (num_stack, H, W, 1)
    # However, PyTorch Conv2d expects (Batch, Channels, H, W), so we'll treat num_stack as Channels later
    env = FrameStack(env, num_stack=4) # Output shape (4, 84, 84, 1)
    # Print environment information for verification
    print(f"Base Environment: SuperMarioBros-v0")
    print(f"Action Space Wrapper: JoypadSpace with COMPLEX_MOVEMENT")
    print(f"Preprocessing Wrappers Applied: GrayScaleObservation, ResizeObservation, FrameStack(4)")
    print(f"Final Observation Space: {env.observation_space}")
    print(f"Final Action Space: {env.action_space}")
    print(f"Expected Final Observation Shape (NumPy): (4, 84, 84, 1)")
    return env
