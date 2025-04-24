# Configuration parameters for Mario DDQN+PER training
import torch

# Environment settings
ENV = 'SuperMarioBros-v0'
MAX_STEPS_PER_EPISODE = 10000
NUM_EPISODES = 10000  # Mario needs more training episodes

# Agent hyperparameters
GAMMA = 0.99  # Discount factor
LR = 5e-5  # Learning rate (smaller for Mario)
TAU = 1e-3  # Soft update parameter

# Epsilon-greedy exploration
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999  # Slower decay for Mario

# Prioritized Experience Replay parameters
BUFFER_CAPACITY = 100000
BATCH_SIZE = 32
PER_ALPHA = 0.6  # Prioritization amount
PER_BETA_START = 0.4  # Initial importance sampling weight
PER_BETA_FRAMES = 1000000  # Frames over which to anneal beta to 1.0

# Training settings
LEARNING_START_STEP = 10000  # Steps to fill buffer before learning starts
TRAIN_FREQ_STEP = 4  # Train every 4 steps
TARGET_UPDATE_FREQ_STEP = 1000  # Target network update frequency
SAVE_FREQ_EPISODE = 50  # Save model every 50 episodes

# Model settings
MODEL_SAVE_PATH = 'mario_ddqn_per_qnet.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'