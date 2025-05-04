# Configuration parameters for Mario DDQN+PER training
import torch

# Environment settings
ENV = 'SuperMarioBros-v0'
MAX_STEPS_PER_EPISODE = 10000
NUM_EPISODES = 10000  # Mario needs more training episodes

# Agent hyperparameters
GAMMA = 0.99  # Discount factor
LR = 5e-5  # Learning rate (smaller for Mario)
TAU = 1e-4  # Soft update parameter
N_STEP = 10  # Number of steps for N-step bootstrapping

# Epsilon-greedy exploration
EPSILON_START = 1.0
EPSILON_MIN = 0.05  # Increased to maintain exploration
EPSILON_DECAY = 0.995 # Slower decay for Mario

# Prioritized Experience Replay parameters
BUFFER_CAPACITY = 100000
BATCH_SIZE = 32
PER_ALPHA = 0.6  # Prioritization amount
PER_BETA_START = 0.4  # Initial importance sampling weight
PER_BETA_FRAMES = 1000000  # Frames over which to anneal beta to 1.0

# Training settings
LEARNING_START_STEP = 1000  # Steps to fill buffer before learning starts
TRAIN_FREQ_STEP = 4  # Train every 4 steps
TARGET_UPDATE_FREQ_STEP = 10000  # Target network update frequency
SAVE_FREQ_EPISODE = 50  # Save model every 50 episodes
TARGET_UPDATE_TYPE = 'hard'  # 'soft', 'hard', or 'none'

# Checkpoint settings
LOAD_CHECKPOINT_EPISODE = 5400  # Episode to load checkpoint from (0 = don't load)
RESTART_EPSILON = 0.25  # Epsilon value to use when restarting training

# Model settings
MODEL_SAVE_PATH = 'mario_ddqn_per_qnet.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Reward Shaping Parameters ---
PROGRESS_WEIGHT = 0.5       # Reward per unit of X position increase
LIFE_LOSS_PENALTY = -250    # Penalty for losing a life
TIME_PENALTY_PER_STEP = -0.01 # Small penalty for each step

# Define milestones as X positions and their corresponding bonus
MILESTONES = {
    500: 50,
    1000: 80,
    1500: 120,
    2000: 180,
    2200: 500,
    3200: 700, # past first flag
    5500: 900,  # Deeper into World 1-2
    6000: 2000,  # Approaching the end of World 1-2
    6500: 3000,  # Reaching the end of World 1-2
    # Add more milestones for later levels if needed
}

# New reward shaping parameters
FLAG_GET_BONUS = 2000       # Extra bonus when flag is actually gotten (in addition to milestone)
COIN_REWARD = 50            # Reward for collecting a coin
ENEMY_DEFEAT_REWARD = 100   # Reward for defeating an enemy
SPEED_BONUS_THRESHOLD = 5   # X position increase per step threshold for speed bonus
SPEED_BONUS_REWARD = 10     # Reward for moving quickly
HEIGHT_EXPLORATION_REWARD = 0.2  # Reward multiplier for exploring higher y positions
JUMP_ACTION_REWARD = 0.5    # Small reward for using jump actions to encourage exploration
STUCK_PENALTY = -0.5        # Penalty per step when Mario is stuck at the same x position
STUCK_STEPS_THRESHOLD = 30  # Number of steps to consider Mario as "stuck"

# Enable/disable specific reward shaping components
USE_REWARD_SHAPING = True
USE_FLAG_BONUS = True
USE_COIN_REWARD = True
USE_ENEMY_DEFEAT_REWARD = True
USE_SPEED_BONUS = True
USE_HEIGHT_EXPLORATION = True
USE_JUMP_REWARD = True
USE_STUCK_PENALTY = True