# config.py
# Configuration parameters for Mario DDQN+PER training
import torch

# Environment settings
ENV = 'SuperMarioBros-v0'
MAX_STEPS_PER_EPISODE = 10000
NUM_EPISODES = 10000  # Mario needs more training episodes

# Agent hyperparameters
GAMMA = 0.99  # Discount factor
LR = 1e-4  # Learning rate (increased from 1e-4 to 3e-4 for faster learning)
TAU = 1e-3  # Soft update parameter (increased from 1e-4)
N_STEP = 5  # Number of steps for N-step bootstrapping

# Noisy Network settings
USE_NOISY_NET = True  # Whether to use Noisy Networks for exploration

# Epsilon-greedy exploration
EPSILON_START = 1.0
EPSILON_MIN = 0.1  # Increased from 0.05 to maintain more exploration
EPSILON_DECAY = 0.9995  # Even slower decay for Mario (increased from 0.995)

# Prioritized Experience Replay parameters
BUFFER_CAPACITY = 500000
BATCH_SIZE = 256  # Increased from 64 for more stable learning
PER_ALPHA = 0.6  # Prioritization amount
PER_BETA_START = 0.4  # Initial importance sampling weight
PER_BETA_FRAMES = 1000000  # Frames over which to anneal beta to 1.0

# Training settings
LEARNING_START_STEP = 10000  # Steps to fill buffer before learning starts
TRAIN_FREQ_STEP = 3  # Train every 4 steps
TARGET_UPDATE_FREQ_STEP = 5000  # Target network update frequency (reduced from 10000)
SAVE_FREQ_EPISODE = 50  # Save model every 50 episodes
TARGET_UPDATE_TYPE = 'hard'  # 'soft', 'hard', or 'none'

# Checkpoint settings
LOAD_CHECKPOINT_EPISODE = 850  # Episode to load checkpoint from (0 = don't load)
RESTART_EPSILON = 0.25  # Epsilon value to use when restarting training

# Model settings
MODEL_SAVE_PATH = 'mario_ddqn_per_qnet.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Reward Shaping Parameters ---
PROGRESS_WEIGHT = 0.2       # Increased from 0.3 to 1.0 - Reward per unit of X position increase
LIFE_LOSS_PENALTY = -300    # Increased from -50 to -100 - Penalty for losing a life
TIME_PENALTY_PER_STEP = -0.01 # Increased from -0.01 to -0.05 - Small penalty for each step

# Define milestones as X positions and their corresponding bonus
MILESTONES = {
    100: 50,    # Added early milestone to encourage initial progress
    200: 100,   # Added early milestone
    300: 150,   # Added early milestone
    500: 200,   # Increased from 50 to 200
    1000: 300,  # Increased from 80 to 300
    1500: 500,  # Increased from 120 to 500
    2000: 800,  # Increased from 180 to 800
    2200: 1000, # Increased from 500 to 1000
    3200: 1500, # Increased from 700 to 1500
    5500: 2000, # Increased from 900 to 2000
    6000: 3000, # Increased from 2000 to 3000
    6500: 5000, # Increased from 3000 to 5000
}

# New reward shaping parameters
FLAG_GET_BONUS = 5000       # Increased from 2000 to 5000 - Extra bonus when flag is actually gotten
COIN_REWARD = 100           # Increased from 50 to 100 - Reward for collecting a coin
ENEMY_DEFEAT_REWARD = 200   # Increased from 100 to 200 - Reward for defeating an enemy
SPEED_BONUS_THRESHOLD = 3   # Decreased from 5 to 3 - X position increase per step threshold for speed bonus
SPEED_BONUS_REWARD = 20     # Increased from 10 to 20 - Reward for moving quickly
HEIGHT_EXPLORATION_REWARD = 0.5  # Increased from 0.2 to 0.5 - Reward multiplier for exploring higher y positions
JUMP_ACTION_REWARD = 1.0    # Increased from 0.5 to 1.0 - Small reward for using jump actions
STUCK_PENALTY = -1.0        # Increased from -0.3 to -1.0 - Penalty per step when Mario is stuck
STUCK_STEPS_THRESHOLD = 10  # Decreased from 20 to 10 - Number of steps to consider Mario as "stuck"

# Enable/disable specific reward shaping components
USE_REWARD_SHAPING = True
USE_FLAG_BONUS = True
USE_COIN_REWARD = True
USE_ENEMY_DEFEAT_REWARD = True
USE_SPEED_BONUS = True
USE_HEIGHT_EXPLORATION = False  # Enabled to encourage exploration
USE_JUMP_REWARD = False         # Enabled to encourage exploration
USE_STUCK_PENALTY = True

# Visualization settings
RENDER_EPISODES = False  # Enable/disable rendering episodes every 10 episodes
RENDER_DELAY = 0.05     # Delay between frames when rendering (seconds)