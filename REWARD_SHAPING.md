# Enhanced Reward Shaping for Super Mario Bros

This document explains the enhanced reward shaping functions implemented to improve the learning process of the reinforcement learning agent in the Super Mario Bros environment.

## Reward Shaping Functions

### 1. Progress Bonus
- **Description**: Rewards the agent for making forward progress in the level.
- **Implementation**: Provides a reward proportional to the increase in x-position.
- **Parameter**: `PROGRESS_WEIGHT` (default: 0.5)
- **Purpose**: Encourages the agent to move right and make progress through the level.

### 2. Life Preservation Penalty
- **Description**: Penalizes the agent for losing a life.
- **Implementation**: Applies a negative reward when Mario loses a life.
- **Parameter**: `LIFE_LOSS_PENALTY` (default: -250)
- **Purpose**: Teaches the agent to avoid dangerous situations and enemies.

### 3. Milestone Bonus
- **Description**: Rewards the agent for reaching specific x-positions in the level.
- **Implementation**: Provides a one-time bonus when Mario reaches predefined x-coordinates.
- **Parameter**: `MILESTONES` dictionary mapping x-positions to reward values.
- **Purpose**: Provides intermediate goals and helps guide the agent through the level.

### 4. Time Penalty
- **Description**: Small penalty for each step taken.
- **Implementation**: Applies a small negative reward for each step.
- **Parameter**: `TIME_PENALTY_PER_STEP` (default: -0.01)
- **Purpose**: Encourages the agent to complete the level efficiently.

### 5. Flag Get Bonus
- **Description**: Extra bonus when Mario actually gets the flag (in addition to milestone).
- **Implementation**: Provides a large reward when the 'flag_get' event occurs.
- **Parameter**: `FLAG_GET_BONUS` (default: 2000)
- **Purpose**: Strongly reinforces the ultimate goal of completing the level.

### 6. Coin Collection Reward
- **Description**: Rewards the agent for collecting coins.
- **Implementation**: Provides a reward for each coin collected.
- **Parameter**: `COIN_REWARD` (default: 50)
- **Purpose**: Encourages exploration and teaches the agent to recognize and collect valuable items.

### 7. Enemy Defeat Reward
- **Description**: Rewards the agent for defeating enemies.
- **Implementation**: Provides a reward when the score increases (typically from defeating enemies).
- **Parameter**: `ENEMY_DEFEAT_REWARD` (default: 100)
- **Purpose**: Encourages the agent to engage with and defeat enemies rather than just avoiding them.

### 8. Speed Bonus
- **Description**: Rewards the agent for moving quickly through the level.
- **Implementation**: Provides a bonus when Mario makes significant progress in a single step.
- **Parameters**: `SPEED_BONUS_THRESHOLD` (default: 5), `SPEED_BONUS_REWARD` (default: 10)
- **Purpose**: Encourages the agent to move efficiently and maintain momentum.

### 9. Height Exploration Reward
- **Description**: Rewards the agent for exploring higher positions in the level.
- **Implementation**: Provides a reward proportional to upward movement.
- **Parameter**: `HEIGHT_EXPLORATION_REWARD` (default: 0.2)
- **Purpose**: Encourages the agent to jump and explore the vertical space of the level.

### 10. Jump Action Reward
- **Description**: Small reward for using jump actions.
- **Implementation**: Provides a small reward when the agent selects actions that involve jumping.
- **Parameter**: `JUMP_ACTION_REWARD` (default: 0.5)
- **Purpose**: Encourages the agent to use jumping actions, which are essential for navigating the Mario environment.

### 11. Stuck Penalty
- **Description**: Penalizes the agent for remaining at the same x-position for too long.
- **Implementation**: Applies a negative reward when Mario stays at the same x-position for a certain number of steps.
- **Parameters**: `STUCK_PENALTY` (default: -0.5), `STUCK_STEPS_THRESHOLD` (default: 30)
- **Purpose**: Discourages the agent from getting stuck or being indecisive.

## Configuration

All reward shaping parameters can be configured in the `config.py` file. Each reward shaping function can be enabled or disabled individually using the corresponding `USE_*` parameters.

Example:
```python
# Enable/disable specific reward shaping components
USE_REWARD_SHAPING = True
USE_FLAG_BONUS = True
USE_COIN_REWARD = True
USE_ENEMY_DEFEAT_REWARD = True
USE_SPEED_BONUS = True
USE_HEIGHT_EXPLORATION = True
USE_JUMP_REWARD = True
USE_STUCK_PENALTY = True
```

## Benefits of Enhanced Reward Shaping

1. **Faster Learning**: By providing more frequent and informative feedback, the agent can learn more quickly.
2. **Better Exploration**: Rewards for coins, height exploration, and jumping encourage the agent to explore the environment more thoroughly.
3. **More Robust Behavior**: Penalties for getting stuck and losing lives help the agent develop more robust strategies.
4. **Goal-Oriented Learning**: The combination of milestone bonuses and flag get bonus helps the agent understand and prioritize the ultimate goal.

## Tuning Recommendations

- Start with the default values and observe the agent's behavior.
- If the agent is too cautious, consider reducing the `LIFE_LOSS_PENALTY`.
- If the agent is not exploring enough, increase the `HEIGHT_EXPLORATION_REWARD` and `JUMP_ACTION_REWARD`.
- If the agent gets stuck frequently, increase the `STUCK_PENALTY`.
- The `PROGRESS_WEIGHT` is one of the most important parameters - adjust it based on how much you want to emphasize forward progress.
