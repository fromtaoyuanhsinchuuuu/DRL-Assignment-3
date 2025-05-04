# train_dueling_nstep.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os
import csv
from tqdm import tqdm

# Import custom modules
from env_configuration_preprocess import create_mario_env
from dueling_nstep_agent import Dueling_NSTEP_DDQN_Agent
import config

def calculate_shaping_reward(info, action, episode, progress_bar,
                            episode_max_x, previous_life_this_episode,
                            reached_milestones_this_episode, previous_y_pos,
                            previous_coins, previous_score, steps_at_current_x,
                            previous_x_pos, episode_steps):
    """
    Calculate the reward shaping component based on the environment info.

    Args:
        info: Environment info dictionary
        action: The action taken
        episode: Current episode number
        progress_bar: TQDM progress bar for printing
        episode_max_x: Maximum x position reached in this episode
        previous_life_this_episode: Previous life count in this episode
        reached_milestones_this_episode: Dictionary tracking reached milestones
        previous_y_pos: Previous y position
        previous_coins: Previous coin count
        previous_score: Previous score
        steps_at_current_x: Number of steps at current x position
        previous_x_pos: Previous x position
        episode_steps: Current step count in this episode

    Returns:
        tuple: (shaping_reward, updated_episode_max_x, updated_previous_life,
               updated_reached_milestones, updated_previous_y_pos,
               updated_previous_coins, updated_previous_score,
               updated_steps_at_current_x, updated_previous_x_pos)
    """
    shaping_reward = 0.0

    # Get current state values from info
    current_x_pos = info.get('x_pos', 0)
    current_life = info.get('life', 2)
    current_y_pos = info.get('y_pos', 0)
    current_coins = info.get('coins', 0)
    current_score = info.get('score', 0)

    # 1. Progress Bonus
    if current_x_pos > episode_max_x:
        shaping_reward += (current_x_pos - episode_max_x) * config.PROGRESS_WEIGHT
        episode_max_x = current_x_pos

    # 2. Life Preservation Penalty
    if current_life < previous_life_this_episode:
        shaping_reward += config.LIFE_LOSS_PENALTY
        progress_bar.write(f"💔 Life lost in episode {episode}! Lives remaining: {current_life}")
    previous_life_this_episode = current_life

    # 3. Milestone Bonus
    for m, reached in reached_milestones_this_episode.items():
        if not reached and current_x_pos >= m:
            shaping_reward += config.MILESTONES[m]
            reached_milestones_this_episode[m] = True
            # Only print for significant milestones (reward >= 500)
            if config.MILESTONES[m] >= 500:
                progress_bar.write(f"🎯 Milestone {m} reached in episode {episode}!")

    # 4. Time Penalty
    shaping_reward += config.TIME_PENALTY_PER_STEP

    # 5. Flag Get Bonus (in addition to milestone)
    if config.USE_FLAG_BONUS and 'flag_get' in info and info['flag_get']:
        shaping_reward += config.FLAG_GET_BONUS
        progress_bar.write(f"🚩 FLAG BONUS in episode {episode}! Extra reward: {config.FLAG_GET_BONUS}")

    # 6. Coin Collection Reward
    if config.USE_COIN_REWARD:
        coins_collected = current_coins - previous_coins
        if coins_collected > 0:
            coin_reward = coins_collected * config.COIN_REWARD
            shaping_reward += coin_reward
            # Only print if multiple coins collected or less frequent
            if coins_collected > 1 or episode_steps % 10 == 0:
                progress_bar.write(f"💰 Coins collected in episode {episode}: {coins_collected}")
        previous_coins = current_coins

    # 7. Enemy Defeat Reward
    if config.USE_ENEMY_DEFEAT_REWARD:
        # Score increases when enemies are defeated
        if current_score > previous_score:
            # Assuming score increases are primarily from defeating enemies
            shaping_reward += config.ENEMY_DEFEAT_REWARD
            progress_bar.write(f"💥 Enemy defeated in episode {episode}!")
        previous_score = current_score

    # 8. Speed Bonus
    if config.USE_SPEED_BONUS:
        x_progress = current_x_pos - previous_x_pos
        if x_progress >= config.SPEED_BONUS_THRESHOLD:
            shaping_reward += config.SPEED_BONUS_REWARD
            # No printing for speed bonus
        previous_x_pos = current_x_pos

    # 9. Height Exploration Reward
    if config.USE_HEIGHT_EXPLORATION:
        # Reward for exploring higher positions (smaller y values in Mario)
        if current_y_pos < previous_y_pos:
            height_reward = (previous_y_pos - current_y_pos) * config.HEIGHT_EXPLORATION_REWARD
            shaping_reward += height_reward
        previous_y_pos = current_y_pos

    # 10. Jump Action Reward
    if config.USE_JUMP_REWARD:
        # Check if the action involves jumping (actions 1, 2, 3, 5, 6, 7 in COMPLEX_MOVEMENT)
        jump_actions = [1, 2, 3, 5, 6, 7]
        if action in jump_actions:
            shaping_reward += config.JUMP_ACTION_REWARD

    # 11. Stuck Penalty
    if config.USE_STUCK_PENALTY:
        if current_x_pos == previous_x_pos:
            steps_at_current_x += 1
            if steps_at_current_x >= config.STUCK_STEPS_THRESHOLD:
                shaping_reward += config.STUCK_PENALTY
                # No printing for stuck penalty
        else:
            steps_at_current_x = 0
        previous_x_pos = current_x_pos

    return (shaping_reward, episode_max_x, previous_life_this_episode,
            reached_milestones_this_episode, previous_y_pos, previous_coins,
            previous_score, steps_at_current_x, previous_x_pos)

def main():
    """
    Main training loop for the Dueling DDQN+PER agent with N-step bootstrapping playing Super Mario Bros.
    """
    print("Starting Mario Dueling DDQN+PER with N-step training...")
    print(f"Using device: {config.DEVICE}")
    print(f"N-step value: {config.N_STEP}")

    # Create and configure the Super Mario Bros environment
    env = create_mario_env()

    # Get state shape and action size
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    print(f"State shape: {state_shape}, Action size: {action_size}")

    # Create the agent
    agent = Dueling_NSTEP_DDQN_Agent(
        state_shape=state_shape,
        action_size=action_size,
        buffer_capacity=config.BUFFER_CAPACITY,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        lr=config.LR,
        tau=config.TAU,
        per_alpha=config.PER_ALPHA,
        per_beta_start=config.PER_BETA_START,
        per_beta_frames=config.PER_BETA_FRAMES,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        n_step=config.N_STEP,
        device=config.DEVICE
    )

    # Load the latest checkpoint if available
    start_episode = 1
    rewards_history = []
    avg_rewards_history = []
    steps_history = []
    epsilon_history = []
    flags_gotten = 0
    max_x_position = 0

    # Define model save paths for dueling network
    model_save_path = 'mario_dueling_nstep_qnet.pth'
    checkpoints_dir = 'dueling_nstep_checkpoints'
    logs_dir = 'dueling_nstep_logs'

    # Check if we should load a specific checkpoint episode
    if config.LOAD_CHECKPOINT_EPISODE > 0:
        specific_checkpoint = f"{checkpoints_dir}/mario_dueling_nstep_ep{config.LOAD_CHECKPOINT_EPISODE}.pth"
        if os.path.exists(specific_checkpoint):
            print(f"Loading specified checkpoint: {specific_checkpoint}")
            checkpoint = torch.load(specific_checkpoint)
            latest_episode = config.LOAD_CHECKPOINT_EPISODE

            # If we're restarting training, reset epsilon to the specified value
            if config.RESTART_EPSILON > 0:
                print(f"Resetting epsilon to {config.RESTART_EPSILON}")
                agent.epsilon = config.RESTART_EPSILON
        else:
            print(f"Warning: Specified checkpoint {specific_checkpoint} not found.")
            print("Falling back to latest available checkpoint...")

            # Fall back to finding the latest checkpoint
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith('mario_dueling_nstep_ep') and f.endswith('.pth')]
                if checkpoint_files:
                    # Extract episode numbers and find the latest
                    episode_nums = []
                    for f in checkpoint_files:
                        try:
                            # Try to extract episode number safely
                            parts = f.split('ep')[1].split('.')
                            if parts and parts[0].isdigit():
                                episode_nums.append(int(parts[0]))
                        except (IndexError, ValueError):
                            print(f"Skipping file with invalid format: {f}")
                            continue

                    if not episode_nums:
                        print("No valid checkpoint files found with proper episode numbering.")
                        start_episode = 1
                    else:
                        latest_episode = max(episode_nums)
                        latest_checkpoint = f"{checkpoints_dir}/mario_dueling_nstep_ep{latest_episode}.pth"

                        print(f"Loading latest checkpoint: {latest_checkpoint}")
                        checkpoint = torch.load(latest_checkpoint)
    # If no specific checkpoint is requested, look for the latest one
    elif os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith('mario_dueling_nstep_ep') and f.endswith('.pth')]
        if checkpoint_files:
            # Extract episode numbers and find the latest
            episode_nums = []
            for f in checkpoint_files:
                try:
                    # Try to extract episode number safely
                    parts = f.split('ep')[1].split('.')
                    if parts and parts[0].isdigit():
                        episode_nums.append(int(parts[0]))
                except (IndexError, ValueError):
                    print(f"Skipping file with invalid format: {f}")
                    continue

            if not episode_nums:
                print("No valid checkpoint files found with proper episode numbering.")
                start_episode = 1
            else:
                latest_episode = max(episode_nums)
                latest_checkpoint = f"{checkpoints_dir}/mario_dueling_nstep_ep{latest_episode}.pth"

                print(f"Loading latest checkpoint: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint)

    # If we have a checkpoint, load it
    if 'checkpoint' in locals():
        # Load model weights and optimizer state
        agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Only set epsilon from checkpoint if we're not explicitly resetting it
        if not (config.LOAD_CHECKPOINT_EPISODE > 0 and config.RESTART_EPSILON > 0):
            agent.epsilon = checkpoint['epsilon']

        # Load training history if available
        if 'rewards_history' in checkpoint:
            rewards_history = checkpoint['rewards_history']
        if 'avg_rewards_history' in checkpoint:
            avg_rewards_history = checkpoint['avg_rewards_history']

        # Continue from the next episode
        if 'latest_episode' in locals():
            start_episode = latest_episode + 1

        # Try to load additional metrics from CSV if available
        try:
            if os.path.exists(f'{logs_dir}/training_progress.csv'):
                with open(f'{logs_dir}/training_progress.csv', 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        episode_num = int(row[0])
                        if episode_num == latest_episode:
                            # Get the latest metrics
                            max_x_position = float(row[5])
                            flags_gotten = int(row[7])
                            break
        except Exception as e:
            print(f"Error loading metrics from CSV: {e}")

        print(f"Resuming training from episode {start_episode}")
        print(f"Current epsilon: {agent.epsilon:.4f}")
        print(f"Max X position so far: {max_x_position}")
        print(f"Flags gotten so far: {flags_gotten}")

    # Initialize training metrics if not loaded from checkpoint
    if not rewards_history:
        rewards_history = []
    if not avg_rewards_history:
        avg_rewards_history = []
    if not steps_history:
        steps_history = []
    if not epsilon_history:
        epsilon_history = []

    # For tracking average reward over last 100 episodes
    recent_rewards = deque(maxlen=100)

    # Create directories for model checkpoints and logs if they don't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Create CSV files for tracking progress
    progress_file = f'{logs_dir}/training_progress.csv'
    file_mode = 'a' if os.path.exists(progress_file) else 'w'
    with open(progress_file, file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':                        # 只有第一次才寫表頭
            header = ['Episode','Steps','Reward','Avg_Reward','Epsilon',
                     'Max_X_Position','Episode_X_Position','Flags_Gotten']
            if config.USE_REWARD_SHAPING:
                header.append('Shaped_Reward')
            writer.writerow(header)

    # Create a separate file to track flag events
    flag_events_file = f'{logs_dir}/flag_events.csv'
    with open(flag_events_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Episode', 'X_Position', 'Steps', 'Reward']
        if config.USE_REWARD_SHAPING:
            header.append('Shaped_Reward')
        writer.writerow(header)

    # Main training loop
    total_steps = 0
    start_time = time.time()

    # Create progress bar with tqdm
    progress_bar = tqdm(range(start_episode, config.NUM_EPISODES + 1), desc="Training", unit="episode")

    for episode in progress_bar:
        # Reset environment and get initial state
        try:
            # Try newer Gym API (returns obs, info)
            state, info = env.reset()
            # Initialize tracking variables
            info_dict = info if isinstance(info, dict) else {}

            # Track initial x position
            episode_max_x = info_dict.get('x_pos', 0)

            # Track life count for reward shaping - initialize to 3 (Mario's starting lives)
            previous_life_this_episode = info_dict.get('life', 3)

            # Track reached milestones for reward shaping
            reached_milestones_this_episode = {m: False for m in config.MILESTONES.keys()} if config.USE_REWARD_SHAPING else {}

            # Track previous y position for height exploration reward
            previous_y_pos = info_dict.get('y_pos', 0)

            # Track previous coins for coin collection reward
            previous_coins = info_dict.get('coins', 0)

            # Track previous score for enemy defeat detection
            previous_score = info_dict.get('score', 0)

            # Track steps at current x position for stuck detection
            steps_at_current_x = 0

            # Track previous x position for speed bonus and stuck detection
            previous_x_pos = episode_max_x
        except ValueError:
            # Fall back to older Gym API (returns just obs)
            state = env.reset()

            # Initialize tracking variables with default values
            episode_max_x = 0
            previous_life_this_episode = 3  # Mario's starting lives
            reached_milestones_this_episode = {m: False for m in config.MILESTONES.keys()} if config.USE_REWARD_SHAPING else {}
            previous_y_pos = 0
            previous_coins = 0
            previous_score = 0
            steps_at_current_x = 0
            previous_x_pos = 0

        total_reward = 0  # Keep track of ORIGINAL reward for logging
        shaped_reward = 0  # Keep track of shaped reward
        episode_steps = 0

        # Episode loop
        done = False  # Initialize done flag
        for step in range(config.MAX_STEPS_PER_EPISODE):
            if done:  # Skip the rest of the episode if done
                break

            # Select action
            action = agent.get_action(state)

            # Take action in environment
            try:
                # Try newer Gym API (returns obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # Combine terminated and truncated flags

                # Original reward for logging
                original_reward = reward

                # Apply reward shaping if enabled
                if config.USE_REWARD_SHAPING:
                    # Call the reward shaping function
                    (shaping_reward, episode_max_x, previous_life_this_episode,
                     reached_milestones_this_episode, previous_y_pos, previous_coins,
                     previous_score, steps_at_current_x, previous_x_pos) = calculate_shaping_reward(
                        info, action, episode, progress_bar, episode_max_x,
                        previous_life_this_episode, reached_milestones_this_episode,
                        previous_y_pos, previous_coins, previous_score,
                        steps_at_current_x, previous_x_pos, episode_steps
                    )

                    # Combine original reward and shaping reward
                    reward = original_reward + shaping_reward
                    shaped_reward += shaping_reward

                # Check if Mario got the flag
                if 'flag_get' in info and info['flag_get']:
                    flags_gotten += 1
                    x_pos = info.get('x_pos', 0)
                    progress_bar.write(f"🚩 FLAG GOTTEN in episode {episode}! Position: {x_pos}, Total flags: {flags_gotten}")

                    # Log flag event to CSV
                    with open(flag_events_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row_data = [episode, x_pos, episode_steps, total_reward]
                        if config.USE_REWARD_SHAPING:
                            row_data.append(shaped_reward)
                        writer.writerow(row_data)

                # Track Mario's x position (this is now redundant as it's handled in the reward shaping function)
                if 'x_pos' in info and not config.USE_REWARD_SHAPING:
                    episode_max_x = max(episode_max_x, info['x_pos'])

            except ValueError:
                # Fall back to older Gym API (returns obs, reward, done, info)
                try:
                    next_state, reward, done, info = env.step(action)
                    terminated = done
                    truncated = False

                    # Original reward for logging
                    original_reward = reward

                    # Apply reward shaping if enabled
                    if config.USE_REWARD_SHAPING:
                        # Call the reward shaping function
                        (shaping_reward, episode_max_x, previous_life_this_episode,
                         reached_milestones_this_episode, previous_y_pos, previous_coins,
                         previous_score, steps_at_current_x, previous_x_pos) = calculate_shaping_reward(
                            info, action, episode, progress_bar, episode_max_x,
                            previous_life_this_episode, reached_milestones_this_episode,
                            previous_y_pos, previous_coins, previous_score,
                            steps_at_current_x, previous_x_pos, episode_steps
                        )

                        # Combine original reward and shaping reward
                        reward = original_reward + shaping_reward
                        shaped_reward += shaping_reward

                    # Check if Mario got the flag
                    if 'flag_get' in info and info['flag_get']:
                        flags_gotten += 1
                        x_pos = info.get('x_pos', 0)
                        progress_bar.write(f"🚩 FLAG GOTTEN in episode {episode}! Position: {x_pos}, Total flags: {flags_gotten}")

                        # Log flag event to CSV
                        with open(flag_events_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            row_data = [episode, x_pos, episode_steps, total_reward]
                            if config.USE_REWARD_SHAPING:
                                row_data.append(shaped_reward)
                            writer.writerow(row_data)

                    # Track Mario's x position (this is now redundant as it's handled in the reward shaping function)
                    if 'x_pos' in info and not config.USE_REWARD_SHAPING:
                        episode_max_x = max(episode_max_x, info['x_pos'])

                except Exception as e:
                    # If we get an error, the environment might be done
                    progress_bar.write(f"Environment step error: {e}")
                    next_state = state  # Use current state as next state
                    reward = -100  # Example penalty
                    original_reward = -100
                    done = True

            # Store experience with MODIFIED reward
            agent.step(state, action, reward, next_state, done, train_freq=config.TRAIN_FREQ_STEP)

            # Update state and metrics using ORIGINAL reward for logging
            state = next_state
            total_reward += original_reward  # Log the ORIGINAL reward
            episode_steps += 1
            total_steps += 1

            # Update target network periodically
            # soft update, update freq == 1
            if total_steps % config.TARGET_UPDATE_FREQ_STEP == 0:
                print("Update Target Network")
                agent.update_target_network(update_type=config.TARGET_UPDATE_TYPE)


            # Break if episode is done
            if done:
                break

        # Decay epsilon
        if agent.epsilon > config.EPSILON_MIN:
            agent.epsilon *= config.EPSILON_DECAY

        # Record metrics
        rewards_history.append(total_reward)
        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)
        avg_rewards_history.append(avg_reward)
        steps_history.append(episode_steps)
        epsilon_history.append(agent.epsilon)

        # Update max x position
        max_x_position = max(max_x_position, episode_max_x)

        # Print progress information if Mario made significant progress
        if episode_max_x > 50:  # Only print if Mario moved significantly
            progress_bar.write(f"Mario reached x position {episode_max_x} in episode {episode} | Steps: {episode_steps} | Reward: {total_reward:.2f} (max ever: {max_x_position})")

        # Log progress to CSV file
        with open(progress_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row_data = [episode, episode_steps, total_reward, avg_reward, agent.epsilon,
                       max_x_position, episode_max_x, flags_gotten]
            if config.USE_REWARD_SHAPING:
                row_data.append(shaped_reward)
            writer.writerow(row_data)

        # Update progress bar with current metrics
        elapsed_time = time.time() - start_time
        postfix_dict = {
            'steps': episode_steps,
            'reward': f"{total_reward:.2f}",
            'avg_reward': f"{avg_reward:.2f}",
            'epsilon': f"{agent.epsilon:.4f}",
            'alpha': f"{config.PER_ALPHA:.2f}",
            'beta': f"{agent.replay_buffer.beta:.2f}",
            'buffer': len(agent.replay_buffer),
            'flags': flags_gotten,
            'max_x': max_x_position,
            'time': f"{elapsed_time:.2f}s"
        }

        # Add shaped reward info if reward shaping is enabled
        if config.USE_REWARD_SHAPING:
            postfix_dict['shaped'] = f"{shaped_reward:.2f}"

        progress_bar.set_postfix(postfix_dict)

        # Print detailed progress every 10 episodes
        if episode % 10 == 0:
            progress_info = (
                f"Episode {episode}/{config.NUM_EPISODES} | "
                f"Steps: {episode_steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg Reward (100): {avg_reward:.2f} | "
            )

            # Add shaped reward info if reward shaping is enabled
            if config.USE_REWARD_SHAPING:
                progress_info += f"Shaped Reward: {shaped_reward:.2f} | "

            progress_info += (
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Alpha: {config.PER_ALPHA:.2f} | "
                f"Beta: {agent.replay_buffer.beta:.2f} | "
                f"Buffer Size: {len(agent.replay_buffer)} | "
                f"Flags Gotten: {flags_gotten} | "
                f"Max X: {max_x_position} | "
                f"Episode X: {episode_max_x} | "
                f"Time: {elapsed_time:.2f}s"
            )

            progress_bar.write(progress_info)

        # Save model periodically
        if episode % config.SAVE_FREQ_EPISODE == 0:
            # Create checkpoints directory if it doesn't exist
            os.makedirs(checkpoints_dir, exist_ok=True)

            # Save full checkpoint with all training state
            checkpoint_path = f"{checkpoints_dir}/mario_dueling_nstep_ep{episode}.pth"
            torch.save({
                'episode': episode,
                'q_net_state_dict': agent.q_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'rewards_history': rewards_history,
                'avg_rewards_history': avg_rewards_history
            }, checkpoint_path)
            progress_bar.write(f"Full checkpoint saved to {checkpoint_path}")

            # Save just the Q-network weights for easier loading in student agent
            weights_path = f"{checkpoints_dir}/mario_dueling_nstep_qnet_ep{episode}.pth"
            torch.save(agent.q_net.state_dict(), weights_path)
            progress_bar.write(f"Q-network weights saved to {weights_path}")

            # Also save to the main path
            torch.save(agent.q_net.state_dict(), model_save_path)

    # Save final model
    torch.save(agent.q_net.state_dict(), model_save_path)
    progress_bar.write(f"Training completed. Final model saved to {model_save_path}")
    progress_bar.close()

    # Close environment
    env.close()

    # Plot training results
    plot_training_results(rewards_history, avg_rewards_history, epsilon_history, steps_history, max_x_position, flags_gotten, logs_dir)

def plot_training_results(rewards, avg_rewards, epsilons, steps, max_x_position, flags_gotten, logs_dir):
    """
    Plot training metrics.

    Args:
        rewards: List of episode rewards
        avg_rewards: List of average rewards (over 100 episodes)
        epsilons: List of epsilon values
        steps: List of episode steps
        max_x_position: Maximum x position reached
        flags_gotten: Number of flags gotten
        logs_dir: Directory to save logs
    """
    # Create a figure with 6 subplots (3x2 grid)
    plt.figure(figsize=(15, 15))

    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Reward', 'Avg Reward (100)'])

    # Plot epsilon decay
    plt.subplot(3, 2, 2)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    # Plot steps per episode
    plt.subplot(3, 2, 3)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    # Plot average reward trend
    plt.subplot(3, 2, 4)
    window_size = min(100, len(avg_rewards))
    avg_reward_trend = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(avg_reward_trend)
    plt.title(f'Average Reward Trend (Window: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    # Plot X position progress from CSV
    plt.subplot(3, 2, 5)
    try:
        # Read the CSV file
        episodes = []
        x_positions = []
        with open(f'{logs_dir}/training_progress.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                episodes.append(int(row[0]))
                x_positions.append(float(row[6]))  # Episode_X_Position

        plt.plot(episodes, x_positions)
        plt.axhline(y=max_x_position, color='r', linestyle='--', label=f'Max: {max_x_position}')
        plt.title(f'X Position Progress (Max: {max_x_position})')
        plt.xlabel('Episode')
        plt.ylabel('X Position')
        plt.legend()
    except Exception as e:
        plt.text(0.5, 0.5, f"Error loading X position data: {e}", ha='center', va='center')

    # Plot flags gotten
    plt.subplot(3, 2, 6)
    try:
        # Read the flag events CSV file
        flag_episodes = []
        flag_positions = []
        with open(f'{logs_dir}/flag_events.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                flag_episodes.append(int(row[0]))
                flag_positions.append(float(row[1]))

        if flag_episodes:  # If we have flag events
            plt.scatter(flag_episodes, flag_positions, color='red', marker='*', s=100, label='Flag Gotten')
            plt.title(f'Flag Events (Total: {flags_gotten})')
            plt.xlabel('Episode')
            plt.ylabel('X Position')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, f"No Flags Gotten Yet", ha='center', va='center', fontsize=20)
            plt.axis('off')
    except Exception as e:
        plt.text(0.5, 0.5, f"Total Flags Gotten: {flags_gotten}\nError loading flag data: {e}", ha='center', va='center', fontsize=15)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{logs_dir}/training_results.png')
    plt.show()

    # Also create a separate plot just for X position progress with flag events
    plt.figure(figsize=(10, 6))
    try:
        # Plot X position progress
        plt.plot(episodes, x_positions, label='X Position')
        plt.axhline(y=max_x_position, color='r', linestyle='--', label=f'Max: {max_x_position}')

        # Add flag events as stars
        try:
            if flag_episodes:  # If we have flag events
                plt.scatter(flag_episodes, flag_positions, color='red', marker='*', s=150, label='Flag Gotten')
        except:
            pass  # If flag_episodes is not defined, just skip it

        plt.title(f'Mario X Position Progress (Max: {max_x_position}, Flags: {flags_gotten})')
        plt.xlabel('Episode')
        plt.ylabel('X Position')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{logs_dir}/x_position_progress.png')
        plt.show()
    except Exception as e:
        plt.text(0.5, 0.5, f"Error loading X position data: {e}", ha='center', va='center')
        plt.savefig(f'{logs_dir}/x_position_progress.png')
        plt.show()

if __name__ == "__main__":
    main()
