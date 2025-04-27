import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os
import pathlib
import csv
from tqdm import tqdm

# Import custom modules
from env_configuration_preprocess import create_mario_env
from ddqn_per_agent import DDQN_PER_Agent
import config

def main():
    """
    Main training loop for the DDQN+PER agent playing Super Mario Bros.
    """
    print("Starting Mario DDQN+PER training...")
    print(f"Using device: {config.DEVICE}")

    # Create and configure the Super Mario Bros environment
    env = create_mario_env()

    # Get state shape and action size
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    print(f"State shape: {state_shape}, Action size: {action_size}")

    # Create the agent
    agent = DDQN_PER_Agent(
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
    
    if os.path.exists('checkpoints'):
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('mario_ddqn_per_ep') and f.endswith('.pth')]
        if checkpoint_files:
            # Extract episode numbers and find the latest
            episode_nums = [int(f.split('ep')[1].split('.')[0]) for f in checkpoint_files]
            latest_episode = max(episode_nums)
            latest_checkpoint = f"checkpoints/mario_ddqn_per_ep{latest_episode}.pth"
            
            print(f"Loading latest checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            
            # Load model weights and optimizer state
            agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            
            # Load training history if available
            if 'rewards_history' in checkpoint:
                rewards_history = checkpoint['rewards_history']
            if 'avg_rewards_history' in checkpoint:
                avg_rewards_history = checkpoint['avg_rewards_history']
            
            # Continue from the next episode
            start_episode = latest_episode + 1
            
            # Try to load additional metrics from CSV if available
            try:
                if os.path.exists('logs/training_progress.csv'):
                    with open('logs/training_progress.csv', 'r') as f:
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
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Create CSV files for tracking progress
    progress_file = 'logs/training_progress.csv'
    with open(progress_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Steps', 'Reward', 'Avg_Reward', 'Epsilon', 'Max_X_Position', 'Episode_X_Position', 'Flags_Gotten'])

    # Create a separate file to track flag events
    flag_events_file = 'logs/flag_events.csv'
    with open(flag_events_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'X_Position', 'Steps', 'Reward'])

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
            # Track initial x position
            episode_max_x = info.get('x_pos', 0) if isinstance(info, dict) else 0
        except ValueError:
            # Fall back to older Gym API (returns just obs)
            state = env.reset()
            episode_max_x = 0

        total_reward = 0
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

                # Check if Mario got the flag
                if 'flag_get' in info and info['flag_get']:
                    flags_gotten += 1
                    x_pos = info.get('x_pos', 0)
                    progress_bar.write(f"🚩 FLAG GOTTEN in episode {episode}! Position: {x_pos}, Total flags: {flags_gotten}")

                    # Log flag event to CSV
                    with open('logs/flag_events.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([episode, x_pos, episode_steps, total_reward])

                # Track Mario's x position
                if 'x_pos' in info:
                    episode_max_x = max(episode_max_x, info['x_pos'])
            except ValueError:
                try:
                    # Fall back to older Gym API (returns obs, reward, done, info)
                    next_state, reward, done, info = env.step(action)
                    terminated = done
                    truncated = False

                    # Check if Mario got the flag
                    if 'flag_get' in info and info['flag_get']:
                        flags_gotten += 1
                        x_pos = info.get('x_pos', 0)
                        progress_bar.write(f"🚩 FLAG GOTTEN in episode {episode}! Position: {x_pos}, Total flags: {flags_gotten}")

                        # Log flag event to CSV
                        with open('logs/flag_events.csv', 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([episode, x_pos, episode_steps, total_reward])

                    # Track Mario's x position
                    if 'x_pos' in info:
                        episode_max_x = max(episode_max_x, info['x_pos'])
                except Exception as e:
                    # If we get a ValueError, the environment might be done
                    break
            # Store experience and train
            agent.step(state, action, reward, next_state, done, train_freq=config.TRAIN_FREQ_STEP)

            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_steps += 1
            total_steps += 1

            # Update target network periodically
            # soft update, update freq == 1
            if total_steps % config.TARGET_UPDATE_FREQ_STEP == 0:
                agent.update_target_network()


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
            writer.writerow([episode, episode_steps, total_reward, avg_reward, agent.epsilon, max_x_position, episode_max_x, flags_gotten])

        # Update progress bar with current metrics
        elapsed_time = time.time() - start_time
        progress_bar.set_postfix({
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
        })

        # Print detailed progress every 10 episodes
        if episode % 10 == 0:
            progress_bar.write(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Avg Reward (100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Alpha: {config.PER_ALPHA:.2f} | "
                  f"Beta: {agent.replay_buffer.beta:.2f} | "
                  f"Buffer Size: {len(agent.replay_buffer)} | "
                  f"Flags Gotten: {flags_gotten} | "
                  f"Max X: {max_x_position} | "
                  f"Episode X: {episode_max_x} | "
                  f"Time: {elapsed_time:.2f}s")

        # Save model periodically
        if episode % config.SAVE_FREQ_EPISODE == 0:
            # Create checkpoints directory if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)

            # Save full checkpoint with all training state
            checkpoint_path = f"checkpoints/mario_ddqn_per_ep{episode}.pth"
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
            weights_path = f"checkpoints/mario_qnet_ep{episode}.pth"
            torch.save(agent.q_net.state_dict(), weights_path)
            progress_bar.write(f"Q-network weights saved to {weights_path}")

            # Also save to the main path
            torch.save(agent.q_net.state_dict(), config.MODEL_SAVE_PATH)

    # Save final model
    torch.save(agent.q_net.state_dict(), config.MODEL_SAVE_PATH)
    progress_bar.write(f"Training completed. Final model saved to {config.MODEL_SAVE_PATH}")
    progress_bar.close()

    # Close environment
    env.close()

    # Plot training results
    plot_training_results(rewards_history, avg_rewards_history, epsilon_history, steps_history, max_x_position, flags_gotten)

def plot_training_results(rewards, avg_rewards, epsilons, steps, max_x_position, flags_gotten):
    """
    Plot training metrics.

    Args:
        rewards: List of episode rewards
        avg_rewards: List of average rewards (over 100 episodes)
        epsilons: List of epsilon values
        steps: List of episode steps
        max_x_position: Maximum x position reached
        flags_gotten: Number of flags gotten
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
        with open('logs/training_progress.csv', 'r') as f:
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
        with open('logs/flag_events.csv', 'r') as f:
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
    plt.savefig('logs/training_results.png')
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
        plt.savefig('logs/x_position_progress.png')
        plt.show()
    except Exception as e:
        plt.text(0.5, 0.5, f"Error loading X position data: {e}", ha='center', va='center')
        plt.savefig('logs/x_position_progress.png')
        plt.show()

if __name__ == "__main__":
    main()
