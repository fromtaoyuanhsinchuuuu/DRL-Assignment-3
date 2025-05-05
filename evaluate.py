#!/usr/bin/env python3
# evaluate.py
import os
import gym
import numpy as np
import time
from student_agent import Agent
from env_configuration_preprocess import create_mario_env

def main():
    """
    Evaluate the trained agent on the Mario environment.
    """
    print("Starting Mario agent evaluation...")

    # Create and configure the Super Mario Bros environment
    env = create_mario_env(render_mode='human')  # Set render_mode to 'human' to visualize

    # Create the agent
    agent = Agent()

    # Check if the agent loaded a model
    if not agent.model_loaded:
        print("Warning: Agent could not load a model. Evaluation will use random actions.")

    # Run evaluation episodes
    num_episodes = 5
    total_rewards = []
    max_x_positions = []
    flags_gotten = 0

    for episode in range(1, num_episodes + 1):
        # Reset environment and get initial state (old API format - returns just state)
        state = env.reset()
        episode_max_x = 0

        total_reward = 0
        episode_steps = 0
        done = False

        print(f"\nStarting Episode {episode}/{num_episodes}")
        start_time = time.time()

        # Episode loop
        while not done and episode_steps < 10000:  # Limit to 10000 steps per episode
            # Select action
            action = agent.act(state)

            # Take action in environment
            try:
                # Try newer Gym API (returns obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Check if Mario got the flag
                if 'flag_get' in info and info['flag_get']:
                    flags_gotten += 1
                    x_pos = info.get('x_pos', 0)
                    print(f"🚩 FLAG GOTTEN in episode {episode}! Position: {x_pos}")

                # Track Mario's x position
                if 'x_pos' in info:
                    episode_max_x = max(episode_max_x, info['x_pos'])

            except ValueError:
                try:
                    # Fall back to older Gym API (returns obs, reward, done, info)
                    next_state, reward, done, info = env.step(action)

                    # Check if Mario got the flag
                    if 'flag_get' in info and info['flag_get']:
                        flags_gotten += 1
                        x_pos = info.get('x_pos', 0)
                        print(f"🚩 FLAG GOTTEN in episode {episode}! Position: {x_pos}")

                    # Track Mario's x position
                    if 'x_pos' in info:
                        episode_max_x = max(episode_max_x, info['x_pos'])

                except Exception as e:
                    print(f"Error during step: {e}")
                    break

            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_steps += 1

            # Print progress every 100 steps
            if episode_steps % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode} - Steps: {episode_steps}, Reward: {total_reward:.2f}, "
                      f"X Position: {episode_max_x}, Time: {elapsed_time:.2f}s")

        # Episode summary
        elapsed_time = time.time() - start_time
        print(f"\nEpisode {episode} Summary:")
        print(f"Steps: {episode_steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Max X Position: {episode_max_x}")
        print(f"Time: {elapsed_time:.2f}s")

        # Record metrics
        total_rewards.append(total_reward)
        max_x_positions.append(episode_max_x)

    # Overall evaluation summary
    print("\n===== Evaluation Summary =====")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Max X Position: {np.mean(max_x_positions):.2f}")
    print(f"Flags Gotten: {flags_gotten}/{num_episodes}")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
