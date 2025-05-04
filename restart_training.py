#!/usr/bin/env python3
# restart_training.py
import torch
import os
import argparse
from dueling_nstep_agent import Dueling_NSTEP_DDQN_Agent
import config

def main():
    parser = argparse.ArgumentParser(description='Restart training from a checkpoint with reset epsilon')
    parser.add_argument('--episode', type=int, default=2000, help='Episode number to load from')
    parser.add_argument('--epsilon', type=float, default=0.2, help='New epsilon value to start with')
    args = parser.parse_args()

    # Define checkpoint directory and path
    checkpoints_dir = 'dueling_nstep_checkpoints'
    checkpoint_path = f"{checkpoints_dir}/mario_dueling_nstep_ep{args.episode}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found!")

        # Find all valid checkpoint files
        valid_checkpoints = []
        if os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.startswith('mario_dueling_nstep_ep') and f.endswith('.pth'):
                    try:
                        parts = f.split('ep')[1].split('.')
                        if parts and parts[0].isdigit():
                            valid_checkpoints.append((int(parts[0]), f))
                    except (IndexError, ValueError):
                        continue

        if valid_checkpoints:
            # Sort by episode number
            valid_checkpoints.sort(reverse=True)
            print("Available checkpoints:")
            for ep_num, cp in valid_checkpoints[:10]:  # Show top 10
                print(f"  {cp} (Episode {ep_num})")

            # Suggest using the latest checkpoint
            latest_ep, latest_cp = valid_checkpoints[0]
            print(f"\nSuggestion: Use the latest checkpoint (Episode {latest_ep}):")
            print(f"python restart_training.py --episode {latest_ep} --epsilon {args.epsilon}")
        else:
            print("No valid checkpoints found in directory.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Create a temporary agent to load the checkpoint
    # We need to know the state shape and action size from the checkpoint
    # For Mario, these are typically (4, 84, 84, 1) and 12
    state_shape = (4, 84, 84, 1)  # Updated to match the actual shape used in the environment
    action_size = 12

    # Create agent with the new n-step value from config
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

    # Load model weights and optimizer state
    agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Reset epsilon to the specified value
    agent.epsilon = args.epsilon
    print(f"Reset epsilon to {agent.epsilon}")

    # Save the modified checkpoint
    modified_checkpoint = {
        'episode': args.episode,
        'q_net_state_dict': agent.q_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'rewards_history': checkpoint.get('rewards_history', []),
        'avg_rewards_history': checkpoint.get('avg_rewards_history', [])
    }

    # Save to the same path
    torch.save(modified_checkpoint, checkpoint_path)
    print(f"Saved modified checkpoint with reset epsilon to {checkpoint_path}")
    print("\nNow you can run the training script to continue training:")
    print("python train_dueling_nstep.py")

if __name__ == "__main__":
    main()
