# N-step Bootstrapping Implementation for Super Mario Bros

This directory contains an implementation of N-step bootstrapping for the Dueling Double Deep Q-Network (DDQN) with Prioritized Experience Replay (PER) for playing Super Mario Bros.

## N-step Bootstrapping

N-step bootstrapping is a technique that extends the standard 1-step TD learning by looking ahead N steps before bootstrapping with the value function. Instead of updating Q-values using just the immediate reward and the estimated value of the next state, we use the sum of N rewards and then bootstrap with the estimated value of the state N steps ahead.

The N-step return is defined as:
```
R_t^n = r_{t+1} + γr_{t+2} + γ^2r_{t+3} + ... + γ^{n-1}r_{t+n} + γ^n V(s_{t+n})
```

Where:
- `R_t^n` is the N-step return starting from time t
- `r_{t+i}` is the reward received at time t+i
- `γ` is the discount factor
- `V(s_{t+n})` is the estimated value of the state at time t+n

This approach allows the agent to learn from a longer sequence of rewards before relying on its own value estimates, which can lead to faster learning and better performance, especially in environments with sparse rewards.

## Files

- `per_buffer.py`: Modified to support N-step returns in the Prioritized Experience Replay buffer.
- `dueling_nstep_agent.py`: Implementation of the Dueling DDQN agent with PER and N-step bootstrapping.
- `train_dueling_nstep.py`: Training script for the Dueling DDQN agent with N-step bootstrapping.
- `dueling_nstep_student_agent.py`: Agent implementation for evaluation using the trained model.

## N-step Implementation Details

The N-step bootstrapping is implemented in two main components:

1. **Prioritized Experience Replay Buffer**:
   - The buffer now stores sequences of experiences and calculates N-step returns.
   - When adding a new experience, it's added to a temporary N-step buffer.
   - When sampling from the buffer, N-step returns are calculated by summing discounted rewards over N steps.
   - The target for learning is now based on the N-step return rather than just the immediate reward.

2. **Agent Training**:
   - The agent's training method is modified to use the N-step returns from the buffer.
   - The bootstrap value is now discounted by γ^N instead of just γ.
   - TD errors are calculated based on the difference between the current Q-value and the N-step target.

## Training the Agent

To train the agent with N-step bootstrapping, run:

```bash
python train_dueling_nstep.py
```

The training script will create checkpoints in the `dueling_nstep_checkpoints` directory and logs in the `dueling_nstep_logs` directory.

## Using the Agent

To use the trained agent for evaluation, use the `dueling_nstep_student_agent.py` file. This agent will automatically load the latest weights from the `dueling_nstep_checkpoints` directory or fall back to other available weights if no N-step weights are found.

## Configuration

The N-step parameter can be configured in the `config.py` file:

```python
N_STEP = 3  # Number of steps for N-step bootstrapping
```

You can experiment with different values of N to find the optimal setting for your environment. Larger values of N can lead to faster learning but may also introduce more variance in the updates.

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT press.
- Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2018). Rainbow: Combining improvements in deep reinforcement learning. In Thirty-Second AAAI Conference on Artificial Intelligence.
