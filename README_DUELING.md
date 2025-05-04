# Dueling DDQN Implementation for Super Mario Bros

This directory contains an implementation of a Dueling Double Deep Q-Network (DDQN) with Prioritized Experience Replay (PER) for playing Super Mario Bros.

## Dueling Network Architecture

The Dueling Network architecture separates the estimation of state values and action advantages, which can lead to better policy evaluation in the presence of many similar-valued actions. The key components are:

1. **Value Stream**: Estimates the value function V(s) for a given state.
2. **Advantage Stream**: Estimates the advantage function A(s,a) for each action in a given state.
3. **Aggregation Layer**: Combines the value and advantage streams to produce Q-values using the formula:
   Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

This architecture helps the network learn which states are valuable without having to learn the effect of each action for each state, which can lead to faster convergence and better performance.

## Files

- `dueling_qnet.py`: Implementation of the Dueling Q-Network architecture.
- `dueling_ddqn_per_agent.py`: Implementation of the Dueling DDQN agent with PER.
- `train_dueling_ddqn.py`: Training script for the Dueling DDQN agent.
- `dueling_student_agent.py`: Agent implementation for evaluation using the Dueling Q-Network.

## Dueling Network Implementation Details

The Dueling Network architecture is implemented in `dueling_qnet.py` with the following key components:

1. **Convolutional Layers**: Same as the original network, used for feature extraction from the game frames.
2. **Feature Layer**: A common fully connected layer that processes the flattened convolutional features.
3. **Value Stream**: A sequence of fully connected layers that outputs a single scalar value representing the state value V(s).
4. **Advantage Stream**: A sequence of fully connected layers that outputs a vector of size equal to the number of actions, representing the advantages A(s,a) for each action.
5. **Aggregation**: The value and advantage streams are combined using the formula Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))) to produce the final Q-values.

## Training the Dueling DDQN Agent

To train the Dueling DDQN agent, run:

```bash
python train_dueling_ddqn.py
```

The training script will create checkpoints in the `dueling_checkpoints` directory and logs in the `dueling_logs` directory.

## Using the Dueling DDQN Agent

To use the trained Dueling DDQN agent for evaluation, use the `dueling_student_agent.py` file. This agent will automatically load the latest Dueling Q-Network weights from the `dueling_checkpoints` directory or fall back to the regular Q-Network weights if no Dueling weights are found.

## Differences from Regular DDQN

The main differences between the Dueling DDQN and the regular DDQN are:

1. **Network Architecture**: The Dueling network separates the value and advantage streams, while the regular network directly estimates Q-values.
2. **Parameter Efficiency**: The Dueling network can learn the value of states without having to learn the effect of each action for each state, which can be more parameter-efficient.
3. **Performance**: The Dueling network often performs better in environments with many similar-valued actions, as it can better identify the value of states independent of the actions.

## References

- Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. In International Conference on Machine Learning (pp. 1995-2003).
