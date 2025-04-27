# per_buffer.py
import random
import numpy as np
import torch
from collections import namedtuple

class SumTree:
    """
    A binary sum tree data structure for efficient sampling based on priorities.
    Used for Prioritized Experience Replay (PER).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree array to store priorities
        self.write_pos = 0  # Current position to write in the tree
        self.size = 0  # Current size of the tree

    def _propagate(self, idx, change):
        """
        Update parent nodes after changing a leaf node.

        Args:
            idx: Index of the changed node
            change: Change in value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_idx, p):
        """
        Update the priority of a leaf node.

        Args:
            tree_idx: Index of the leaf node in the tree
            p: New priority value
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def add(self, p):
        """
        Add a new experience with priority p.

        Args:
            p: Priority value

        Returns:
            Index where the data was stored
        """
        tree_idx = self.write_pos + self.capacity - 1
        self.tree[tree_idx] = p
        data_idx = self.write_pos

        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        self._propagate(tree_idx, p)
        return data_idx

    def get_leaf(self, s):
        """
        Find the leaf node based on a value s.

        Args:
            s: Value to search for (0 <= s <= total_priority)

        Returns:
            (tree_idx, priority, data_idx)
        """
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we reach a leaf node
            if left_child_idx >= len(self.tree):
                tree_idx = parent_idx
                break

            # Otherwise, go left or right based on the value s
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = tree_idx - self.capacity + 1

        # Ensure data_idx is within valid range
        if data_idx < 0:
            data_idx = 0
        elif data_idx >= self.capacity:
            data_idx = self.capacity - 1

        return tree_idx, self.tree[tree_idx], data_idx

    def total_priority(self):
        """
        Get the total priority (sum of all priorities).

        Returns:
            Total priority
        """
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for storing and sampling experiences.
    """
    # PER hyperparameters
    epsilon = 0.01  # Small constant to ensure non-zero priority
    alpha = 0.6     # How much prioritization to use (0 = no prioritization, 1 = full prioritization)
    beta = 0.4      # Importance sampling correction factor (start value)
    beta_increment_per_sampling = 0.001  # Beta annealing
    abs_err_upper = 1.0  # Clipping for priorities

    def __init__(self, capacity, seed=None):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tree = SumTree(capacity)
        self.memory = np.empty(capacity, dtype=object)
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.max_priority = 1.0
        self.size = 0
        self.capacity = capacity

    def _get_priority(self, error):
        """
        Convert TD error to priority.

        Args:
            error: TD error

        Returns:
            Priority value
        """
        # Clip error and convert to priority
        error = min(error, self.abs_err_upper)
        return (error + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Args:
            state: Current state (numpy array)
            action: Action taken
            reward: Reward received
            next_state: Next state (numpy array)
            done: Whether the episode is done
        """
        # Check if any of the inputs are None
        if state is None or next_state is None:
            print("Warning: Attempted to add None state or next_state to buffer. Skipping.")
            return

        try:
            # Create experience tuple
            experience = self.Experience(
                state=np.array(state, dtype=np.uint8),
                action=action,
                reward=reward,
                next_state=np.array(next_state, dtype=np.uint8),
                done=done
            )

            # Add to memory with maximum priority
            idx = self.tree.add(self.max_priority)
            self.memory[idx] = experience
            self.size = min(self.size + 1, self.capacity)
        except Exception as e:
            print(f"Error adding experience to buffer: {e}")

    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            (experiences_tensors, indices, importance_sampling_weights)
            experiences_tensors: tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors
            indices: list of indices in the buffer for updating priorities
            importance_sampling_weights: PyTorch tensor of importance sampling weights

            If not enough valid experiences are found, returns None instead of experiences_tensors
        """
        # Check if we have enough experiences to sample
        if len(self) < batch_size:
            print(f"Warning: Not enough experiences in buffer. Have {len(self)}, need {batch_size}.")
            return None, None, None

        indices = []
        priorities = []

        # Calculate segment size
        segment = self.tree.total_priority() / batch_size

        # Update beta (annealing)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            _, priority, data_idx = self.tree.get_leaf(s)  # We don't need tree_idx here

            # Only add indices that are within the valid range (< self.size)
            if data_idx < self.size:
                indices.append(data_idx)
                priorities.append(priority)
            else:
                # If we get an invalid index, try again with a different random value
                # We'll try up to 5 times to find a valid index within this segment
                valid_idx_found = False
                for _ in range(5):  # Try 5 times
                    s = random.uniform(a, b)
                    _, priority, data_idx = self.tree.get_leaf(s)
                    if data_idx < self.size:
                        indices.append(data_idx)
                        priorities.append(priority)
                        valid_idx_found = True
                        break

                # If we still couldn't find a valid index, print a warning
                if not valid_idx_found:
                    print(f"Warning: Could not find valid index in segment {i}. Tree size: {self.tree.size}, Buffer size: {self.size}")

        # If we don't have enough indices after filtering, return None
        if len(indices) < batch_size // 2:  # If we have less than half the requested batch size
            print(f"Warning: Not enough valid indices after filtering. Have {len(indices)}, need {batch_size}.")
            if len(indices) == 0:
                print("Error: No valid indices found.")
                return None, None, None
            # We'll continue with the indices we have, but the batch size will be smaller

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)

        # Normalize weights
        max_weight = np.max(is_weights) if len(is_weights) > 0 else 1.0
        is_weights = is_weights / max_weight

        # Get experiences from memory
        experiences = [self.memory[idx] for idx in indices]

        # Filter out None values and invalid experiences
        valid_experiences = []
        valid_indices = []
        valid_weights = []

        for i, exp in enumerate(experiences):
            if exp is not None and hasattr(exp, 'state') and hasattr(exp, 'next_state'):
                valid_experiences.append(exp)
                valid_indices.append(indices[i])
                valid_weights.append(is_weights[i])

        # If we don't have enough valid experiences, return None
        if len(valid_experiences) < batch_size // 2:  # If we have less than half the requested batch size
            print(f"Warning: Only found {len(valid_experiences)} valid experiences out of {len(indices)} indices after filtering.")
            if len(valid_experiences) == 0:
                print("Error: No valid experiences found.")
                return None, None, None

        # Convert to tensors
        # Stack states and next_states with shape (batch_size, frames, height, width, channels)
        # We need to be careful with the stacking to maintain the batch dimension
        try:
            states_list = [torch.from_numpy(e.state).float() for e in valid_experiences]
            next_states_list = [torch.from_numpy(e.next_state).float() for e in valid_experiences]

            # Stack the tensors along the batch dimension
            states = torch.stack(states_list)
            next_states = torch.stack(next_states_list)

            # Remove the channel dimension (which is 1) to get shape (batch_size, frames, height, width)
            states = states.squeeze(-1)
            next_states = next_states.squeeze(-1)

            # Normalize
            states = states / 255.0
            next_states = next_states / 255.0

            # Convert other experience components to tensors
            actions = torch.from_numpy(np.vstack([e.action for e in valid_experiences])).long()
            rewards = torch.from_numpy(np.vstack([e.reward for e in valid_experiences])).float()
            dones = torch.from_numpy(np.vstack([e.done for e in valid_experiences]).astype(np.uint8)).float()

            # Convert valid_weights to PyTorch tensor directly
            valid_weights = torch.FloatTensor(valid_weights).unsqueeze(1)  # Shape: (batch_size, 1)

            # Print a warning if we're returning fewer experiences than requested
            actual_batch_size = len(valid_experiences)
            if actual_batch_size < batch_size:
                print(f"Note: Returning {actual_batch_size} experiences instead of the requested {batch_size}.")

            return (states, actions, rewards, next_states, dones), valid_indices, valid_weights
        except Exception as e:
            print(f"Error converting experiences to tensors: {e}")
            return None, None, None

    def update_priority(self, idx, error):
        """
        Update priority for a sampled experience.

        Args:
            idx: Index of the sampled experience
            error: New TD error for this experience
        """
        # Check if idx is valid
        if idx < 0 or idx >= self.capacity:
            print(f"Warning: Invalid index {idx} for update_priority. Skipping.")
            return

        priority = self._get_priority(error)
        tree_idx = idx + self.tree.capacity - 1
        self.tree.update(tree_idx, priority)
        self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            Current size
        """
        return self.size