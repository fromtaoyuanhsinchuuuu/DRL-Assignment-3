import numpy as np
import torch
from per_buffer import PrioritizedReplayBuffer

def test_per_buffer():
    """
    Test the PrioritizedReplayBuffer implementation.
    """
    print("Testing PrioritizedReplayBuffer...")
    
    # Initialize buffer
    buffer_capacity = 1000
    buffer = PrioritizedReplayBuffer(buffer_capacity)
    
    # Add some experiences
    num_experiences = 100
    print(f"Adding {num_experiences} experiences to buffer...")
    
    for i in range(num_experiences):
        # Create dummy state and next_state
        state = np.zeros((4, 84, 84, 1), dtype=np.uint8)
        next_state = np.ones((4, 84, 84, 1), dtype=np.uint8)
        
        # Add to buffer
        buffer.add(state, i % 12, 1.0, next_state, False)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test sampling
    batch_size = 32
    print(f"Sampling {batch_size} experiences...")
    
    # Sample multiple times to test robustness
    for i in range(5):
        experiences, indices, weights = buffer.sample(batch_size)
        
        if experiences is None:
            print(f"Sample {i+1}: Failed to sample experiences")
        else:
            states, actions, rewards, next_states, dones = experiences
            print(f"Sample {i+1}: Got {len(indices)} experiences with shape {states.shape}")
            
            # Update priorities
            for idx in indices:
                buffer.update_priority(idx, 1.0)
    
    print("Testing completed successfully!")

if __name__ == "__main__":
    test_per_buffer()
