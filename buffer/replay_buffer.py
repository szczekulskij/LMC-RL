import collections
import numpy as np
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps") 
elif torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu") 

class ReplayBuffer:
    '''
    Pre-allocate the memory for efficiency.
    Store as numpy arrays and convert to tensors when sampling for efficiency as well.
    '''
    #TODO: This should be correct implementation, but think more about whether this uncessarily 
    # increases the memory movement between GPU and CPU. 
    # Side-note: Learn more about memory movement on "merged chips" like Apple M1/M2 (it should be lightspeed, no?)

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1000000):
        self.capacity = capacity
        self.ptr = 0        # current index to insert
        self.size = 0       # current number of transitions stored
        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(self, state, action, reward, next_state, done):
        #TODO: Double check we don't silently store tensors here rather than numpy
        """Add a transition to the buffer."""
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0  # store done as float (1.0 or 0.0)
        # Update pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions."""
        #TODO: Check if we need to add .to(device) in code below
        assert self.size > 0, "Buffer is empty!"
        batch_indices = np.random.choice(self.size, size=batch_size, replace=False)
        # Convert to tensors for training
        state_batch = torch.tensor(self.states[batch_indices], dtype=torch.float32).to(device)
        action_batch = torch.tensor(self.actions[batch_indices], dtype=torch.float32).to(device)
        reward_batch = torch.tensor(self.rewards[batch_indices], dtype=torch.float32).unsqueeze(1).to(device)
        next_state_batch = torch.tensor(self.next_states[batch_indices], dtype=torch.float32).to(device)
        done_batch = torch.tensor(self.dones[batch_indices], dtype=torch.float32).unsqueeze(1).to(device)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """Return the current size of the replay buffer."""
        return self.size
    
    def set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)