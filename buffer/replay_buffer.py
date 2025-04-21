import collections
import numpy as np
import torch
from abc import ABC, abstractmethod

class ReplayBufferSuperClass(ABC):
    """Abstract base class for replay buffers."""
    @abstractmethod
    def add(self, *args, **kwargs):
        """
        Add a transition to the buffer.

        For basic (eg. non-parallel) ReplayBuffer this will take as input state, action, reward, next_state, done
        For ParallelReplayBuffer this will take same inputs as above, but in the form of a batch of transitions each

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the current size of the replay buffer."""
        raise NotImplementedError
    
    def set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

class ReplayBuffer(ReplayBufferSuperClass):
    '''
    Pre-allocate the memory for efficiency.
    Store as numpy arrays and convert to tensors when sampling for efficiency as well.
    '''
    #TODO: This should be correct implementation, but think more about whether this uncessarily 
    # increases the memory movement between GPU and CPU. 
    # Side-note: Learn more about memory movement on "merged chips" like Apple M1/M2 (it should be lightspeed, no?)

    def __init__(self, state_dim: int, action_dim: int, device : torch.device, capacity: int = 1000000):
        self.capacity = capacity
        self.device = device
        self.ptr = 0        # current index to insert
        self.size = 0       # current number of transitions stored
        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
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
        state_batch = torch.tensor(self.states[batch_indices], dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(self.actions[batch_indices], dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(self.rewards[batch_indices], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(self.next_states[batch_indices], dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(self.dones[batch_indices], dtype=torch.float32).unsqueeze(1).to(self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """Return the current size of the replay buffer."""
        return self.size
    
    def set_seed(self, seed: int):
        #TODO: Remove set_seed from both replay buffer implementations since it's implemented in the abstarct class
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

class ParallelReplayBuffer(ReplayBufferSuperClass):
    """Replay buffer for parallel environments."""
    def __init__(self, state_dim: int, action_dim: int, num_envs: int, device: torch.device, capacity: int = 1000000):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device

        self.per_env_capacity = capacity // num_envs
        print(f"Per env buffer size is: {self.per_env_capacity} given buffer size {capacity} and num_envs {num_envs}")

        self.states = torch.zeros((self.per_env_capacity, num_envs, state_dim), device=device)
        self.next_states = torch.zeros((self.per_env_capacity, num_envs, state_dim), device=device)
        self.actions = torch.zeros((self.per_env_capacity, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((self.per_env_capacity, num_envs), device=device)
        self.dones = torch.zeros((self.per_env_capacity, num_envs), device=device)
        self.ptr = 0
        self.size = 0  # Initialize size to track the number of transitions

    def add(self, states, next_states, actions, rewards, dones):
        """Add transitions for parallel environments."""
        # check that all inputs are of the same shape eg. envs x state_dim
        for arr in [states, next_states, actions, rewards, dones]:
            assert arr.shape[0] == self.num_envs, f"Expected {self.num_envs} environments, but got {arr.shape[0]}."

        assert states.shape[1] == self.states.shape[2], f"Expected state_dim {self.states.shape[2]}, but got {states.shape[1]}."
        assert next_states.shape[1] == self.next_states.shape[2], f"Expected state_dim {self.next_states.shape[2]}, but got {next_states.shape[1]}."
        assert actions.shape[1] == self.actions.shape[2], f"Expected action_dim {self.actions.shape[2]}, but got {actions.shape[1]}."
        assert rewards.shape[1] == self.rewards.shape[1], f"Expected num_envs {self.rewards.shape[1]}, but got {rewards.shape[1]}."
        assert dones.shape[1] == self.dones.shape[1], f"Expected num_envs {self.dones.shape[1]}, but got {dones.shape[1]}."


        idx = self.ptr
        self.states[idx] = states
        self.next_states[idx] = next_states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones

        
        self.ptr = (self.ptr + 1) % self.per_env_capacity
        self.size = min(self.size + self.num_envs, self.per_env_capacity)

    def sample_unflattened(self, batch_size: int):
        """Sample a batch of transitions."""
        batch_indices = np.random.choice(self.size, size=batch_size, replace=False)
        env_indices = torch.randint(0, self.num_envs, (batch_size,))
        return (
            self.states[batch_indices, env_indices],
            self.next_states[batch_indices, env_indices],
            self.actions[batch_indices, env_indices],
            self.rewards[batch_indices, env_indices],
            self.dones[batch_indices, env_indices],
        )
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions and flatten the batch dimension."""
        batch_indices = np.random.choice(self.size, size=batch_size, replace=False)
        env_indices = torch.randint(0, self.num_envs, (batch_size,))

        #TODO: Is this the right way to flatten the batch dimension?
        # We need pairs (state,next_state) to be next to each other in the flattened array
        # I guess that's achieved by default
        return (
            self.states[batch_indices, env_indices].view(-1),
            self.next_states[batch_indices, env_indices].view(-1),
            self.actions[batch_indices, env_indices].view(-1),
            self.rewards[batch_indices, env_indices].view(-1),
            self.dones[batch_indices, env_indices].view(-1),
        )

    def __len__(self):
        """Return the current size of the replay buffer."""
        return self.size
    
    def set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)