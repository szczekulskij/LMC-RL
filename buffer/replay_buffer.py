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
    Store everything as PyTorch tensors directly on the specified device.
    '''
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, capacity: int = 1000000):
        self.capacity = capacity
        self.device = device
        self.ptr = 0        # current index to insert
        self.size = 0       # current number of transitions stored
        # Pre-allocate memory for efficiency
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        idx = self.ptr
        self.states[idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[idx] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[idx] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.tensor(1.0 if done else 0.0, dtype=torch.float32, device=self.device)
        # Update pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions."""
        assert self.size > 0, "Buffer is empty!"
        batch_indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices].unsqueeze(1),
            self.next_states[batch_indices],
            self.dones[batch_indices].unsqueeze(1),
        )

    def __len__(self):
        """Return the current size of the replay buffer."""
        return self.size

class ParallelReplayBuffer(ReplayBufferSuperClass):
    """Replay buffer for parallel environments."""
    def __init__(self, state_dim: int, action_dim: int, num_envs: int, device: torch.device, capacity: int = 1000000):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.per_env_capacity = capacity // num_envs
        print(f"Per env buffer size is: {self.per_env_capacity} given buffer size {capacity} and num_envs {num_envs}")

        self.states = torch.zeros((self.per_env_capacity, num_envs, state_dim), device=device)
        self.next_states = torch.zeros((self.per_env_capacity, num_envs, state_dim), device=device)
        self.actions = torch.zeros((self.per_env_capacity, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((self.per_env_capacity, num_envs), device=device)
        self.dones = torch.zeros((self.per_env_capacity, num_envs), device=device)
        self.ptr = 0
        self.size = 0  # Initialize size to track the number of transitions

    def add(self, states, actions, rewards, next_states, dones):
        """Add transitions for parallel environments."""
        # Convert inputs to PyTorch tensors on the correct device
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Check that all inputs are of the correct shape
        assert states.shape[1] == self.state_dim, f"For states input, expected state_dim {self.state_dim}, but got {states.shape[1]}."
        assert states.shape[0] == self.num_envs, f"For states input, expected num_envs {self.num_envs}, but got {states.shape[0]}."
        assert next_states.shape[1] == self.state_dim, f"For next_states input, expected state_dim {self.state_dim}, but got {next_states.shape[1]}."
        assert next_states.shape[0] == self.num_envs, f"For next_states input, expected num_envs {self.num_envs}, but got {next_states.shape[0]}."
        assert actions.shape[1] == self.action_dim, f"For actions input, expected action_dim {self.action_dim}, but got {actions.shape[1]}."
        assert actions.shape[0] == self.num_envs, f"For actions input, expected num_envs {self.num_envs}, but got {actions.shape[0]}."
        assert rewards.shape[0] == self.num_envs, f"For rewards input, expected num_envs {self.num_envs}, but got {rewards.shape[0]}."
        assert dones.shape[0] == self.num_envs, f"For dones input, expected num_envs {self.num_envs}, but got {dones.shape[0]}."

        # Add transitions to the buffer
        idx = self.ptr
        self.states[idx] = states
        self.next_states[idx] = next_states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones

        # Update pointers
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