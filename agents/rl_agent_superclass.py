from abc import ABC, abstractmethod

class RLAgentSuperClass(ABC):
    def __init__(self):
        # Ensure algo_name is defined in subclasses
        if not hasattr(self, "algo_name") or not self.algo_name:
            raise ValueError("Subclasses of RLAgentSuperClass must define 'self.algo_name'.")

    @abstractmethod
    def get_action(self, state, deterministic=False):
        """Sample an action from the policy."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, replay_buffer, batch_size=256):
        """Perform one training step."""
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath):
        """Save the agent's state to a file."""
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath):
        """Load the agent's state from a file."""
        raise NotImplementedError

    @abstractmethod
    def load_from_another_agent(self, another_agent):
        """Load weights from another agent."""
        raise NotImplementedError

    @abstractmethod
    def interpolate_with_other_agent(self, other_agent, alpha):
        """Interpolate weights with another agent."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """Return the state dictionary of the agent."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load the state dictionary into the agent."""
        raise NotImplementedError
