import torch.nn as nn
import torch.nn.functional as F
import torch

# Common hidden layer sizes for MuJoCo agents
HIDDEN_DIMS = (256, 256)

class ActorDDPG(nn.Module):
    """Deterministic policy network for DDPG (outputs action mean directly)."""
    def __init__(self, state_dim, action_dim, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        dims = [state_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-1], action_dim))
        layers.append(nn.Tanh())  # output layer, bound actions to [-1, 1]
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.model(state)  # outputs deterministic action

class ActorSAC(nn.Module):
    """Stochastic policy network for SAC (outputs mean and log-std of Gaussian)."""
    def __init__(self, state_dim, action_dim, hidden_dims=HIDDEN_DIMS, log_std_bounds=(-20, 2)):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_bounds
        dims = [state_dim] + list(hidden_dims)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
        self.mean_layer = nn.Linear(dims[-1], action_dim)
        self.log_std_layer = nn.Linear(dims[-1], action_dim)
    
    def forward(self, state):
        x = state
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std to reasonable bounds to stabilize training
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std  # to be used for sampling an action

class Critic(nn.Module):
    """Q-value network: approximates Q(s,a)."""
    def __init__(self, state_dim, action_dim, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        input_dim = state_dim + action_dim
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, state, action):
        # Ensure state and action are concatenated along the last dimension
        if state.dim() == 1:  # if single state (1D tensor), add batch dimension
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        return self.model(torch.cat([state, action], dim=-1))
