import yaml  # Add this import
from agents.networks import ActorDDPG, Critic
import torch
import torch.nn.functional as F

class DDPGAgent:
    #TODO: I forgot to rescale the actions to the action space, (eg. DDPG returns [-1,1], so that might need to be possibly rescaled)
    def __init__(self, state_dim, action_dim, config_path="configs/default_ddpg.yaml", device='cpu'):
        # Load hyperparameters from YAML
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Validate and convert types
        self.gamma = float(config["gamma"])
        self.tau = float(config["tau"])
        self.device = device  # Store the device

        # Networks
        hidden_dims = [int(dim) for dim in config["hidden_dims"]]  # Ensure hidden_dims are integers
        self.actor = ActorDDPG(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        # Target networks (start as clones of the originals)
        self.target_actor = ActorDDPG(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config["actor_lr"]))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config["critic_lr"]))
        # Exploration noise
        self.noise_std = float(config["noise_std"])
    
    def get_action(self, state, noise=True, deterministic=True):
        """Select an action for a given state, with optional exploration noise or deterministic mode."""
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        a = self.actor(state_t)
        if noise:
            # Add Gaussian noise for exploration, and clip to [-1, 1]
            a = a + torch.normal(mean=0.0, std=self.noise_std, size=a.shape).to(self.device)
            a = torch.clamp(a, -1.0, 1.0)
        return a.detach().cpu().numpy()
    
    def train_step(self, replay_buffer, batch_size=256):
        """Perform one training update (on one batch of data)."""
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device), actions.to(self.device), rewards.to(self.device),
            next_states.to(self.device), dones.to(self.device)
        )
        # Compute target Q value: r + gamma * Q_target(s', \pi_target(s')) * (1 - done)
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_Q = self.target_critic(next_states, target_actions)
            target_Q = rewards + self.gamma * (1 - dones) * target_Q
        # Critic loss: MSE between current Q and target Q
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor loss: -Q(s, \pi(s)) (we want to maximize Q, so minimize negative Q)
        actor_loss = - self.critic(states, self.actor(states)).mean()
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save the agent (network weights and optimizers, and optionally other state)."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load the agent state from file (networks and optimizers)."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])

    def load_from_another_agent(self, other_agent):
        """
        Load weights from another DDPG agent.
        
        Args:
            other_agent (DDPGAgent): Another DDPG agent to load weights from.
        """
        self.actor.load_state_dict(other_agent.actor.state_dict())
        self.critic.load_state_dict(other_agent.critic.state_dict())
        self.target_actor.load_state_dict(other_agent.target_actor.state_dict())
        self.target_critic.load_state_dict(other_agent.target_critic.state_dict())
        self.actor_optimizer.load_state_dict(other_agent.actor_optimizer.state_dict())
        self.critic_optimizer.load_state_dict(other_agent.critic_optimizer.state_dict())

    def interpolate_with_other_agent(self, other_agent_weights, alpha):
        """
        Interpolate the weights of this agent with another agent's weights.
        
        Args:
            other_agent_weights (dict): State dictionary of the other agent.
            alpha (float): Interpolation factor (0.0 = this agent, 1.0 = other agent).
        """
        # Interpolate actor weights
        for param, other_param in zip(self.actor.parameters(), other_agent_weights['actor'].values()):
            param.data.copy_((1 - alpha) * param.data + alpha * other_param.data)

        # Interpolate critic weights
        for param, other_param in zip(self.critic.parameters(), other_agent_weights['critic'].values()):
            param.data.copy_((1 - alpha) * param.data + alpha * other_param.data)

        # Interpolate target actor weights
        for param, other_param in zip(self.target_actor.parameters(), other_agent_weights['target_actor'].values()):
            param.data.copy_((1 - alpha) * param.data + alpha * other_param.data)

        # Interpolate target critic weights
        for param, other_param in zip(self.target_critic.parameters(), other_agent_weights['target_critic'].values()):
            param.data.copy_((1 - alpha) * param.data + alpha * other_param.data)