from agents.networks import ActorDDPG, Critic
import torch
import torch.nn.functional as F  # Add this import

class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, device='cpu'):
        self.gamma = gamma
        self.tau = tau
        self.device = device  # Store the device

        # Networks
        self.actor = ActorDDPG(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        # Target networks (start as clones of the originals)
        self.target_actor = ActorDDPG(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # Exploration noise
        self.noise_std = 0.1  # standard deviation of Gaussian noise for exploration
    
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