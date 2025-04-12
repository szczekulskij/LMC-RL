import yaml  # Add this import
from agents.networks import ActorSAC, Critic
import torch
import torch.nn.functional as F

class SACAgent:
    def __init__(self, state_dim, action_dim, config_path="configs/default_sac.yaml", device='cpu'):
        # Load hyperparameters from YAML
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Validate and convert types
        self.gamma = float(config["gamma"])
        self.tau = float(config["tau"])
        self.alpha = float(config["alpha"])
        self.device = device  # Store the device

        # Networks
        hidden_dims = [int(dim) for dim in config["hidden_dims"]]  # Ensure hidden_dims are integers
        self.actor = ActorSAC(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        # Target critics
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config["actor_lr"]))
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=float(config["critic_lr"]))
        # (If automating alpha, would create alpha param and optimizer here)

        self.algo_name = "SAC"
    
    def get_action(self, state, deterministic=False):
        """Sample an action from the policy. If deterministic=True, return the mean action (for evaluation)."""
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        mean, log_std = self.actor(state_t)
        if deterministic:
            # Simply take mean and apply tanh (no randomness) for evaluation
            action = torch.tanh(mean)
        else:
            # Sample from Gaussian and apply tanh transform
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            u = dist.rsample()              # sample in Gaussian space (rsample for reparameterization)
            action = torch.tanh(u)          # apply squashing
        return action.detach().cpu().numpy()
    
    def train_step(self, replay_buffer, batch_size=256):
        #TODO: De-couple batch_size to config file (do the same for DDPG)
        # Batch size of 256 as per SpinningUp
        """One SAC training step on a batch."""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device), actions.to(self.device), rewards.to(self.device),
            next_states.to(self.device), dones.to(self.device)
        )
        # Sample action from current policy for next_states, for computing target Q
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            dist = torch.distributions.Normal(next_mean, next_std)
            u2 = dist.rsample()
            next_actions = torch.tanh(u2)
            # Compute log-probability of next_actions (for entropy term)
            log_prob_u2 = dist.log_prob(u2)  # log prob in Gaussian space
            # Correction for Tanh squashing: log probability in action space
            log_prob_next = log_prob_u2 - torch.log(torch.clamp(1 - next_actions**2, min=1e-6))
            log_prob_next = log_prob_next.sum(dim=1, keepdim=True)
            # Target Q: use target critics and include entropy term
            target_Q1 = self.target_critic1(next_states, next_actions)
            target_Q2 = self.target_critic2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_value = target_Q - self.alpha * log_prob_next
            target = rewards + self.gamma * (1 - dones) * target_value
        # Current Q estimates
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        # Critic losses (MSE)
        critic_loss = F.mse_loss(current_Q1, target) + F.mse_loss(current_Q2, target)
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor loss: maximize Q + entropy -> minimize (alpha * log_prob - Q)
        # Re-sample action for states for actor update
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        actions_pi = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(torch.clamp(1 - actions_pi**2, min=1e-6))
        log_prob = log_prob.sum(dim=1, keepdim=True)
        # Q values for new actions
        Q1_pi = self.critic1(states, actions_pi)
        Q2_pi = self.critic2(states, actions_pi)
        Q_pi = torch.min(Q1_pi, Q2_pi)
        actor_loss = (self.alpha * log_prob - Q_pi).mean()
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # (If alpha were learnable, update alpha here using target entropy loss)
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            # 'alpha': self.alpha  (if alpha is learned, include its state)
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
        # If alpha were learnable, load it and its optimizer state as well

