import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from diffusion import Diffusion
from tianshou.data import Batch, ReplayBuffer, to_torch
from env import make_aigc_env  
from diffusion.model import MLP
from tianshou.data import Batch, ReplayBuffer, to_torch

class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, device, n_timesteps=6, beta_schedule='vp', max_action=1.0):
        super().__init__()
        self.device = device
        self.diffusion = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=MLP(state_dim,action_dim).to(device),            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps
        )

    def forward(self, state):
        return self.diffusion.sample(state.to(self.device))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, state):
        return self.critic(state.to(self.device))

class PPO_Diffusion:
    def __init__(self, state_dim, action_dim, device='cuda:0', lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.actor = DiffusionPolicy(state_dim, action_dim, device).to(device)
        self.critic = Critic(state_dim, device=device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.scaler = torch.cuda.amp.GradScaler()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def compute_advantage(self, rewards, values, dones):
        advantages = []
        advantage = 0
        rewards, values, dones = map(torch.tensor, (rewards, values, dones))
        rewards, values, dones = rewards.to(self.device), values.to(self.device), dones.to(self.device)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * 0.95 * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)

        return torch.tensor(advantages, dtype=torch.float).to(self.device)

    def update(self, memory):
        states, actions, rewards, dones, old_values = memory
        states, actions, old_values = map(lambda x: torch.FloatTensor(x).to(self.device), (states, actions, old_values))

        values = self.critic(states).squeeze(-1)

        advantages = self.compute_advantage(rewards, old_values, dones)

        with torch.cuda.amp.autocast():
            new_actions = self.actor(states)
            ratio = (new_actions / actions).clamp(0.8, 1.2)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = nn.MSELoss()(values, advantages + old_values)

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)

        self.scaler.update()


def train_ppo():
    env, train_envs, test_envs = make_aigc_env(1, 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPO_Diffusion(state_dim, action_dim, device)

    num_episodes = 1000
    memory = []

    for episode in range(num_episodes):
        state = torch.FloatTensor(env.reset()).to(device)
        done = False
        episode_rewards = []
        episode_values = []

        while not done:
            action = agent.select_action(state.cpu().numpy())
            next_state, reward, done, _ = env.step(action)

            episode_rewards.append(reward)
            episode_values.append(agent.critic(state).item())

            state = torch.FloatTensor(next_state).to(device)

        memory.append((state.cpu().numpy(), action, episode_rewards, done, episode_values))

        agent.update(memory)
        memory.clear()
        print('Episode ', episode+1, 'Reward: ', sum(episode_rewards))
        # print(f'Episode {episode+1}, Reward: ', sum(episode_rewards))
train_ppo()
