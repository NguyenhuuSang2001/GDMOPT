import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from .helpers import (
    Losses
)
class DiffusionPPO(BasePolicy):
    """
    Một thuật toán kết hợp PPO với mô hình diffusion.
    Sử dụng actor diffusion để sinh hành động và cập nhật theo tiêu chí PPO.
    """
    def __init__(
        self,
        state_dim: int,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
        action_dim: int,
        critic: nn.Module,
        critic_optim: torch.optim.Optimizer,
        device: torch.device,
        tau: float = 0.005,
        gamma: float = 0.99,
        estimation_step: int = 1,
        lr_decay: bool = False,
        lr_maxt: int = 1000,
        bc_coef: bool = False,
        action_space=None,
        exploration_noise: float = 0.1,
        clip_ratio: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._actor = actor
        self._target_actor = deepcopy(actor)
        self._target_actor.eval()
        self._actor_optim = actor_optim

        self._critic = critic
        self._target_critic = deepcopy(critic)
        self._target_critic.eval()
        self._critic_optim = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)
        self._lr_decay = lr_decay

        self._tau = tau
        self._gamma = gamma
        self._n_step = estimation_step
        self._bc_coef = bc_coef
        self._device = device
        self._clip_ratio = clip_ratio
        self._action_space = action_space
        self._exploration_noise = exploration_noise

    def forward(self, batch: Batch, input: str = "obs", **kwargs) -> Batch:
        """
        Từ các quan sát (obs) trong batch, tạo hành động thông qua quá trình diffusion
        và tính log-prob tương ứng.
        """
        obs = to_torch(batch[input], device=self._device, dtype=torch.float32)
        # Tạo nhiễu khởi tạo cho diffusion (điều chỉnh theo exploration_noise)
        init_noise = torch.randn((obs.shape[0], self._actor.action_dim), device=self._device) * self._exploration_noise
        actions = self._actor.diffuse(obs, init_noise)
        logp = self._actor.log_prob(obs, actions)
        return Batch(act=actions, logp=logp)

    def compute_advantage(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float = None, lam: float = 0.95) -> torch.Tensor:
        """
        Tính toán Generalized Advantage Estimate (GAE).
        Giả sử rewards có độ dài T và values có độ dài T+1 (với giá trị bootstrapping ở cuối).
        """
        if gamma is None:
            gamma = self._gamma
        adv = torch.zeros_like(rewards)
        lastgaelam = 0
        T = rewards.shape[0]
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        return adv

    def update(self, batch: Batch, **kwargs) -> dict:
        """
        Cập nhật policy theo tiêu chí PPO với mục tiêu clipping.
        """
        # Chuyển dữ liệu batch về tensor trên thiết bị
        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        actions = to_torch(batch.act, device=self._device, dtype=torch.float32)
        old_logp = to_torch(batch.logp, device=self._device, dtype=torch.float32)
        rewards = to_torch(batch.rew, device=self._device, dtype=torch.float32)
        
        # Tính giá trị hiện tại từ critic
        values = self._critic(obs).squeeze(-1)
        # Append thêm giá trị 0 làm bootstrapping (có thể thay bằng V(s_T) nếu có)
        values = torch.cat([values, torch.zeros(1, device=values.device)], dim=0)
        
        advantages = self.compute_advantage(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Tính log-prob mới từ policy hiện tại
        new_batch = Batch(obs=obs)
        current = self.forward(new_batch)
        new_logp = current.logp
        
        # Tỷ số giữa log-prob mới và cũ (PPO ratio)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Tính critic loss (MSE giữa giá trị dự đoán và returns)
        returns = advantages + values[:-1]
        critic_loss = F.mse_loss(values[:-1], returns.detach())
        
        # Cập nhật actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()
        
        # Cập nhật critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        
        # Cập nhật mềm (soft update) cho target networks
        self._soft_update(self._target_actor, self._actor, self._tau)
        self._soft_update(self._target_critic, self._critic, self._tau)
        
        return {"loss/actor": actor_loss.item(), "loss/critic": critic_loss.item()}
    
    def learn(self, batch: Batch, **kwargs) -> dict:
        """
        Override abstract method learn() của BasePolicy.
        Tại đây chỉ đơn giản gọi đến update.
        """
        return self.update(batch, **kwargs)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
