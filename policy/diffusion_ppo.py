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
        
        self._actor_old = deepcopy(self._actor)

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
        self.noise_generator = GaussianNoise(sigma=exploration_noise)

    def forward(self, batch: Batch, input: str = "obs", model: str = "actor", **kwargs) -> Batch:
        """
        Từ các quan sát (obs) trong batch, tạo hành động thông qua quá trình diffusion
        và tính log-prob tương ứng.
        """
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        # Tạo nhiễu khởi tạo cho diffusion (điều chỉnh theo exploration_noise)
        # Use actor or target actor based on provided model argument
        if model == "actor":
            model_ = self._actor 
        elif model == 'actor_old':
            model_ = self._actor_old
        else:
            self._target_actor
        # Feed observations through the selected model to get action logits
        logp, hidden = model_(obs_), None

        if self._bc_coef:
            acts = logp
        else:
            if np.random.rand() < 0.1:
                # Add exploration noise to the actions
                noise = to_torch(self.noise_generator.generate(logp.shape),
                                 dtype=torch.float32, device=self._device)
                # Add the noise to the action
                acts = logp + noise
                acts = torch.clamp(acts, -1, 1)
            else:
                acts = logp

        dist = None  # does not use a probability distribution for actions

        return Batch(logp=logp, act=acts, state=obs_, dist=dist)
    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        # Convert the provided data to one-hot representation
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        # print(data[1])
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)
    
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
    def update(self, sample_size: int, buffer, **kwargs) -> dict:
        if buffer is None:
            return {}
        self.updating = True
        # Lấy một batch từ buffer theo số lượng sample_size
        batch, indices = buffer.sample(sample_size)
        # Nếu có hàm process_fn, sử dụng nó để xử lý batch
        if hasattr(self, "process_fn"):
            batch = self.process_fn(batch, buffer, indices)
        result = self._learn_update(batch, **kwargs)
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False
        return result
    def _learn_update(self, batch: Batch, **kwargs) -> dict:
        # Chuyển đổi dữ liệu batch về tensor trên thiết bị
        # print(batch)

        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        actions = to_torch(batch.act, device=self._device, dtype=torch.float32)
        rewards = to_torch(batch.rew, device=self._device, dtype=torch.float32)

        # Tính giá trị hiện tại từ critic
        values, values_ = self._critic(obs, actions)
        
        values = values.squeeze(-1)
        # Append thêm một giá trị 0 làm bootstrapping
        values = torch.cat([values, torch.zeros(1, device=values.device)], dim=0) #torch.Size([512, 11])

        advantages = self.compute_advantage(rewards, values) 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = rewards
        # Tính critic loss (MSE giữa giá trị dự đoán và returns)
        returns = advantages + values[:-1]
        critic_loss = F.mse_loss(values[:-1], returns.detach())

        # Tính log-prob mới từ policy hiện tại
        new_batch = Batch(obs=obs)
        current = self.forward(new_batch)
        # Lấy new_logp từ actor chính, clone sau khi slicing
        new_logp = current.logp[:, -1].clone()
        # Lấy old_logp từ actor_old, nhưng tách gradient trước rồi clone
        old_logp = self.forward(new_batch, model='actor_old').logp[:, -1].detach().clone()

        # Tính tỷ số giữa log-prob mới và cũ (PPO ratio)
        ratio = torch.exp(new_logp - old_logp)
        # print('ratio:', ratio.shape)
        # print('advantages:', advantages.shape)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        self._actor_old = deepcopy(self._actor)
        
        self._actor_optim.zero_grad()
        self._critic_optim.zero_grad()
        
        # critic_loss.backward()
        # actor_loss.backward()
        
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        self._actor_optim.step()
        self._critic_optim.step()




        # Cập nhật mềm (soft update) cho target networks
        self._soft_update(self._target_actor, self._actor, self._tau)
        self._soft_update(self._target_critic, self._critic, self._tau)

        return {"loss/actor": actor_loss.item(), "loss/critic": critic_loss.item()}


    # def update(self, batch: Batch, **kwargs) -> dict:
    #     """
    #     Cập nhật policy theo tiêu chí PPO với mục tiêu clipping.
    #     """
    #     # Chuyển dữ liệu batch về tensor trên thiết bị
    #     obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
    #     actions = to_torch(batch.act, device=self._device, dtype=torch.float32)
    #     old_logp = to_torch(batch.logp, device=self._device, dtype=torch.float32)
    #     rewards = to_torch(batch.rew, device=self._device, dtype=torch.float32)
        
    #     # Tính giá trị hiện tại từ critic
    #     values = self._critic(obs).squeeze(-1)
    #     # Append thêm giá trị 0 làm bootstrapping (có thể thay bằng V(s_T) nếu có)
    #     values = torch.cat([values, torch.zeros(1, device=values.device)], dim=0)
        
    #     advantages = self.compute_advantage(rewards, values)
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    #     # Tính log-prob mới từ policy hiện tại
    #     new_batch = Batch(obs=obs)
    #     current = self.forward(new_batch)
    #     new_logp = current.logp
        
    #     # Tỷ số giữa log-prob mới và cũ (PPO ratio)
    #     ratio = torch.exp(new_logp - old_logp)
    #     surr1 = ratio * advantages
    #     surr2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantages
    #     actor_loss = -torch.min(surr1, surr2).mean()
        
    #     # Tính critic loss (MSE giữa giá trị dự đoán và returns)
    #     returns = advantages + values[:-1]
    #     critic_loss = F.mse_loss(values[:-1], returns.detach())
        
    #     # Cập nhật actor
    #     self._actor_optim.zero_grad()
    #     actor_loss.backward()
    #     self._actor_optim.step()
        
    #     # Cập nhật critic
    #     self._critic_optim.zero_grad()
    #     critic_loss.backward()
    #     self._critic_optim.step()
        
    #     if self._lr_decay:
    #         self._actor_lr_scheduler.step()
    #         self._critic_lr_scheduler.step()
        
    #     # Cập nhật mềm (soft update) cho target networks
    #     self._soft_update(self._target_actor, self._actor, self._tau)
    #     self._soft_update(self._target_critic, self._critic, self._tau)
        
    #     return {"loss/actor": actor_loss.item(), "loss/critic": critic_loss.item()}
    def learn(self, batch: Batch, **kwargs) -> dict:
        # Chỉ gọi _learn_update với batch đã có
        return self._learn_update(batch, **kwargs)

    # def learn(self, batch: Batch, **kwargs) -> dict:
    #     """
    #     Override abstract method learn() của BasePolicy.
    #     Tại đây chỉ đơn giản gọi đến update.
    #     """
    #     return self.update(batch, **kwargs)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class GaussianNoise:
    """Generates Gaussian noise."""

    def __init__(self, mu=0.0, sigma=0.1):
        """
        :param mu: Mean of the Gaussian distribution.
        :param sigma: Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        """
        Generate Gaussian noise based on a shape.

        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise