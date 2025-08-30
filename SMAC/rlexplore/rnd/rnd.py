#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rnd.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 21:46 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder, Encoder, RNDModel
from torch import optim
from torch.nn import functional as F
import torch
import numpy as np


class RND(object):
    def __init__(self,
                 obs_shape,
                 device,
                 beta: float = 0.05,
                 kappa: float = 2.5e-05,
                 latent_dim: int = 128,
                 lr: float = 0.001,
                 batch_size: int = 64,
                 epsilon=1e-4,
                 ):
        """
        Exploration by Random Network Distillation (RND)
        Paper: https://arxiv.org/pdf/1810.12894.pdf

        :param obs_shape: The data shape of observations.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param lr: The learning rate of predictor network.
        :param batch_size: The batch size to train the predictor network.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = obs_shape
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa
        self.epsilon = epsilon

        self.predictor = CnnEncoder(obs_shape[0], obs_shape[1], latent_dim)
        self.target = CnnEncoder(obs_shape[0], obs_shape[1], latent_dim)
        # self.predictor = RNDModel(obs_shape[0], latent_dim)
        # self.target = RNDModel(obs_shape[0], latent_dim)
        # self.predictor = Encoder(
        #     obs_shape=self.obs_shape,
        #     latent_dim=latent_dim,
        # )
        #
        # self.target = Encoder(
        #     obs_shape=self.obs_shape,
        #     latent_dim=latent_dim,
        # )

        self.predictor.to(self.device)
        self.target.to(self.device)

        self.opt = optim.Adam(lr=self.lr, params=self.predictor.parameters())

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False

        self.running_mean_std = RunningMeanStd(epsilon=self.epsilon, shape=obs_shape)

    def compute_irs(self, rollouts, time_steps):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_agent = rollouts['obs'].shape[0]
        # n_envs = rollouts['obs'].shape[1]

        # observations shape ((n_steps, n_envs) + obs_shape)
        next_obs_tensor = rollouts['obs'].to(self.device)

        with torch.no_grad():
            src_feats = self.predictor(next_obs_tensor)

            tgt_feats = self.target(next_obs_tensor)

            # Calculate the mean distance for all agents
            dist = F.mse_loss(src_feats, tgt_feats, reduction='none').mean(dim=1)
            dist_scalar = dist.mean().item()  # 计算张量的平均值并转换为标量
            # intrinsic_reward = (dist_scalar - dist.min().item()) / (dist.max().item() - dist.min().item() + 1e-11)

            # 更新RunningMeanStd对象的状态
            dist_numpy = dist.cpu().numpy()
            self.running_mean_std.update(dist_numpy)

            # 根据标准差归一化内部奖励
            normalized_intrinsic_reward = (dist_scalar / (np.sqrt(self.running_mean_std.var) + self.epsilon)).mean()
        # update the predictor network
        # self.update(torch.clone(next_obs_tensor).reshape(n_agent * n_envs, *next_obs_tensor.size()[2:]))
        self.update(rollouts)
        return beta_t * normalized_intrinsic_reward

    def update(self, rollouts):
        """Update the intrinsic reward module if necessary.

        Args:
            rollouts: The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                 next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.

        Returns:
               None
        """
        obs_tensor = rollouts['obs'].to(self.device)

        src_feats = self.predictor(obs_tensor)
        tgt_feats = self.target(obs_tensor)

        loss = F.mse_loss(src_feats, tgt_feats)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    @staticmethod
    def adjust_smac_ir(reward):
        # 缩放为SMAC最大奖励20的十分之一区间[0,2]
        """
        分段函数方式
        if 0.01 < reward < 0.2:
            return reward * 10  # 放大到2到0.1之间
        elif reward > 0.2:
            return reward + 1  # 大于0.2的放大到2到3之间
        else:
            return reward  # 其他情况不变"""
        # + 1 是为了避免对0进行对数操作
        return np.log(reward + 1) / np.log(0.068 + 1)

    @staticmethod
    def adjust_smac_ir_v2(reward_ave, reward, adj):
        # 缩放为SMAC最大奖励20的十分之一区间[0,2]
        """
        通过
        :param reward_ave: 当前的平均奖励
        :param reward: ir的奖励值
        :param adj: 列表{list:2}，说明奖励的区间是adj[0]到adj[1]
        :return: 自适应后的奖励，在0-2之间
        """

        if reward_ave == 0:
            return reward
        reward = np.log(reward + 1) / np.log(0.068 + 1)
        dist = float(abs(reward - reward_ave))
        p = dist / reward_ave
        mid = adj[0] + (adj[1] - adj[0]) / 2
        step_size = adj[1] - mid
        if reward_ave > reward:
            p = 0 - p
        return mid + step_size * p


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
