#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：random_encoder.py
@Author ：YUAN Mingqi
@Date ：2022/12/03 13:44 
'''
import numpy as np
from torch import nn
from torch.nn import functional as F, Flatten, init
import torch
from typing import Dict, Tuple


class CnnEncoder(nn.Module):
    def __init__(self, n_agent, n_obs, latent_dim):
        self.n_agent = n_agent
        self.n_obs = n_obs
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 1))
        self.flatten = nn.Flatten()
        # Calculating output size of convolutional layers
        self._to_fc = self._calculate_conv_output()

        self.fc1 = nn.Linear(self._to_fc, latent_dim)  # 根据卷积层输出的大小调整全连接层的输入维度
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.n_agent, self.n_obs)  # 将输入变换成适合卷积层的形状
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    def _calculate_conv_output(self):
        # Dummy input to calculate output size
        x = torch.randn(1, self.n_agent, self.n_obs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return x.size(1)


class MlpEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(MlpEncoder, self).__init__()
        self.obs_shape = obs_shape  # 记录观测形状
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, ob):
        x = F.relu(self.fc1(ob))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.layer_norm(x)

        return x

class Encoder(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()

        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.ones(size=tuple(obs_shape))
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.linear = nn.Linear(n_flatten, latent_dim)
        else:
            self.trunk = nn.Sequential(nn.Linear(obs_shape[0], 256), nn.ReLU())
            self.linear = nn.Linear(256, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.linear(self.trunk(obs))


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = self.input_size * self.output_size
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=2,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 2),
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=2,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 2),
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

