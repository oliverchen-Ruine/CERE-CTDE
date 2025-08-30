import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # Initialise the hyper networks with a fixed variance, if specified
        if self.args.hyper_initialization_nonzeros > 0:
            std = self.args.hyper_initialization_nonzeros ** -0.5
            self.hyper_w_1.weight.data.normal_(std=std)
            self.hyper_w_1.bias.data.normal_(std=std)
            self.hyper_w_final.weight.data.normal_(std=std)
            self.hyper_w_final.bias.data.normal_(std=std)

        # Initialise the hyper-network of the skip-connections, such that the result is close to VDN
        if self.args.skip_connections:
            self.skip_connections = nn.Linear(self.state_dim, self.args.n_agents, bias=True)
            self.skip_connections.bias.data.fill_(1.0)  # bias produces initial VDN weights

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.args.skip_connections:
            ws = th.abs(self.skip_connections(states)).view(-1, self.n_agents, 1)
            s = th.bmm(agent_qs, ws)
        # Compute final output
        y = th.bmm(hidden, w_final) + v + s
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
        # th.autograd.set_detect_anomaly(True)  # 添加这行
        # # 保护性复制
        # if agent_qs.requires_grad:
        #     agent_qs = agent_qs.clone()
        # if states.requires_grad:
        #     states = states.clone()
        #
        # bs = agent_qs.size(0)
        # states = states.reshape(-1, self.state_dim)
        # agent_qs = agent_qs.view(-1, 1, self.n_agents)
        #
        # # 第一层（添加detach检查）
        # w1 = th.abs(self.hyper_w_1(states.detach())).requires_grad_()
        # b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        # w1 = w1.view(-1, self.n_agents, self.embed_dim)
        #
        # hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        #
        # # 第二层
        # w_final = th.abs(self.hyper_w_final(states)).view(-1, self.embed_dim, 1)
        # v = self.V(states).view(-1, 1, 1)
        #
        # # Skip connections（特别保护）
        # s = 0
        # if self.args.skip_connections:
        #     with th.no_grad():  # 先确保不计算梯度
        #         ws = th.abs(self.skip_connections(states)).view(-1, self.n_agents, 1)
        #     s = th.bmm(agent_qs, ws.requires_grad_())
        #
        # y = th.bmm(hidden, w_final) + v + s
        # return y.view(bs, -1, 1)
