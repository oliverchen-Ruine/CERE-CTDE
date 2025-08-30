from datetime import datetime
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_smac import MAPPO_SMAC

# from maven.envs.starcraft2.starcraft2 import SC2
from smac.env import StarCraft2Env

import gym


class Runner_MAPPO_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
        # self.env = SC2(env_args=args.env_args)
        self.half_t_max = int(args.max_train_steps / 2)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        # The observation dimension of rnd n_agents * (state_dim + obs_dim)
        self.args.rnd_dim = gym.spaces.Box(0, 255, shape=(self.env_info["n_agents"], self.env_info["state_shape"] +
                                                          self.env_info["obs_shape"]), dtype=np.float32).shape

        self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.win_rate_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/win_rate'.format(self.env_name, self.number, self.seed))
        self.reward_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/evaluate_reward'.format(self.env_name, self.number,
                                                                                       self.seed))
        # self.entropy_mean_writer = SummaryWriter(
        #     log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/entropy/mean'.format(self.env_name, self.number,
        #                                                                             self.seed))
        # self.entropy_weight_writer = SummaryWriter(
        #     log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/entropy/weight'.format(self.env_name, self.number,
        #                                                                               self.seed))

        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            # Both internal and external rewards are subject to the same discount rate
            self.in_reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
            self.ext_reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma_ext)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        current_avg_ir = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy(evaluate_num)  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, eval_entropy, episode_steps, current_total_ir = self.run_episode_smac(evaluate=False,
                                                                                        current_avg_ir=current_avg_ir,
                                                                                        t_s=self.total_steps)  # Run an episode
            # self.entropy_mean_writer.add_scalar('eval_entropy_mean{}'.format(self.env_name), eval_entropy[0],
            #                                global_step=self.total_steps)
            # self.entropy_weight_writer.add_scalar('eval_entropy_weight{}'.format(self.env_name), eval_entropy[1],
            #                                     global_step=self.total_steps)
            self.total_steps += episode_steps
            # current_avg_ir = current_total_ir / episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy(evaluate_num)
        self.env.close()

    def evaluate_policy(self, evaluate_num):
        win_times = 0
        evaluate_reward = 0

        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _, _, _ = self.run_episode_smac(evaluate=True, t_s=self.total_steps)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times

        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        self.win_rate_writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        self.reward_writer.add_scalar('evaluate_reward_{}'.format(self.env_name), evaluate_reward,
                                      global_step=self.total_steps)
        # Save the win rates
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
                np.array(self.win_rates))
        if win_rate == 0:
            self.agent_n.ts.reset_observed()
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_smac(self, evaluate=False, current_avg_ir=-1, t_s=0):
        win_tag = False
        episode_reward = 0
        current_total_ir = 0
        # episode_entropy_log = []
        self.env.reset()
        if '2_corridors' in self.env_name and t_s >= self.half_t_max:
            self.env.close_corridor()
        if self.args.use_reward_scaling:
            self.ext_reward_scaling.reset()
            self.in_reward_scaling.reset()
        if self.args.use_rnn:  # If we use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            a_n, a_logprob_n = self.agent_n.choose_action_ts(obs_n, avail_a_n,
                                                             evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            # episode_entropy_log.append(entropy_log)
            v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            r, done, info = self.env.step(a_n)  # Take a step
            self.agent_n.ts.update_prob(r, a_n)
            if not evaluate:
                ir = self.agent_n.get_intrinsic_reward(self.env.get_state(), self.env.get_obs(),
                                                       self.total_steps + episode_step,
                                                       current_avg_ir)
                total_reward = r + ir * self.args.weight_internal_reward * np.power(1. - 0.25e-05,
                                                                                    self.total_steps + episode_step + 1)

                current_total_ir += ir

                if self.args.use_reward_norm:
                    # Internal rewards have been standardized, so there is no need to do it again
                    r = self.reward_norm(total_reward)
                elif args.use_reward_scaling:
                    r = self.ext_reward_scaling(r) + self.in_reward_scaling(
                        ir) * self.args.weight_internal_reward * np.power(1. - 0.25e-05,
                                                                          self.total_steps + episode_step + 1)

                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if done:
                break
        episode_entropy_mean = None
        episode_entropy_weighted = None
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
            # episode_entropy_mean = self.compute_episode_entropy(episode_entropy_log, mode='mean')
            # episode_entropy_weighted = self.compute_episode_entropy(episode_entropy_log, mode='weighted')

        return win_tag, episode_reward, (
            episode_entropy_mean, episode_entropy_weighted), episode_step + 1, current_total_ir

    def compute_episode_entropy(self, episode_entropy_log, mode='mean'):
        if mode == 'mean':
            total_entropy = sum([e for e, _ in episode_entropy_log])
            episode_entropy = total_entropy / len(episode_entropy_log)
        elif mode == 'weighted':
            total_weighted_entropy = sum([e * n for e, n in episode_entropy_log])
            total_alive = sum([n for _, n in episode_entropy_log])
            episode_entropy = total_weighted_entropy / (total_alive + 1e-10)
        else:
            raise ValueError("Unsupported mode. Choose 'mean' or 'weighted'.")

        return episode_entropy


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gamma_ext", type=float, default=0.99, help="Extrinsic rewards discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False,
                        help="Trick 3:reward normalization. Here, we do not use it.")
    parser.add_argument("--use_reward_scaling", type=bool, default=True,
                        help="Trick 4:reward scaling. ")
    parser.add_argument("--weight_internal_reward", type=float, default=2.0, help="Weight of the internal reward")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific", type=float, default=True,
                        help="Whether to use agent specific global state.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    args = parser.parse_args()
    args.weight_internal_reward = 2.0
    # Environment name
    env_names = ['so_many_baneling', '8m', 'MMM', '10m_vs_11m', '3s5z']
    for _ in range(6):
        current_date = datetime.now()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        hour = current_date.hour
        minute = current_date.minute
        # Combine the month, day, hour and minute into seed values
        seed_value = int(f"{month:02d}{day:02d}{hour:02d}{minute:02d}")  # Format as MMDDHHMM
        env_index = 0
        for env_index in range(len(env_names)):
            runner = Runner_MAPPO_SMAC(args, env_name=env_names[env_index], number=1, seed=seed_value)
            runner.run()
