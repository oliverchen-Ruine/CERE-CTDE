from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_pp import MAPPO_PP
from env.pressureplate import PressurePlateWrapper


class Runner_MAPPO_PP:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = PressurePlateWrapper(map_name=self.env_name)
        if self.number !=1:
            self.env.set_random_position()
        self.half_t_max = int(args.max_train_steps / 2)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_PP(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.win_rate_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/win_rate'.format(self.env_name, self.number, self.seed))
        self.episode_length_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/episode_length'.format(self.env_name, self.number,
                                                                                      self.seed))
        self.reward_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/evaluate_reward'.format(self.env_name, self.number,
                                                                                       self.seed))
        self.entropy_mean_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/entropy/mean'.format(self.env_name, self.number,
                                                                                    self.seed))
        self.entropy_weight_writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}/entropy/weight'.format(self.env_name, self.number,
                                                                                      self.seed))

        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, eval_entropy, episode_steps = self.run_episode_smac(evaluate=False,
                                                                      t_s=self.total_steps)  # Run an episode
            self.entropy_mean_writer.add_scalar('eval_entropy_mean{}'.format(self.env_name), eval_entropy[0],
                                                global_step=self.total_steps)
            self.entropy_weight_writer.add_scalar('eval_entropy_weight{}'.format(self.env_name), eval_entropy[1],
                                                  global_step=self.total_steps)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        episode_length = 0
        evaluate_reward = 0
        win_num = 0
        for _ in range(self.args.evaluate_times):
            win_tage, episode_reward, _, episode_step = self.run_episode_smac(evaluate=True, t_s=self.total_steps)
            episode_length += episode_step
            evaluate_reward += episode_reward
            # win_num = win_num + 1 if win_tage else win_num
            if win_tage:
                win_num += 1

        evaluate_episode_length = episode_length / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        win_rate = win_num / self.args.evaluate_times
        print("total_steps:{} \t eval_win_rate:{} \t eval_episode_length:{} \t eval_reward:{}".format(self.total_steps,
                                                                                                     win_rate,
                                                                                                     evaluate_episode_length,
                                                                                                     evaluate_reward))
        self.win_rate_writer.add_scalar('eval_win_rate{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        self.episode_length_writer.add_scalar('eval_episode_length_{}'.format(self.env_name), evaluate_episode_length,
                                              global_step=self.total_steps)
        self.reward_writer.add_scalar('eval_reward_{}'.format(self.env_name), evaluate_reward,
                                      global_step=self.total_steps)
        # Save the win rates
        # np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
        #         np.array(self.win_rates))
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_smac(self, evaluate=False, t_s=0):
        win_tag = False
        episode_reward = 0
        episode_entropy_log = []

        all_obs, state, env_info = self.env.reset()

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = all_obs  # obs_n.shape=(N,obs_dim)
            s = np.concatenate(all_obs, 0)  # s.shape=(state_dim,)
            avail_a_n = env_info['avail_actions']  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            a_n, a_logprob_n, entropy_log = self.agent_n.choose_action(obs_n, avail_a_n,
                                                                       evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            episode_entropy_log.append(entropy_log)
            # a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)
            v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            all_obs, r, done, env_info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in env_info and env_info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif args.use_reward_scaling:
                    r = self.reward_scaling(r)
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

            if done:
                break
        episode_entropy_mean = None
        episode_entropy_weighted = None
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = all_obs
            s = np.concatenate(all_obs, 0)
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
            episode_entropy_mean = self.compute_episode_entropy(episode_entropy_log, mode='mean')
            episode_entropy_weighted = self.compute_episode_entropy(episode_entropy_log, mode='weighted')

        return win_tag, episode_reward, (episode_entropy_mean, episode_entropy_weighted), episode_step + 1

    @staticmethod
    def compute_episode_entropy(episode_entropy_log, mode='mean'):

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
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False,
                        help="Trick 4:reward scaling. Here, we do not use it.")
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
    env_names = ['linear-4p']
    number = 2
    for _ in range(6):
        current_date = datetime.now()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        hour = current_date.hour
        minute = current_date.minute
        seed_value = int(f"{month:02d}{day:02d}{hour:02d}{minute:02d}")
        for env_index in range(len(env_names)):
            runner = Runner_MAPPO_PP(args, env_name=env_names[env_index], number=number, seed=seed_value)
            runner.run()
