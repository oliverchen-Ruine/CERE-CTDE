import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env.pressureplate import PressurePlateWrapper
import argparse
from replay_buffer import ReplayBuffer
from qmix_pp import QMIX_SMAC
from normalization import Normalization


class Runner_QMIX_PP:
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
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm,
                                                                   self.env_name, self.number, self.seed))

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, eval_entropy, episode_steps = self.run_episode(evaluate=False)  # Run an episode
            self.total_steps += episode_steps
            self.writer.add_scalar('eval_entropy_mean{}'.format(self.env_name), eval_entropy[0],
                                   global_step=self.total_steps)
            self.writer.add_scalar('eval_entropy_weight{}'.format(self.env_name), eval_entropy[1],
                                   global_step=self.total_steps)

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        episode_length = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _, episode_step = self.run_episode(evaluate=True)
            episode_length += episode_step
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        evaluate_episode_length = episode_length / self.args.evaluate_times
        print("total_steps:{} \t eval_win_rate:{} \t eval_episode_length:{} \t eval_reward:{}".format(self.total_steps,
                                                                                                      win_rate,
                                                                                                      evaluate_episode_length,
                                                                                                      evaluate_reward))
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_reward{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        self.writer.add_scalar('eval_episode_length_{}'.format(self.env_name), evaluate_episode_length,
                                              global_step=self.total_steps)

    def run_episode(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        episode_entropy_log = []
        all_obs, state, env_info = self.env.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim), dtype=np.float16)  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n = all_obs  # obs_n.shape=(N,obs_dim)
            s = np.concatenate(all_obs, 0)  # s.shape=(state_dim,)
            avail_a_n = env_info['avail_actions']  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            if evaluate:
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            else:
                a_n, entropy_log = self.agent_n.choose_action_entropy(obs_n, last_onehot_a_n, avail_a_n, epsilon)
                episode_entropy_log.append(entropy_log)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            all_obs, r, done, env_info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in env_info and env_info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
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
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break
        episode_entropy_weighted = None
        episode_entropy_mean = None
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = all_obs
            s = np.concatenate(all_obs, 0)
            avail_a_n = env_info['avail_actions']
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)
            episode_entropy_weighted = self.compute_episode_entropy(episode_entropy_log, mode='weighted')
            episode_entropy_mean = self.compute_episode_entropy(episode_entropy_log, mode='mean')

        return win_tag, episode_reward, (episode_entropy_mean, episode_entropy_weighted), episode_step + 1

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
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000,
                        help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32,
                        help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64,
                        help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True,
                        help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    args.algorithm = "VDN"
    env_names = ['linear-4p']
    number = 2
    for _ in range(6):
        current_date = datetime.datetime.now()
        month = current_date.month
        day = current_date.day
        hour = current_date.hour
        minute = current_date.minute
        seed_value = int(f"{month:02d}{day:02d}{hour:02d}{minute:02d}")
        for env_index in range(len(env_names)):
            runner = Runner_QMIX_PP(args, env_name=env_names[env_index], number=number, seed=seed_value)
            runner.run()
