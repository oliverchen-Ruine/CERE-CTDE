from scipy.stats import beta
import numpy as np
from torch.distributions import Categorical
import torch


class ThompsonSampling:
    def __init__(self, N, action_dim):
        self.num = N
        self.action_dim = action_dim
        # The prior distribution uses the output result of the Actor
        # Initialize the number of successful observations and the number of attempts (for calculating rewards)
        self.observed_successes = np.ones((self.num, self.action_dim))  # Add 1 for smoothing
        self.observed_fail = np.ones((self.num, self.action_dim))  # Add 1 for smoothing

    def sample_action(self, probs) -> np.ndarray:
        '''
        :param probs:  The probability output by the Actor is taken as the prior probability
        :return: actions
        '''

        valid_actions = []
        for i in range(len(probs)):
            valid_actions.append(np.argwhere(probs[i] > 0).flatten())

        alpha_posterior = probs * self.observed_successes
        beta_posterior = (1 - probs) * self.observed_fail

        alpha_posterior[alpha_posterior == 0] = 1e-6
        beta_posterior[beta_posterior == 0] = 1e-6

        samples = []
        for i in range(self.num):
            sample = beta(alpha_posterior[i], beta_posterior[i]).rvs()
            sample[probs[i] == 0] = 0
            samples.append(sample)
        sampled_actions = []

        for i in range(len(samples)):
            sampled_actions.append(samples[i].argmax())
        sampled_actions = np.array(sampled_actions)

        return sampled_actions

    def sample_action_value(self, q_value, avail_actions) -> np.ndarray:
        '''
        TS based on Q value
        :param q_value:
        :param avail_actions:
        :param prior_beta_value:
        :return:
        '''
        # Calculate prior_alpha from positive Q values
        prior_alpha, prior_beta = self.update_beta_parameters_V(q_value)

        # Sample from the Beta distribution
        samples = beta(prior_alpha, prior_beta).rvs()

        # Mask unavailable actions
        samples[avail_actions == 0] = 0
        # Select the action with the highest sample value
        sampled_actions = samples.argmax(axis=-1)
        return np.array(sampled_actions)

    def update_prob(self, reward, sampled_actions):
        '''
            Update the number of successful and failed observations
        :param reward: Environmental rewards obtained after the action is executed
        :param sampled_actions: The action that comes down to
        :return:
        '''
        for i in range(self.num):
            action_chosen = sampled_actions[i]

            if reward > 0:  # Assuming a positive reward indicates success
                self.observed_successes[i, action_chosen] += 1
            else:
                self.observed_fail[i, action_chosen] += 1

    def calculate_log_prob(self, probs, actions):
        '''
            Calculate the logarithmic probability of the action
        '''

        log_probs = []
        for agent_probs, action in zip(probs, actions):
            dist = Categorical(probs=agent_probs.clone().detach())
            log_prob = dist.log_prob(torch.tensor(action))
            log_probs.append(log_prob.item())
        return log_probs

    def reset_observed(self):
        """
          Reset success and failure
        :return:
        """
        # self.observed_successes = np.ones((self.num, self.action_dim))  # 加1进行平滑处理
        self.observed_fail = np.ones((self.num, self.action_dim))  # 加1进行平滑处理

    def update_beta_parameters_V(self, q_value, prior_beta_value=1.0):
        '''

        :param q_value:
        :param prior_beta_value:
        :return:
        '''
        # Normalize Q values using softmax
        normalized_q_values = self.softmax(q_value)
        prior_alpha = normalized_q_values * self.observed_successes
        prior_beta = (1 - normalized_q_values) * self.observed_fail
        prior_alpha[prior_alpha <= 0] = 1e-6
        prior_beta[prior_beta <= 0] = 1e-6


        return prior_alpha, prior_beta

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
