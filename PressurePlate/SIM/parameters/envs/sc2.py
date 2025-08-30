import os
src_path = os.getcwd()
replay_path = os.path.join(src_path, "results", "replays")


class SC2:
    ENV = 'sc2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    TEST_GREEDY = True
    TEST_NEPISODE = 32
    TEST_INTERVAL = 2500
    LOG_INTERVAL = 2500
    RUNNER_LOG_INTERVAL = 2500
    LEARNER_LOG_INTERVAL = 2500
    TRAIN_STEP_MAX = 10000

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 0.2
    INPUT_NOISE_CLIP = 0.5
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 100000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class SC2_3m(SC2):
    MAP = '3m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    TRAIN_STEP_MAX = 30000


class SC2_3s5z(SC2):
    MAP = '3s5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s5z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 50000

    TRAIN_STEP_MAX = 30000


class SC2_27m_vs_30m(SC2):
    MAP = '27m_vs_30m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "27m_vs_30m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_3s_vs_5z(SC2):
    MAP = '3s_vs_5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s_vs_5z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_Corridor(SC2):
    MAP = 'corridor'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "corridor",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_5m_vs_6m(SC2):
    MAP = '5m_vs_6m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "5m_vs_6m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_MMM(SC2):
    MAP = 'MMM'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "MMM",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_MMM2(SC2):
    MAP = 'MMM2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "MMM2",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_2c_vs_64zg(SC2):
    MAP = '2c_vs_64zg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2c_vs_64zg",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 30000


class SC2_6h_vs_8z(SC2):
    MAP = '6h_vs_8z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "6h_vs_8z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000000

    TRAIN_STEP_MAX = 50000


class SC2_3s5z_vs_3s6z(SC2):
    MAP = '3s5z_vs_3s6z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s5z_vs_3s6z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000000

    TRAIN_STEP_MAX = 50000

class SC2_2s_vs_1sc(SC2):
    MAP = '2s_vs_1sc'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2s_vs_1sc",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 500000

    TRAIN_STEP_MAX = 50000

class SC2_10m_vs_11m(SC2):
    MAP = '10m_vs_11m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10m_vs_11m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

class SC2_so_many_baneling(SC2):
    MAP = 'so_many_baneling'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "so_many_baneling",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

class SC2_2s3z(SC2):
    MAP = '2s3z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2s3z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

class SC2_8m(SC2):
    MAP = '8m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "8m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }
class SC2_2_corridors(SC2):
    MAP = '2_corridors'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2_corridors",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': 1,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False,
        'env_args':
        {
            'map_name': "2_corridors",  # SC2 map name
            'difficulty': "7",  # Very hard
            'move_amount': 2,  # How much units are ordered to move per step
            'step_mul': 8,  # How many frames are skiped per step
            'reward_sparse': False,  # Only +1/-1 reward for win/defeat (the rest of reward configs are ignored if True)
            'reward_only_positive': True,  # Reward is always positive
            'reward_negative_scale': 0.5,  # How much to scale negative rewards, ignored if reward_only_positive=True
            'reward_death_value': 10,
            # Reward for killing an enemy unit and penalty for having an allied unit killed (if reward_only_poitive=False)
            'reward_scale': True,  # Whether or not to scale rewards before returning to agents
            'reward_scale_rate': 20,
            # If reward_scale=True, the agents receive the reward of (max_reward / reward_scale_rate), where max_reward is the maximum possible reward per episode w/o shield regen
            'reward_win': 200,  # Reward for win
            'reward_defeat': 0,  # Reward for defeat (should be nonpositive)
            'state_last_action': True,  # Whether the last actions of units is a part of the state
            'obs_instead_of_state': False,  # Use combination of all agnets' observations as state
            'obs_own_health': True,  # Whether agents receive their own health as a part of observation
            'obs_all_health': True,
            # Whether agents receive the health of all units (in the sight range) as a part of observataion
            'continuing_episode': False,  # Stop/continue episode after its termination
            'game_version': "4.1.2",  # Ignored for Mac/Windows
            'save_replay_prefix': "",  # Prefix of the replay to be saved
            'heuristic': False,  # Whether or not use a simple nonlearning hearistic as a controller
            'restrict_actions': True,
            'obs_pathing_grid': False,  # Whether observations include pathing grid centered around agent (8 points)
            'obs_terrain_height': False,
            # Whether observations include terrain height centered around agent (8 + 1 points)
            'obs_last_action': False,
            # Whether the last action of all agents (in the sight range) are included in the obs
            'bunker_enter_range': 5,
            'seed': 1,
        }
    }