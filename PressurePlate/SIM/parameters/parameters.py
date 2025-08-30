from .parameters_preamble import *

class Parameters_3S5Zvs3S6Z(ExpQmixSc3S5Zvs3S6Z):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_somanybaneling(ExpQmixScsomanybaneling):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_2s3z(ExpQmixSc2s3z):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_8m(ExpQmixSc8m):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_MMM(ExpQmixScMMM):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_10Mvs11M(ExpQmixSc10mvs11m):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_3s5z(ExpQmixSc3S5Z):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0


class Parameters_MMM2(ExpQmixScMMM2):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_Corridor(ExpQmixScCorridor):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_6Hvs8Z(ExpQmixSc6Hvs8Z):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_27Mvs30M(ExpQmixSc27Mvs30M):
    use_tensorboard = True
    RUNS = 1
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_2_corridors(ExpQmixSc2_2_corridors):
    use_tensorboard = True
    RUNS = 3
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TRAIN_STEP_MAX = 50000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False

    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_PP4P(ExpQmixPP4P):
    use_tensorboard = True
    RUNS = 6
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False
    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0
    IS_D = True

class Parameters_PP5P(ExpQmixPP5P):
    use_tensorboard = True
    RUNS = 3
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False
    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0

class Parameters_PP6P(ExpQmixPP6P):
    use_tensorboard = True
    RUNS = 6
    DEVICE = 'cuda:0'
    T_MAX = 1000000
    TEST_NEPISODE = 32  # Number of episodes to test for
    TEST_INTERVAL = 5000  # Test after {} timesteps have passed
    LOG_INTERVAL = 5000
    BUFFER_CPU_ONLY = False
    EPSILON_ANNEAL_TIME = 50000
    EPSILON_UPDATE_STANDARD = 'steps'
    HIDDEN_POLICY = True
    RHO = 0.5
    BETA = 1.0