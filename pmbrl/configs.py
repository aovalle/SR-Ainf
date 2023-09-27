import pprint

MOUNTAIN_CAR_CONFIG = "mountain_car"
CARTPOLE_CONFIG = "cartpole"
LUNAR_LANDER_CONFIG = "lunar_lander"
SPARSE_LUNAR_LANDER_CONFIG = "sparse_lunar_lander"
PIXELCOPTER_CONFIG = "pixelcopter"
FLAPPYBIRD_CONFIG = "flappybird"
DEBUG_CONFIG = "debug"

def print_configs():
    print(f"[{MOUNTAIN_CAR_CONFIG}, {CARTPOLE_CONFIG}, {LUNAR_LANDER_CONFIG}, {DEBUG_CONFIG}, {SPARSE_LUNAR_LANDER_CONFIG}, {PIXELCOPTER_CONFIG}, {FLAPPYBIRD_CONFIG} ")

def get_config(args):
    if args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == CARTPOLE_CONFIG:
        config = CartPoleConfig()
    elif args.config_name == LUNAR_LANDER_CONFIG:
        config = LunarLanderConfig()
    elif args.config_name == SPARSE_LUNAR_LANDER_CONFIG:
        config = SparseLunarLanderConfig()
    elif args.config_name == PIXELCOPTER_CONFIG:
        config = PixelcopterConfig()
    elif args.config_name == FLAPPYBIRD_CONFIG:
        config = FlappyBirdConfig()
    elif args.config_name == DEBUG_CONFIG:
        config = DebugConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    config.set_logdir(args.logdir)
    config.set_seed(args.seed)
    config.set_strategy(args.strategy)
    return config


class Config(object):
    def __init__(self):
        self.logdir = "log"
        self.seed = 1
        self.fixed_seed = True
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.record_every = 1
        self.coverage = False

        self.env_name = None
        self.max_episode_len = 500
        self.action_repeat = 3
        self.action_noise = None

        self.ensemble_size = 10
        self.hidden_size = 200

        self.n_train_epochs = 100
        self.batch_size = 50
        self.n_batches = 5
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.grad_clip_norm = 1000

        self.plan_horizon = 30
        self.optimisation_iters = 1#5
        self.n_candidates = 500
        self.top_candidates = 50

        self.expl_strategy = "information"
        self.use_reward = True
        self.use_exploration = True
        self.use_mean = True#False

        self.expl_scale = 1.0
        self.reward_scale = 1.0

        # RHE
        self.mutation_rate = 0.3
        self.shift_buffer_on = True

        # SMIRL
        self.smirl_param_space = 1          # Needed to augment state (number of density params)
        self.mem_size = 20#20*self.batch_size #20
        self.smirl_surprise_scale = 1.0
        self.smirl_reward_scale = 0.0#1.0

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_seed(self, seed):
        self.seed = seed

    def set_strategy(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return pprint.pformat(vars(self))


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v0"
        self.n_episodes = 5
        self.max_episode_len = 100
        self.hidden_size = 64
        self.plan_horizon = 5


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mountain_car"
        self.env_name = "SparseMountainCar" #"MountainCar-v0"
        self.max_episode_len = 500
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.expl_scale = 1.0
        self.n_episodes = 30
        self.ensemble_size = 25
        self.record_every = None
        self.n_episodes = 50

class CartPoleConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "cartpole"
        self.env_name = "CorrectedCartPole"#"CartPole-v0"
        self.max_episode_len = 500
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.expl_scale = 1.0
        self.n_episodes = 1000
        self.ensemble_size = 25
        self.record_every = None
        self.action_repeat = 1
        self.plan_horizon = 10
        # RHE
        self.mutation_rate = 0.3
        self.shift_buffer_on = True

class LunarLanderConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "lunar_lander"
        self.env_name = "LunarLander-v2"
        self.max_episode_len = 1000
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.n_episodes = 30
        self.ensemble_size = 25
        self.record_every = None
        self.n_episodes = 300
        self.action_repeat = 1
        self.plan_horizon = 10
        self.expl_scale = 0.1

class SparseLunarLanderConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "sparse_lunar_lander"
        self.env_name = "SparseLunarLander"
        self.max_episode_len = 500
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.expl_scale = 1.0
        self.n_episodes = 30
        self.ensemble_size = 25
        self.record_every = None
        self.n_episodes = 50
        self.action_repeat = 1

class PixelcopterConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "pixelcopter"
        self.env_name = "Pixelcopter-PLE-v0"
        self.max_episode_len = 10000
        self.n_train_epochs = 100
        self.n_seed_episodes = 3
        self.expl_scale = 0.1
        self.n_episodes = 300
        self.ensemble_size = 25
        self.action_repeat = 1
        self.plan_horizon = 10
        # RHE
        self.mutation_rate = 0.3
        self.shift_buffer_on = True

class FlappyBirdConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "flappybird"
        self.env_name = "FlappyBird-PLE-v0"
        self.max_episode_len = 10000
        self.n_train_epochs = 100
        self.n_seed_episodes = 3
        self.expl_scale = 0.1
        self.n_episodes = 250
        self.ensemble_size = 25
        self.action_repeat = 1
        self.plan_horizon = 15
        # RHE
        self.mutation_rate = 0.5
        self.shift_buffer_on = True