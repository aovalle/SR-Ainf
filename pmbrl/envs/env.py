import gym
import gym_pygame

SPARSE_MOUNTAIN_CAR = "SparseMountainCar"
HALF_CHEETAH_RUN = "HalfCheetahRun"
HALF_CHEETAH_FLIP = "HalfCheetahFlip"
ANT_MAZE = "AntMaze"
DM_REACHER = "DeepMindReacher"
DM_CATCH = "DeepMindCatch"

MOUNTAIN_CAR = 'SparseMountainCar' #'MountainCar-v0'
#CARTPOLE = 'CartPole-v0'
CARTPOLE = 'CorrectedCartPole'
LUNAR_LANDER = 'LunarLander-v2'
SPARSE_LUNAR_LANDER = 'SparseLunarLander'
PIXELCOPTER = 'Pixelcopter-PLE-v0'
FLAPPYBIRD = 'FlappyBird-PLE-v0'


class GymEnv(object):
    def __init__(self, env_name, max_episode_len, action_repeat=1, seed=None, fixed_seed=False):
        self._env = self._get_env_object(env_name)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        self.seed = seed
        self.fixed_seed = fixed_seed
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        if self.fixed_seed and self.seed is not None:
            self._env.seed(self.seed)
        state = self._env.reset()
        self.done = False
        return state

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                print('LEAVING AT t ', self.t)
                self.done = True
                break
        return state, reward, done, info

    def sample_action(self):
        return self._env.action_space.sample()

    def render(self, mode="human"):
        self._env.render(mode)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env

    def _get_env_object(self, env_name):
        if env_name == SPARSE_MOUNTAIN_CAR:
            from pmbrl.envs.envs.mountain_car import SparseMountainCarEnv

            return SparseMountainCarEnv()

        elif env_name == HALF_CHEETAH_RUN:
            from pmbrl.envs.envs.half_cheetah_run import HalfCheetahRunEnv

            return HalfCheetahRunEnv()

        elif env_name == HALF_CHEETAH_FLIP:
            from pmbrl.envs.envs.half_cheetah_flip import HalfCheetahFlipEnv

            return HalfCheetahFlipEnv()

        elif env_name == ANT_MAZE:
            from pmbrl.envs.envs.ant import SparseAntEnv

            return SparseAntEnv()

        elif env_name == DM_CATCH:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            return DeepMindWrapper(domain="ball_in_cup", task="catch")

        elif env_name == DM_REACHER:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            return DeepMindWrapper(domain="reacher", task="easy")

        elif env_name == MOUNTAIN_CAR:  # Discrete
            from pmbrl.envs.envs.mountain_car import SparseMountainCarEnv
            return SparseMountainCarEnv()

        elif env_name == SPARSE_LUNAR_LANDER:  # Discrete
            from pmbrl.envs.envs.lunar_lander import SparseLunarLander
            return SparseLunarLander()

        elif env_name == CARTPOLE:
            from pmbrl.envs.envs.cartpole import CorrectedCartPoleEnv
            return CorrectedCartPoleEnv()
            #return gym.make(env_name)
        elif env_name == LUNAR_LANDER:
            return gym.make(env_name)

        else:
            return gym.make(env_name)
