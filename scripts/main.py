import sys
import time
import pathlib
import argparse
import time
import os

import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tensorboardX import SummaryWriter
import wandb

# Add ../../
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Agent, RollingHorizon
from pmbrl.utils import Logger
from pmbrl import get_config

# headless games
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    time_stamp = time.strftime("%d%m%Y-%H%M")

    # Tensorboard and Wandb init
    # tbWriter = SummaryWriter('tensorboard/Cartpole-{}'.format(time_stamp), comment='FEP RHE')
    # tbWriter = SummaryWriter('tensorboard/Lunarlander-{}'.format(time_stamp), comment='FEP RHE')
    #tbWriter = SummaryWriter('tensorboard/Flappybird-SMIRL-FEP-{}'.format(time_stamp), comment='FEP RHE')
    # tbWriter = SummaryWriter('tensorboard/Mountain-Car-{}'.format(time_stamp), comment='FEP RHE')
    #wandb.init(project="smirl-fep", dir='.', config=args, name='smirl-fep-flappy{}'.format(time_stamp))

    logger = Logger(args.logdir, args.seed)
    logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
    logger.log(args)

    rate_buffer = None
    if args.coverage:
        from pmbrl.envs.envs.ant import rate_buffer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Init env
    env = GymEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed, fixed_seed=args.fixed_seed)

    #action_size = env.action_space.shape[0]
    action_dim = 1
    # TODO: DO ONE-HOTE INSTEAD FROM THE BEGINNING
    # It'll be one-hot
    action_space = env.action_space.n
    state_size = env.observation_space.shape[0]
    print(action_space, state_size, env.action_space)

    normalizer = Normalizer()
    buffer = Buffer(state_size, action_dim, args.ensemble_size, normalizer, device=DEVICE)

    ensemble = EnsembleModel(
        state_size + args.smirl_param_space + action_space,
        state_size + args.smirl_param_space,
        args.hidden_size,
        args.ensemble_size,
        action_space,
        normalizer,
        device=DEVICE,
    )
    reward_model = RewardModel(state_size + args.smirl_param_space + action_space, args.hidden_size, action_space, device=DEVICE)
    trainer = Trainer(
        ensemble,
        reward_model,
        buffer,
        n_train_epochs=args.n_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        grad_clip_norm=args.grad_clip_norm,
        logger=logger,
    )

    rhe_planner = RollingHorizon(args, ensemble, reward_model, action_space, action_dim)
    agent = Agent(args, env, rhe_planner, logger=logger)

    # Collect Episodes
    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

    # Main loop
    for episode in range(1, args.n_episodes):
        logger.log("\n=== Episode {} ===".format(episode))
        start_time = time.time()

        msg = "Training on [{}/{}] data points"
        logger.log(msg.format(buffer.n_batches*args.batch_size, buffer.total_steps * args.action_repeat))
        trainer.reset_models()

        # Learn model with experience buffer
        ensemble_loss, mu_loss, reward_loss, err, mu_err = trainer.learn_model()
        logger.log_losses(ensemble_loss, reward_loss)

        recorder = None
        if args.record_every is not None and episode % args.record_every == 0:
            filename = logger.get_video_path(episode)
            recorder = VideoRecorder(env.unwrapped, path=filename)
            logger.log("Setup recoder @ {}".format(filename))

        logger.log("\n=== Collecting data [{}] ===".format(episode))

        # Plan and act in real world
        reward, r_smirl, comb_reward, mus, interoception, steps, stats = agent.run_episode(buffer, action_noise=args.action_noise, recorder=recorder)

        logger.log_episode(reward, steps)
        print('Episode mus {} Rolling interoception {}'.format(np.mean(mus), np.mean(interoception)))
        logger.log_stats(stats)

        if args.coverage:
            coverage = rate_buffer(buffer=buffer)
            logger.log_coverage(coverage)

        logger.log_time(time.time() - start_time)
        logger.save()

        # tbWriter.add_scalar('data/reward', reward, episode)
        # tbWriter.add_scalar('data/steps', steps, episode)
        # tbWriter.add_scalar('data/ensemble_loss', ensemble_loss, episode)
        # tbWriter.add_scalar('data/mu_loss', mu_loss, episode)
        # tbWriter.add_scalar('data/error', err, episode)
        # tbWriter.add_scalar('data/mu_error', mu_err, episode)
        # tbWriter.add_scalar('data/reward_loss', reward_loss, episode)
        # tbWriter.add_scalar('data/pred_reward_mean', float(stats[0]['mean']), episode)
        # tbWriter.add_scalar('data/pred_reward_std', float(stats[0]['std']), episode)
        # tbWriter.add_scalar('data/pred_reward_min', float(stats[0]['min']), episode)
        # tbWriter.add_scalar('data/pred_intrinsic_mean', float(stats[1]['mean']), episode)
        # tbWriter.add_scalar('data/pred_intrinsic_mu_mean', float(stats[2]['mean']), episode)
        # tbWriter.add_scalar('data/pred_intrinsic_std', float(stats[1]['std']), episode)
        # tbWriter.add_scalar('data/pred_intrinsic_mu_std', float(stats[2]['std']), episode)
        # tbWriter.add_scalar('data/episodic_mu', np.mean(mus), episode)
        # tbWriter.add_scalar('data/rolling_interoception', np.mean(interoception), episode)
        # tbWriter.flush()
        #
        # wandb.log({"Reward": reward, "Episode": episode})
        # wandb.log({"Steps": steps, "Episode": episode})
        # wandb.log({"Ensemble loss": ensemble_loss, "Episode": episode})
        # wandb.log({"Mu loss": mu_loss, "Episode": episode})
        # wandb.log({"Error": err, "Episode": episode})
        # wandb.log({"Mu error": mu_err, "Episode": episode})
        # wandb.log({"Reward loss": reward_loss, "Episode": episode})
        # wandb.log({"Pred reward mean": float(stats[0]['mean']), "Episode": episode})
        # wandb.log({"Pred reward std": float(stats[0]['std']), "Episode": episode})
        # wandb.log({"Pred reward min": float(stats[0]['min']), "Episode": episode})
        # wandb.log({"Pred intrinsic mean": float(stats[1]['mean']), "Episode": episode})
        # wandb.log({"Pred intrinsic mu mean": float(stats[2]['mean']), "Episode": episode})
        # wandb.log({"Pred intrinsic std": float(stats[1]['std']), "Episode": episode})
        # wandb.log({"Pred intrinsic mu std": float(stats[2]['std']), "Episode": episode})
        # wandb.log({"Episodic mu": np.mean(mus), "Episode": episode})
        # wandb.log({"Rolling interoception": np.mean(interoception), "Episode": episode})

    # tbWriter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    #parser.add_argument("--config_name", type=str, default="mountain_car")
    #parser.add_argument("--config_name", type=str, default="sparse_lunar_lander")
    # parser.add_argument("--config_name", type=str, default="lunar_lander")
    #parser.add_argument("--config_name", type=str, default="pixelcopter")
    parser.add_argument("--config_name", type=str, default="flappybird")
    #parser.add_argument("--config_name", type=str, default="cartpole")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = get_config(args)
    main(config)
