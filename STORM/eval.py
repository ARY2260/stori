import gymnasium
import argparse
import cv2
import numpy as np
import pandas as pd
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os
import ruamel.yaml as yaml
import pathlib

from utils import seed_np_torch, Logger, load_config, args_type, nested_dict_from_flat, flatten_dict
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
from stochastic_atari import create_stochasticity_profile
import ale_py


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size, stochasticity_config):
    skip=4
    gymnasium.register_envs(ale_py)
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
    game_name = env_name[4:-3] # Remove first 4 characters ("ALE/") and last 3 characters ("-v5")
    stochasticity_config['intrinsic_stochasticity']['action_independent_concept_drift']['skip'] = skip
    stochasticity_profile = create_stochasticity_profile(game_name=game_name, type=stochasticity_config['stochasticity_type'], config=stochasticity_config)
    env = stochasticity_profile.get_env(env)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=skip)
    env = gymnasium.wrappers.ResizeObservation(env, shape=(image_size, image_size))
    return env


def build_vec_env(env_name, image_size, num_envs, stochasticity_config):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, stochasticity_config)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, stochasticity_config):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, stochasticity_config=stochasticity_config)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )

        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episode:
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        return np.mean(final_rewards)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part


def save_tensorboard_event(logdir, csv_path):
    df = pd.read_csv(csv_path)
    # Use SummaryWriter which handles proper TensorBoard event file creation
    writer = SummaryWriter(log_dir=logdir)

    # Write each row as scalar data
    for _, row in df.iterrows():
        step = int(row["step"])
        
        for col in df.columns:
            if col == "step":
                continue
            
            # Add scalar data using SummaryWriter
            writer.add_scalar(col, float(row[col]), step)

    # Close the writer to ensure all data is flushed
    writer.close()


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-stochasticity_config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    args, remaining_stochastic_args = parser.parse_known_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # add stochasticity config
    stochasticity_config = yaml.safe_load(pathlib.Path(args.stochasticity_config_path).read_text())
    stochasticity_config = stochasticity_config["defaults"]

    stochasticity_parser = argparse.ArgumentParser()
    # Also flatten stochasticity_config and add to parser
    flat_stoch = flatten_dict(stochasticity_config)
    for key, value in sorted(flat_stoch, key=lambda x: x[0]):
        arg_type = args_type(value)
        stochasticity_parser.add_argument(f"--{key}", type=arg_type, default=value)

    # Parse args and build nested dict from flat keys
    parsed_args = stochasticity_parser.parse_args(remaining_stochastic_args)
    args_dict = vars(parsed_args)

    nested_args = nested_dict_from_flat(args_dict)

    stochasticity_config.update(nested_args)
    print("Stochasticity config", stochasticity_config)

    # set seed
    os.makedirs("eval_result", exist_ok=True)
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, stochasticity_config=stochasticity_config)
    action_dim = dummy_env.action_space.n
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    # steps = steps[-1:]
    print(steps)
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        episode_avg_return = eval_episodes(
            num_episode=100,
            env_name=args.env_name,
            num_envs=5,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent,
            stochasticity_config=stochasticity_config,
        )
        results.append([step, episode_avg_return])
    with open(f"eval_result/{args.run_name}.csv", "w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")

    save_tensorboard_event(logdir=f"runs/{args.run_name}", csv_path=f"eval_result/{args.run_name}.csv")
