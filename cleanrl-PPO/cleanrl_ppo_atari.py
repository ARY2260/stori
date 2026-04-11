# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from stochastic_atari import create_stochasticity_profile
import ale_py
import argparse
import ruamel.yaml as yaml
import pathlib
from utils import args_type, nested_dict_from_flat, flatten_dict

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_name: str = "ALE/Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # stochasticity arguments
    stochasticity_config_path: str = "/home/ary2260/Work/NUS_Research/RL_Stochastic_Benchmarks/stori/cleanrl-PPO/configs_stochastic.yaml"
    """the path to the stochasticity config file"""

def make_env(env_name, idx, capture_video, run_name, stochasticity_config):
    def thunk():
        skip=4
        gym.register_envs(ale_py)
        env = gym.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
        game_name = env_name[4:-3] # Remove first 4 characters ("ALE/") and last 3 characters ("-v5")
        stochasticity_config['intrinsic_stochasticity']['action_independent_concept_drift']['skip'] = skip
        stochasticity_profile = create_stochasticity_profile(game_name=game_name, type=stochasticity_config['stochasticity_type'], config=stochasticity_config)
        env = stochasticity_profile.get_env(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=skip)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.Linear(64 * 7 * 7, 512)),
            layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def evaluate(
    model_path: str,
    env_name: str,
    num_envs: int,
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    stochasticity_config: Optional[dict] = None,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, capture_video, run_name, stochasticity_config) for i in range(num_envs)],
    )
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # obs, _ = envs.reset()
    # episodic_returns = []
    # while len(episodic_returns) < eval_episodes:
    #     actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
    #     next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
    #     if "final_info" in infos:
    #         print("infos:", infos)
    #         for info in infos["final_info"]:
    #             if "episode" not in info:
    #                 continue
    #             print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
    #             episodic_returns += [info["episode"]["r"]]
    #     obs = next_obs

    sum_reward = np.zeros(num_envs)
    current_obs, current_info = envs.reset()

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # sample part >>>
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(current_obs).to(device))
        obs, reward, done, truncated, info = envs.step(actions.cpu().numpy())
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == eval_episodes:
                        envs.close()
                        print("Mean reward: ", np.mean(final_rewards))
                        return np.mean(final_rewards)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs

if __name__ == "__main__":
    args, remaining_stochastic_args = tyro.cli(Args, return_unknown_args=True)

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

    print("stochasticity_config:", stochasticity_config)
    print("exp_name:", args.exp_name)
    print("env_name:", args.env_name)
    print("seed:", args.seed)
    print("total_timesteps:", args.total_timesteps)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print("num_iterations:", args.num_iterations, "batch_size:", args.batch_size, "minibatch_size:", args.minibatch_size)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup env_name, idx, capture_video, run_name, stochasticity_config
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name, stochasticity_config) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    last_eval_step = 0
    eval_interval = args.total_timesteps // 10

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Periodic evaluation every 10k steps
        if (global_step - last_eval_step >= eval_interval) or np.isclose(global_step, 100000, atol=2000):
            os.makedirs("models", exist_ok=True)
            # Save checkpoint
            model_path = f"models/{run_name}_step{global_step}.pt"
            torch.save(agent.state_dict(), model_path)

            # Evaluate
            episode_avg_return = evaluate(
                model_path, 
                args.env_name, 
                5,
                100, 
                run_name, 
                device, 
                False, 
                stochasticity_config
            )

            os.remove(model_path)
            # Log to TensorBoard
            writer.add_scalar("eval/mean_episodic_return", episode_avg_return, global_step)
            # writer.add_scalar("eval/std_episodic_return", std_return, global_step)

            # Save to CSV (if you want the STORM-style CSV)
            os.makedirs("eval_result", exist_ok=True)
            csv_path = f"eval_result/{run_name}.csv"

            # Append to CSV (create header if file doesn't exist)
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a") as fout:
                if not file_exists:
                    fout.write("step,episode_avg_return\n")
                fout.write(f"{global_step},{episode_avg_return}\n")

            print(f"Eval at step {global_step} - Mean return: {episode_avg_return}")
            last_eval_step = global_step

    # Final evaluation (keep existing code at the end)
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/{run_name}_step{global_step}")
    episode_avg_return = evaluate(f"models/{run_name}_step{global_step}", args.env_name, 1, 1, f"{run_name}_eval_final", device, args.capture_video, stochasticity_config)
    # eval_returns = evaluate(f"models/{run_name}_step{global_step}", args.env_name, 5, 100, f"{run_name}_eval_final", device, args.capture_video, stochasticity_config)

    # mean_return = np.mean(eval_returns)
    writer.add_scalar("eval/mean_episodic_return", episode_avg_return, global_step)
    # writer.add_scalar("eval/std_episodic_return", np.std(eval_returns), global_step)
    print(f"Final eval - Mean return: {episode_avg_return}")

    csv_path = f"eval_result/{run_name}.csv"
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a") as fout:
        if not file_exists:
            fout.write("step,episode_avg_return\n")
        fout.write(f"{global_step},{episode_avg_return}\n")

    envs.close()
    writer.close()
