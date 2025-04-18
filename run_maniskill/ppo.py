import os
import gym
import numpy as np
import mani_skill2.envs
import wandb
import argparse
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from mani_skill2.utils.sapien_utils import check_actor_static

# 在文件开头添加基础路径
BASE_DIR = "/home/yy/text2reward/results"

class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        self.pre_obs = super().reset()
        return self.pre_obs
    
    def compute_dense_reward(self, action):
        assert (0)

    def step(self, action):
        ob, rew, done, info = super().step(action)
        if args.reward_path is not None:
            rew = self.compute_dense_reward(action) # TODO: uncomment this line
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info

class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        if info["success"]:
            done = True
        return ob, rew, done, info

def make_env(env_id, max_episode_steps: int=None, record_dir: str=None):
    def _init() -> gym.Env:
        import mani_skill2.envs
        env = gym.make(env_id, obs_mode="state", reward_mode="dense", control_mode=control_mode,)
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env, max_episode_steps)
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            if "eval" in record_dir:
                env = RecordEpisode(env, record_dir, info_on_video=True, render_mode="cameras")
        return env
    return _init


# environment list
franka_list = ["LiftCube-v0", "PickCube-v0"]
mobile_list = []

def train(args):
    # ... 其他导入 ...
    
    # 禁用 wandb
    os.environ["WANDB_MODE"] = "disabled"
    
    # 或者完全移除 wandb 相关代码
    """
    run = wandb.init(
        project="text2reward",
        name=f"{args.env_id}-{args.exp_name}",
        settings=None if args.reward_path is None else wandb.Settings(code_dir=args.reward_path[:-11]))
    """
    
    # ... 其余代码保持不变 ...

if __name__ == '__main__':
    # add and parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_id', type=str, default=None)
    parser.add_argument('--train_num', type=int, default=8)
    parser.add_argument('--eval_num', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=12800)
    parser.add_argument('--max_episode_steps', type=int, default=100)
    parser.add_argument('--rollout_steps', type=int, default=3200)
    parser.add_argument('--train_max_steps', type=int, default=4_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_seed', type=int, default=1)
    parser.add_argument('--project_name', type=str, default="maniskill")
    parser.add_argument('--exp_name', type=str, default="gen")
    parser.add_argument('--reward_path', type=str, default=None)
    parser.add_argument('--bo_iter', type=int, default=1)
    parser.add_argument('--task_iter', type=int, default=1)
    parser.add_argument('--sample_num', type=int, default=0)  # 添加样本编号参数

    args = parser.parse_args()

    if args.reward_path is not None:
        with open(args.reward_path, "r") as f:
            reward_code_str = f.read()
        namespace = {**globals()}
        exec(reward_code_str, namespace)
        new_function = namespace['compute_dense_reward']
        ContinuousTaskWrapper.compute_dense_reward = new_function

    if args.env_id in franka_list + mobile_list:
        if args.env_id in franka_list:
            control_mode = "pd_ee_delta_pose"
        elif args.env_id in mobile_list:
            control_mode = "base_pd_joint_vel_arm_pd_ee_delta_pose"
        else:
            assert(0)
    else:
        print("Please specify a valid environment!")
        assert(0)

    # set up eval environment
    eval_env = SubprocVecEnv([make_env(args.env_id, record_dir=f"{BASE_DIR}/videos") for i in range(args.eval_num)])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(args.seed)
    eval_env.reset()

    # set up training environment
    env = SubprocVecEnv([make_env(args.env_id, max_episode_steps=args.max_episode_steps) for i in range(args.train_num)])
    env = VecMonitor(env)
    env.seed(args.seed)
    obs = env.reset()

    # set up callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f"{BASE_DIR}/models/",
        log_path=f"{BASE_DIR}/logs/",
        eval_freq=args.eval_freq // args.train_num,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    set_random_seed(args.seed)

    # set up sac algorithm
    policy_kwargs = dict(net_arch=[256, 256])
    tb_log_name = f"{args.env_id}_{args.task_iter}_{args.sample_num}_BOiter{args.bo_iter}"
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        n_steps=args.rollout_steps // args.train_num, 
        batch_size=400, 
        n_epochs=15, 
        tensorboard_log=f"{BASE_DIR}/tensorboard",
        gamma=0.85, 
        target_kl=0.05
    )
    model.learn(
        args.train_max_steps, 
        callback=[eval_callback],
        tb_log_name=tb_log_name
    )
    model.save(f"{BASE_DIR}/models/latest_model_{args.env_id[:-3]}-our")

    # set up model evaluation environment
    eval_env.close()
    record_dir = f"{BASE_DIR}/eval_videos_{args.env_id[:-3]}-our"
    eval_env = SubprocVecEnv([make_env(args.env_id, record_dir=record_dir) for i in range(1)])
    eval_env = VecMonitor(eval_env)
    eval_env.seed(args.eval_seed)
    eval_env.reset()

    # 评估和保存视频
    returns, ep_lens = evaluate_policy(model, eval_env, deterministic=True, render=False, return_episode_rewards=True, 
                                    n_eval_episodes=20)

    # 获取环境的最大步数
    max_steps = args.max_episode_steps  # 直接使用参数中的最大步数
    success = np.array(ep_lens) < max_steps
    success_rate = success.mean()
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Episode Lengths: {ep_lens}")