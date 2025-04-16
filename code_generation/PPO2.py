import os
# 在导入任何模块之前设置环境变量
os.environ.update({
    "SAPIEN_HEADLESS": "1",
    "SAPIEN_RENDERER": "none",  # 完全禁用渲染
    "DISPLAY": "",
    "MUJOCO_GL": "osmesa",
    "SDL_VIDEODRIVER": "dummy"
})

import gym
import numpy as np
import mani_skill2.envs
import sapien.core as sapien
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
BASE_DIR = "/root/mywork/text2reward/results"

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

def make_env(env_id, control_mode, max_episode_steps: int=None, record_dir: str=None):
    def _init() -> gym.Env:
        try:
            print(f"Creating environment with id: {env_id}, control_mode: {control_mode}")
            # 创建无渲染引擎
            env = gym.make(
                env_id, 
                obs_mode="state",  # 只使用状态观察
                reward_mode="dense", 
                control_mode=control_mode,
                renderer="none",  # 使用none禁用渲染
            )
            
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env, max_episode_steps)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
            return env
        except Exception as e:
            import traceback
            print(f"Error in environment creation: {str(e)}")
            print(traceback.format_exc())
            raise e
    return _init


# environment list
franka_list = ["LiftCube-v0", "PickCube-v0"]
mobile_list = []

def train(args):
    try:
        print("Starting training with arguments:", args)
        
        # 禁用 wandb
        os.environ["WANDB_MODE"] = "disabled"
        
        if args.reward_path is not None:
            print(f"Loading reward function from {args.reward_path}")
            with open(args.reward_path, "r") as f:
                reward_code_str = f.read()
            namespace = {}
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

        print("Setting up environment variables...")
        os.environ.update({
            "SAPIEN_HEADLESS": "1",           # 启用无头模式
            "MUJOCO_GL": "osmesa",            # 使用软件渲染
            "SAPIEN_RENDERER": "none",        # 禁用SAPIEN渲染器
            "DISPLAY": "",                    # 清除显示设置
            "SDL_VIDEODRIVER": "dummy",       # 使用虚拟显示驱动
            "EGL_PLATFORM": "surfaceless",    # 使用无表面渲染
        })

        print("Creating directories...")
        os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
        os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
        os.makedirs(f"{BASE_DIR}/tensorboard", exist_ok=True)

        print("Setting up multiprocessing...")
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        print("Testing single environment creation...")
        try:
            test_env = make_env(args.env_id, control_mode)()
            print("Single environment created successfully")
            test_env.close()
        except Exception as e:
            print(f"Error creating single environment: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e

        print("Creating evaluation environment...")
        eval_env = SubprocVecEnv(
            [make_env(args.env_id, control_mode) for i in range(args.eval_num)], 
            start_method="spawn"
        )
        eval_env = VecMonitor(eval_env)
        eval_env.seed(args.seed)
        eval_env.reset()

        print("Creating training environment...")
        env = SubprocVecEnv(
            [make_env(args.env_id, control_mode, max_episode_steps=args.max_episode_steps) for i in range(args.train_num)]
        )
        env = VecMonitor(env)
        env.seed(args.seed)
        obs = env.reset()

        print("Setting up callbacks...")
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

        print("Creating PPO model...")
        policy_kwargs = dict(net_arch=[256, 256])
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

        print("Starting training...")
        model.learn(args.train_max_steps, callback=[eval_callback])
        
        print("Saving final model...")
        model.save(f"{BASE_DIR}/models/latest_model_{args.env_id[:-3]}-our")

        print("Starting evaluation...")
        eval_env.close()
        eval_env = SubprocVecEnv([make_env(args.env_id, control_mode) for i in range(1)])
        eval_env = VecMonitor(eval_env)
        eval_env.seed(args.eval_seed)
        eval_env.reset()

        returns, ep_lens = evaluate_policy(
            model, 
            eval_env, 
            deterministic=True, 
            render=False,
            return_episode_rewards=True, 
            n_eval_episodes=20
        )

        max_steps = args.max_episode_steps
        success = np.array(ep_lens) < max_steps
        success_rate = success.mean()
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Episode Lengths: {ep_lens}")

    except Exception as e:
        print("Error occurred during training:")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--eval_num", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=12800)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--rollout_steps", type=int, default=3200)
    parser.add_argument("--train_max_steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1)
    parser.add_argument("--reward_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="default")
    
    args = parser.parse_args()
    print("Starting script with arguments:", args)
    train(args)