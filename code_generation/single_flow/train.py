import os
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import mani_skill2.envs

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_file):
        super().__init__(env)
        self.env = env
        
        # 加载奖励函数
        with open(reward_file, 'r') as f:
            reward_code = f.read()
        local_dict = {}
        exec(reward_code, globals(), local_dict)
        self.compute_reward = local_dict['compute_dense_reward']
        
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = self.compute_reward(self.env, action)
        return obs, reward, done, info

def make_env(env_name, reward_file=None):
    """创建环境并加载自定义奖励函数"""
    env = gym.make(env_name)
    if reward_file is not None:
        env = RewardWrapper(env, reward_file)
    return env

def train(args):
    # 创建环境
    env = make_env(args.task, args.reward_file)
    eval_env = make_env(args.task, args.reward_file)
    
    # 创建模型
    model = SAC("MlpPolicy", env, verbose=1)
    
    # 训练
    model.learn(total_timesteps=args.max_iterations)
    
    # 评估
    n_eval_episodes = 10
    success_count = 0
    
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if info.get('success', False):
                success_count += 1
                break
    
    success_rate = success_count / n_eval_episodes
    print(f"Success rate: {success_rate:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True)
    parser.add_argument("--max_iterations", type=int, default=1000)
    
    args = parser.parse_args()
    train(args) 