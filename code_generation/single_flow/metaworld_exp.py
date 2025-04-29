import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from code_generation.single_flow.zero_shot.generation import ZeroShotGenerator
from code_generation.single_flow.classlike_prompt.MetaworldPrompt import METAWORLD_PROMPT
from code_generation.eureka.run_eureka_meta import run_eureka

# 任务列表
task_list = [
    "drawer-open-v2", "drawer-close-v2", 
    "window-open-v2", "window-close-v2", 
    "button-press-v2", "sweep-into-v2", 
    "door-unlock-v2", "door-close-v2", 
    "handle-press-v2", "handle-press-side-v2"
]

# 保持原有的映射
instruction_mapping = {
    "window-open-v2": "Push and open a sliding window by its handle.",
    "window-close-v2": "Push and close a sliding window by its handle.",
    "door-close-v2": "Close a door with a revolving joint by pushing door's handle.",
    "drawer-open-v2": "Open a drawer by its handle.",
    "drawer-close-v2": "Close a drawer by its handle.",
    "door-unlock-v2": "Unlock the door by rotating the lock counter-clockwise.",
    "sweep-into-v2": " Sweep a puck from the initial position into a hole.",
    "button-press-v2": "Press a button in y coordination.",
    "handle-press-v2": "Press a handle down.",
    "handle-press-side-v2": "Press a handle down sideways.",
}

mapping_dicts = {
    "self.robot.ee_position": "obs[:3]",
    "self.robot.gripper_openness": "obs[3]",
    "self.obj1.position": "obs[4:7]",
    "self.obj1.quaternion": "obs[7:11]",
    "self.obj2.position": "obs[11:14]",
    "self.obj2.quaternion": "obs[14:18]",
    "self.goal_position": "self.env._get_pos_goal()",
}

mapping_dicts_mapping = {task: mapping_dicts for task in task_list}
prompt_mapping = {task: METAWORLD_PROMPT for task in task_list}

def main(eureka_cfg):
    print("配置信息:")
    print(f"- 总迭代次数: {eureka_cfg.iteration}")
    print(f"- 每次迭代的样本数: {eureka_cfg.sample}\n")
    
    for task_name in task_list:
        # 运行 Eureka 算法
        best_code, best_reward = run_eureka(
            cfg=eureka_cfg,
            task_name=task_name,
            instruction=instruction_mapping[task_name],
            prompt_template=prompt_mapping[task_name],
            map_dict=mapping_dicts_mapping[task_name]
        )
        
        # 保存最佳结果
        if best_code is not None:
            results_dir = Path("/home/yy/text2reward/results/metaworld_zeroshot") / task_name.lower()
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / "best_reward.py", "w") as f:
                f.write(best_code)
            print(f"\n任务 {task_name} 的最佳奖励函数已保存 (reward: {best_reward:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str, default='drawer-open-v2', 
                      choices=task_list,
                      help='任务名称')
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), "config/eureka.yaml")
    eureka_cfg = OmegaConf.load(config_path)
    
    # 修改main函数只运行指定任务
    task_list = [args.TASK]  # 覆盖原有的task_list
    main(eureka_cfg)
