import argparse
import os
from pathlib import Path

from code_generation.single_flow.classlike_prompt import PANDA_PROMPT_FOR_FEW_SHOT, MOBILE_PANDA_PROMPT_FOR_FEW_SHOT, \
    MOBILE_DUAL_ARM_PROMPT_FOR_FEW_SHOT
from code_generation.single_flow.few_shot.generation import FewShotGenerator
from code_generation.single_flow.zero_shot_exp import instruction_mapping, mapping_dicts_mapping, LiftCube_Env, \
    PickCube_Env, StackCube_Env, TurnFaucet_Env
from code_generation.eureka.run_eureka import run_eureka

few_shot_prompt_mapping = {
    "LiftCube-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", LiftCube_Env),
    "PickCube-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", PickCube_Env),
    "StackCube-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", StackCube_Env),
    "TurnFaucet-v0": PANDA_PROMPT_FOR_FEW_SHOT.replace("<environment_description>", TurnFaucet_Env),
    "OpenCabinetDoor-v1": MOBILE_PANDA_PROMPT_FOR_FEW_SHOT,
    "OpenCabinetDrawer-v1": MOBILE_PANDA_PROMPT_FOR_FEW_SHOT,
    "PushChair-v1": MOBILE_DUAL_ARM_PROMPT_FOR_FEW_SHOT
}

gold_reward_path_mapping = {
    "LiftCube-v0": "lift_cube",
    "PickCube-v0": "pick_cube",
    "StackCube-v0": "stack_cube",
    "TurnFaucet-v0": "turn_faucet",
    "OpenCabinetDoor-v1": "open_cabinet_door",
    "OpenCabinetDrawer-v1": "open_cabinet_drawer",
    "PushChair-v1": "push_chair",
}

def load_all_examples(current_task: str, verbose=True):
    examples = []
    for task in few_shot_prompt_mapping.keys():
        if task == current_task:
            continue
        with open(os.path.join("../gold_reward_rewrite", gold_reward_path_mapping[task] + ".py"), "r") as f:
            instruction = instruction_mapping[task]
            reward_code = f.read()
            examples.append({"instruction": instruction, "reward_code": reward_code})
            if verbose:
                print("Load task: {}".format(task))
    return examples

class FewShotEureka:
    def __init__(self, cfg, task_name):
        self.cfg = cfg
        self.task_name = task_name
        self.examples = load_all_examples(task_name)
        
    def run(self):
        # 创建生成器实例，包含示例
        generator = FewShotGenerator(
            info_prompt=few_shot_prompt_mapping[self.task_name],
            examples=self.examples,
            k_examples=3  # 使用3个示例
        )
        
        # 运行 Eureka 算法
        best_code, best_reward = run_eureka(
            cfg=self.cfg,
            task_name=self.task_name,
            instruction=instruction_mapping[self.task_name],
            prompt_template=few_shot_prompt_mapping[self.task_name],
            map_dict=mapping_dicts_mapping[self.task_name],
            generator=generator  # 传入已配置的生成器
        )
        
        return best_code, best_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str, default="LiftCube-v0")
    parser.add_argument('--MODEL_NAME', type=str, default="gpt-4")
    parser.add_argument('--iteration', type=int, default=10)
    parser.add_argument('--sample', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=1000)
    
    args = parser.parse_args()
    
    # 创建配置对象
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    cfg = Config(
        iteration=args.iteration,
        sample=args.sample,
        max_iterations=args.max_iterations
    )
    
    # 创建并运行 FewShotEureka
    few_shot_eureka = FewShotEureka(cfg, args.TASK)
    best_code, best_reward = few_shot_eureka.run()
    
    if best_code:
        print("\n生成成功!")
    else:
        print("\n生成失败!")
