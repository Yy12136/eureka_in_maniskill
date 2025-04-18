import os
import argparse
from pathlib import Path
from typing import Tuple
from langchain.prompts import PromptTemplate
from omegaconf import OmegaConf
from openai import OpenAI

from code_generation.eureka.run_eureka import run_eureka
from code_generation.single_flow.classlike_prompt.MobileDualArmPrompt import MOBILE_DUAL_ARM_PROMPT
from code_generation.single_flow.classlike_prompt.MobilePandaPrompt import MOBILE_PANDA_PROMPT
from code_generation.single_flow.classlike_prompt.PandaPrompt import PANDA_PROMPT

# 任务列表定义
franka_list = ["LiftCube-v0", "PickCube-v0", "StackCube-v0", "TurnFaucet-v0"]
mobile_list = ["OpenCabinetDoor-v1", "OpenCabinetDrawer-v1", "PushChair-v1"]
task_list = franka_list + mobile_list

LiftCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
    self.goal_height = 0.2 # in meters, indicate the z-axis height of our target
""".strip()

PickCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
    self.goal_position : np.ndarray[(3,)] # indicate the 3D position of our target position
""".strip()

StackCube_Env = """
    self.cubeA : RigidObject # cube A in the environment
    self.cubeB : RigidObject # cube B in the environment
    self.cube_half_size = 0.02  # in meters
    self.robot : PandaRobot # a Franka Panda robot
""".strip()

TurnFaucet_Env = """
    self.faucet : ArticulateObject # faucet in the environment
    self.faucet.handle : LinkObject # the handle of the faucet in the environment
    self.robot : PandaRobot # a Franka Panda robot
""".strip()


prompt_mapping = {
    "LiftCube-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", LiftCube_Env)),
    "PickCube-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", PickCube_Env)),
    "StackCube-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", StackCube_Env)),
    "TurnFaucet-v0": PromptTemplate(input_variables=["instruction"], template=PANDA_PROMPT.replace("<environment_description>", TurnFaucet_Env)),
    "OpenCabinetDoor-v1": MOBILE_PANDA_PROMPT,
    "OpenCabinetDrawer-v1": MOBILE_PANDA_PROMPT,
    "PushChair-v1": MOBILE_DUAL_ARM_PROMPT,
}

instruction_mapping = {
    "LiftCube-v0": "Pick up cube A and lift it up by 0.2 meter.",
    "PickCube-v0": "Pick up cube A and move it to the 3D goal position.",
    "StackCube-v0": "Pick up cube A and place it on cube B. The task is finished when cube A is on top of cube B stably (i.e. cube A is static) and isn't grasped by the gripper.",
    "TurnFaucet-v0": "Turn on a faucet by rotating its handle. The task is finished when qpos of faucet handle is larger than target qpos.",
    "OpenCabinetDoor-v1": "A single-arm mobile robot needs to open a cabinet door. The task is finished when qpos of cabinet door is larger than target qpos.",
    "OpenCabinetDrawer-v1": "A single-arm mobile robot needs to open a cabinet drawer. The task is finished when qpos of cabinet drawer is larger than target qpos.",
    "PushChair-v1": "A dual-arm mobile robot needs to push a swivel chair to a target location on the ground and prevent it from falling over.",
}

mapping_dicts_mapping = {
    "LiftCube-v0": {
        "self.cube_A.check_static()": "check_actor_static(self.obj)",
        "self.cube_A": "self.obj",
        "self.cubeA.check_static()": "check_actor_static(self.obj)",
        "self.cubeA" : "self.obj",
        "self.robot.ee_pose": "self.tcp.pose",
        "self.robot.check_grasp": "self.agent.check_grasp",
        "self.cube_half_size": "0.02",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.cube_A.is_grasped": "self.agent.check_grasp(self.obj)",
        "self.robot.is_grasping": "self.agent.check_grasp", 
    },
    "PickCube-v0": {
        "self.cube_A.check_static()": "check_actor_static(self.obj)",
        "self.cube_A": "self.obj",
        "self.cubeA.check_static()": "check_actor_static(self.obj)",
        "self.cubeA" : "self.obj",
        "self.robot.ee_pose" : "self.tcp.pose",
        "self.robot.check_grasp" : "self.agent.check_grasp",
        "self.goal_position" : "self.goal_pos",
        "self.cube_half_size" : "0.02",
        "self.robot.qpos" : "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel" : "self.agent.robot.get_qvel()[:-2]",
        "self.cube_A.is_grasped": "self.agent.check_grasp(self.obj)",
        "self.robot.is_grasping": "self.agent.check_grasp", 
    },
    "StackCube-v0": {
        "self.robot.ee_pose": "self.tcp.pose",
        "self.robot.check_grasp": "self.agent.check_grasp",
        "self.goal_position": "self.goal_pos",
        "self.cube_half_size": "0.02",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cubeA.check_static()": "check_actor_static(self.cubeA)",
        "self.cubeB.check_static()": "check_actor_static(self.cubeB)",
        "self.cube_A.check_static()": "check_actor_static(self.cubeA)",
        "self.cube_B.check_static()": "check_actor_static(self.cubeB)",
    },
    "TurnFaucet-v0": {
        "self.faucet.handle.target_qpos": "self.target_angle",
        "self.faucet.handle.qpos": "self.current_angle",
        "self.faucet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)",
        "self.robot.lfinger.get_world_pcd()": "transform_points(self.lfinger.pose.to_transformation_matrix(), self.lfinger_pcd)",
        "self.robot.rfinger.get_world_pcd()": "transform_points(self.rfinger.pose.to_transformation_matrix(), self.rfinger_pcd)",
        "self.faucet.handle.get_local_pcd()": "self.target_link_pcd",
        "self.robot.lfinger.get_local_pcd()": "self.lfinger_pcd",
        "self.robot.rfinger.get_local_pcd()": "self.rfinger_pcd",
        "self.robot.lfinger": "self.lfinger",
        "self.robot.rfinger": "self.rfinger",
        "self.faucet.handle": "self.target_link",
        "self.robot.ee_pose": "self.tcp.pose",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.check_grasp": "self.agent.check_grasp",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
    },
    "OpenCabinetDoor-v1": {
        "self.robot.ee_pose": "self.agent.hand.pose",
        "self.robot.base_position": "self.agent.base_pose.p[:2]",
        "self.robot.base_velocity": "self.agent.base_link.velocity[:2]",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cabinet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)",
        "self.cabinet.handle.get_local_pcd()": "self.target_handle_pcd",
        "self.cabinet.handle.local_sdf": "self.target_handle_sdf.signed_distance",
        "self.cabinet.handle.target_grasp_poses": "self.target_handles_grasp_poses[self.target_link_idx]",
        "self.cabinet.handle.qpos": "self.link_qpos",
        "self.cabinet.handle.qvel": "self.link_qvel",
        "self.cabinet.handle.target_qpos": "self.target_qpos",
        "self.cabinet.handle.check_static()": "self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1)",
        "self.cabinet.handle": "self.target_link",
        # "self.cabinet": "TODO",
    },
    "OpenCabinetDrawer-v1": {
        "self.robot.ee_pose": "self.agent.hand.pose",
        "self.robot.base_position": "self.agent.base_pose.p[:2]",
        "self.robot.base_velocity": "self.agent.base_link.velocity[:2]",
        "self.robot.qpos": "self.agent.robot.get_qpos()[:-2]",
        "self.robot.qvel": "self.agent.robot.get_qvel()[:-2]",
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.robot.gripper_openness": "self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]",
        "self.cabinet.handle.get_world_pcd()": "transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)",
        "self.cabinet.handle.get_local_pcd()": "self.target_handle_pcd",
        "self.cabinet.handle.local_sdf": "self.target_handle_sdf.signed_distance",
        "self.cabinet.handle.target_grasp_poses": "self.target_handles_grasp_poses[self.target_link_idx]",
        "self.cabinet.handle.qpos": "self.link_qpos",
        "self.cabinet.handle.qvel": "self.link_qvel",
        "self.cabinet.handle.target_qpos": "self.target_qpos",
        "self.cabinet.handle.check_static()": "self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1)",
        "self.cabinet.handle": "self.target_link",
        # "self.cabinet": "TODO",
    },
    "PushChair-v1": {
        "self.robot.get_ee_coords()": "self.agent.get_ee_coords()",
        "self.chair.get_pcd()": "self.env.env._get_chair_pcd()",
        "self.chair.check_static()": "self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2)",
        "self.chair": "self.root_link",
    },
}


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
            results_dir = Path("/home/yy/text2reward/results/maniskill_zero_shot") / task_name.lower()
            results_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
            
            with open(results_dir / "best_reward.py", "w") as f:
                f.write(best_code)
            print(f"\n任务 {task_name} 的最佳奖励函数已保存 (reward: {best_reward:.3f})")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config/eureka.yaml")
    eureka_cfg = OmegaConf.load(config_path)
    main(eureka_cfg)