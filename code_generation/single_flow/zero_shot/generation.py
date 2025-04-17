import re
import time
import os
from typing import Any, List, Mapping, Optional, Tuple

import torch
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from openai import OpenAI
import numpy as np

from code_generation.post_process.post_process import RewardFunctionConverter

# 定义奖励函数模板
SPECIFIC_REWARD_TEMPLATE = """
Generate a specific reward function for the following task:
{instruction}

Important Note for ManiSkill Environment:
- Use object.pose.p to get position vector [x, y, z]
- Use object.pose.q to get rotation quaternion
- DO NOT use indexing directly on pose objects
- Choose between 2 to 5 different reward components
- The sum of all weights must equal 1.0
- Use "reward_" prefix for main reward components
- Optional penalties or bonuses should be added separately after main reward calculation

DO NOT USE:
- Complex numpy functions (apply, isclose, etc.)
- Quaternion operations
- Undefined methods or attributes

Here are possible reward component types (choose 2-5):
1. Task Progress:
   - Distance rewards: use reward_dist, reward_approach, reward_reach, reward_near
   - Grasp state: use reward_grasp, reward_grip, reward_hold, reward_contact
   - Lift progress: use reward_lift, reward_height, reward_elevation, reward_raise

2. Motion Quality:
   - Motion smoothness: use reward_smooth, reward_effort
   - Energy efficiency: use reward_energy, reward_effort
   - Path optimization: use reward_approach, reward_reach

3. Stability:
   - Object orientation: use reward_stable, reward_balance
   - Motion stability: use reward_steady, reward_smooth
   - Balance control: use reward_balance, reward_stable

4. Safety:
   - Collision handling: use reward_contact
   - Joint control: use reward_effort, reward_smooth
   - Motion limits: use reward_steady

5. Precision:
   - Position control: use reward_reach, reward_near
   - Orientation control: use reward_stable, reward_balance
   - Movement accuracy: use reward_smooth, reward_steady

Variable Mappings:
{mappings}

Example format:
```python
def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_1 = ...    # Choose a primary task weight
    weight_2 = ...    # Choose a secondary task weight
    weight_3 = ...    # Choose an additional weight if needed
    weight_4 = ...    # Optional additional weight
    weight_5 = ...    # Optional additional weight
    # Note: weight_1 + weight_2 + ... = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_1 = 0.0    # Main reward component
    reward_2 = 0.0    # Main reward component
    reward_3 = 0.0    # Main reward component
    reward_4 = 0.0    # Optional additional reward component
    reward_5 = 0.0    # Optional additional reward component
    # Calculate reward components
    reward_1 = ...    # Implement your chosen reward calculation
    reward_2 = ...    # Implement your chosen reward calculation
    reward_3 = ...    # Implement your chosen reward calculation
    reward_4 = ...    # Implement your chosen reward calculation（optional）
    reward_5 = ...    # Implement your chosen reward calculation（optional）
    
    # Combine main rewards
    reward = (
        weight_1 * reward_1 +
        weight_2 * reward_2 +
        weight_3 * reward_3 +
        weight_4 * reward_4 +    # Optional additional reward component
        weight_5 * reward_5    # Optional additional reward component
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    # 2. Penalty for large actions (regularization)
    
    return reward
```

Important:
1. Use the correct variable names as specified in the mappings
2. Use "reward_" prefix for main reward components
3. Keep penalties/bonuses separate from main reward components
4. Choose different combinations of components for each sample
5. Create unique reward structures based on the task
6. Avoid copying the exact same components for every sample
7. Use descriptive names for weights and rewards
8. Keep weights and reward names consistent
"""

GENERAL_REWARD_TEMPLATE = """
Generate a general reward function for robotic manipulation:
{instruction}

The reward function should:
1. Be sparse but meaningful
2. Focus on task completion
3. Be applicable to similar tasks
"""

class HuggingFaceLLM(LLM):
    def __init__(self, name: str, temperature: float = 0, **kwargs):
        super().__init__()
        self.name = name
        self.temperature = temperature
        
    @property
    def _llm_type(self) -> str:
        return self.name

    def __call__(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, **kwargs)

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        name_map = {
            "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
            "llama_2_70b": "meta-llama/Llama-2-70b-chat-hf",
            "deepseek-v3-241226": "deepseek-ai/deepseek-v3-241226"
        }
        assert self.name in name_map, f"Model name {self.name} not supported!"
        model = name_map[self.name]
        
        try:
            # 尝试以离线模式加载
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                local_files_only=True,
                trust_remote_code=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model in offline mode: {e}")
            print("Please make sure you have downloaded the model files locally.")
            raise

        pipe.tokenizer.pad_token_id = tokenizer.eos_token_id

        chat = [
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(chat, tokenize=False)

        raw_results = pipe(
            [prompt],
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
            batch_size=1
        )
        return raw_results[0][0]["generated_text"][len(prompt):]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"name": self.name, "temperature": self.temperature}


class ZeroShotGenerator:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="f03c5260-8425-465c-b6c8-c929568a7e60"
        )

    def generate_code(self, instruction: str, map_dict: dict) -> Tuple[str, str]:
        try:
            # 根据映射字典判断当前环境
            if "self.goal_height" in str(map_dict):
                # LiftCube-v0 环境
                available_vars = """
                - self.tcp.pose.p: end-effector position
                - self.obj.pose.p: object position
                - self.agent.check_grasp(self.obj): check if object is grasped
                - self.goal_height: target lifting height
                - self.agent.robot.get_qpos()[:-2]: robot joint positions
                - self.agent.robot.get_qvel()[:-2]: robot joint velocities
                """
            elif "self.goal_pos" in str(map_dict):
                # PickCube-v0 环境
                available_vars = """
                - self.tcp.pose.p: end-effector position
                - self.obj.pose.p: object position
                - self.agent.check_grasp(self.obj): check if object is grasped
                - self.goal_pos: target position
                - self.agent.robot.get_qpos()[:-2]: robot joint positions
                - self.agent.robot.get_qvel()[:-2]: robot joint velocities
                """
            else:
                # 其他环境
                available_vars = "\n".join([f"- {k}: {v}" for k, v in map_dict.items()])

            specific_prompt = SPECIFIC_REWARD_TEMPLATE.format(
                instruction=instruction,
                mappings=available_vars
            )
            
            # 使用不同温度生成多个响应
            temperatures = self.generate_temperatures(20)
            specific_code = None
            
            for temp in temperatures:
                try:
                    response = self.client.chat.completions.create(
                        model="deepseek-v3-241226",
                        messages=[{"role": "user", "content": specific_prompt}],
                        temperature=temp
                    )
                    code = response.choices[0].message.content
                    if "def compute_dense_reward" in code:
                        specific_code = code
                        break
                except Exception as e:
                    print(f"温度 {temp} 生成失败: {e}")
                    continue
            
            if specific_code is None:
                # 如果所有温度都失败，使用默认温度重试一次
                response = self.client.chat.completions.create(
                    model="deepseek-v3-241226",
                    messages=[{"role": "user", "content": specific_prompt}]
                )
                specific_code = response.choices[0].message.content
            
            general_prompt = GENERAL_REWARD_TEMPLATE.format(
                instruction=instruction
            )
            
            general_response = self.client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[{"role": "user", "content": general_prompt}]
            )
            general_code = general_response.choices[0].message.content
            
            # 提取实际的函数代码
            def extract_function(code: str) -> str:
                """从生成的文本中提取实际的函数代码"""
                # 只保留实际的 Python 代码
                code_template = """import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Your reward computation code here
    
    return reward"""
                
                # 查找 Python 代码块
                import re
                pattern = r"```python\n(.*?)```"
                match = re.search(pattern, code, re.DOTALL)
                if match:
                    function_code = match.group(1).strip()
                    # 如果代码中没有 import，添加必要的导入
                    if "import" not in function_code:
                        function_code = "import numpy as np\n\n" + function_code
                    return function_code
                return code_template  # 如果没找到代码块，返回模板
            
            # 处理生成的代码
            if specific_code:
                specific_code = extract_function(specific_code)
                
                # 添加代码转换
                converter = RewardFunctionConverter(map_dict)
                specific_code = converter.general_to_specific(specific_code)
            
            if general_code:
                general_code = extract_function(general_code)
            
            return specific_code, general_code
            
        except Exception as e:
            print(f"生成代码出错: {e}")
            return "", ""

    def generate_temperatures(self, num_samples: int, min_temp: float = 0.0, max_temp: float = 1.0) -> List[float]:
        """生成均匀分布的温度值列表
        
        Args:
            num_samples: 需要的温度样本数量
            min_temp: 最小温度值 (默认: 0.0)
            max_temp: 最大温度值 (默认: 1.0)
        
        Returns:
            temperatures: 均匀分布的温度值列表
        """
        if num_samples < 1:
            raise ValueError("样本数量必须大于0")
        
        if num_samples == 1:
            return [min_temp]
        
        # 使用linspace生成均匀分布的值
        temperatures = np.linspace(min_temp, max_temp, num_samples)
        # 转换为列表并保留4位小数
        temperatures = [round(float(t), 4) for t in temperatures]
        
        return temperatures