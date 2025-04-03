import re
import time
from typing import List, Dict, Tuple
from openai import OpenAI
from code_generation.post_process.post_process import RewardFunctionConverter

class FewShotGenerator:
    def __init__(self, info_prompt, examples: List[Dict] = None, k_examples: int = 3):
        self.info_prompt = info_prompt
        self.examples = examples if examples else []
        self.k_examples = min(k_examples, len(self.examples))
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="f03c5260-8425-465c-b6c8-c929568a7e60"
        )

    def generate_code(self, instruction: str, map_dict: dict) -> Tuple[str, str]:
        try:
            # 构建包含示例的提示
            examples_text = ""
            for i, example in enumerate(self.examples[:self.k_examples]):
                examples_text += f"\nExample {i+1}:\n"
                examples_text += f"Task: {example['instruction']}\n"
                examples_text += f"Solution:\n```python\n{example['reward_code']}\n```\n"

            # 生成特定的奖励函数
            specific_prompt = f"""
            Here are some examples of reward functions for similar tasks:
            {examples_text}
            
            Now, generate a specific reward function for the following task:
            {instruction}
            
            The reward function should:
            1. Be dense and informative
            2. Guide the robot through each step
            3. Include specific milestones for the task
            
            Available variables:
            {self.info_prompt}
            """
            
            specific_response = self.client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[{"role": "user", "content": specific_prompt}]
            )
            specific_code = specific_response.choices[0].message.content
            
            # 生成通用的奖励函数
            general_prompt = f"""
            Based on the previous examples:
            {examples_text}
            
            Generate a general reward function for robotic manipulation:
            {instruction}
            
            The reward function should:
            1. Be sparse but meaningful
            2. Focus on task completion
            3. Be applicable to similar tasks
            """
            
            general_response = self.client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[{"role": "user", "content": general_prompt}]
            )
            general_code = general_response.choices[0].message.content
            
            # 提取实际的函数代码
            def extract_function(code: str) -> str:
                """从生成的文本中提取实际的函数代码"""
                code_template = """import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Your reward computation code here
    
    return reward"""
                
                pattern = r"```python\n(.*?)```"
                match = re.search(pattern, code, re.DOTALL)
                if match:
                    function_code = match.group(1).strip()
                    if "import" not in function_code:
                        function_code = "import numpy as np\n\n" + function_code
                    return function_code
                return code_template
                
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