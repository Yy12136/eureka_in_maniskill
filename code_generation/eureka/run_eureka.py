import os
# 设置环境变量，在导入其他库之前
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from pathlib import Path
import logging
import os
import openai
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from code_generation.eureka.utils.misc import set_freest_gpu, block_until_training
import subprocess
from code_generation.single_flow.zero_shot.generation import ZeroShotGenerator
from code_generation.single_flow.few_shot.generation import FewShotGenerator
import sys
import mani_skill2
import gym
from code_generation.single_flow.sim_benchmark import calculate_edit_distance_matrix, calculate_sample_similarity, calculate_items_frequency
import pandas as pd
from Levenshtein import ratio as levenshtein_ratio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# 在主函数开始处
os.makedirs("temp", exist_ok=True)

def evaluate_reward_function(task_name, reward_path, train_steps, eval_episodes):
    """评估奖励函数"""
    print("\n" + "="*50)
    print(f"开始评估任务: {task_name}")
    print(f"奖励函数路径: {reward_path}")
    print(f"训练步数: {train_steps}")
    print("="*50)

    if task_name in ["LiftCube-v0", "PickCube-v0"]:
        cmd = [
            "python", "-u",
            "/home/yy/text2reward/run_maniskill/ppo.py",
            "--env_id", task_name,
            "--train_num", "8",
            "--eval_num", "5",
            "--eval_freq", "12800",
            "--max_episode_steps", "100",
            "--rollout_steps", "3200",
            "--train_max_steps", str(train_steps),
            "--reward_path", os.path.abspath(reward_path)
        ]
    else:
        # 使用 SAC 评估
        cmd = [
            "python", "-u",
            "/home/yy/text2reward/run_maniskill/sac.py",
            "--env_id", task_name,
            "--train_num", "8",
            "--eval_num", "5",
            "--eval_freq", "16000",
            "--max_episode_steps", "200",
            "--train_max_steps", str(train_steps),
            "--reward_path", os.path.abspath(reward_path)
        ]
    
    print("\n执行命令:")
    print(" ".join(cmd))
    print("\n开始训练和评估...")
    
    # 使用 subprocess.PIPE 来捕获输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,  # 使用文本模式而不是二进制模式
        bufsize=1  # 行缓冲
    )

    success_rates = []  # 存储所有成功率
    
    # 实时打印输出并收集成功率
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()
            print(line)
            
            # 检查是否包含成功率信息
            if "Success rate:" in line:
                try:
                    # 处理 "Success rate: XX%" 格式
                    rate = float(line.split(":")[1].strip().rstrip('%')) / 100
                    success_rates.append(rate)
                except Exception as e:
                    print(f"解析成功率失败: {e}")

    # 等待进程结束
    return_code = process.wait()
    
    if return_code != 0:
        print(f"\n警告: 进程返回码 {return_code}")
        stderr = process.stderr.read()
        if stderr:
            print("错误输出:")
            print(stderr)
        return 0.0

    # 关闭管道
    process.stdout.close()
    process.stderr.close()

    # 返回最后几次评估的平均成功率
    if success_rates:
        last_n = min(5, len(success_rates))  # 取最后5次的平均
        avg_success_rate = np.mean(success_rates[-last_n:])
        print("\n最终评估结果:")
        print(f"所有成功率: {[f'{rate:.1%}' for rate in success_rates]}")
        print(f"最后 {last_n} 次平均成功率: {avg_success_rate:.1%}")
        return avg_success_rate
    else:
        print("\n警告: 未检测到任何成功率")
        return 0.0

def parse_training_results(stdout_str: str) -> float:
    """从训练输出中解析成功率"""
    try:
        for line in stdout_str.split('\n')[::-1]:
            if "Success rate:" in line:
                success_rate = float(line.split("Success rate:")[-1].strip())
                return success_rate
        return -float('inf')
    except Exception as e:
        logging.error(f"解析训练结果失败: {e}")
        return -float('inf')

def run_eureka(cfg, task_name, instruction, prompt_template, map_dict, generator=None):
    """运行 Eureka 算法"""
    print(f"\n{'='*50}")
    print(f"开始任务: {task_name}")
    print(f"计划生成 {cfg.sample} 个样本")
    print(f"迭代次数: {cfg.iteration}")
    print(f"{'='*50}\n")
    
    
    # 设置默认训练步数
    if not hasattr(cfg, 'train_max_steps'):
        if task_name in ["LiftCube-v0", "PickCube-v0"]:
            cfg.train_max_steps = 200000  # 20万步
        else:
            cfg.train_max_steps = 400000  # 40万步
    
    if generator is None:
        generator = ZeroShotGenerator(prompt_template)

    # 保持原来的路径结构
    if isinstance(generator, ZeroShotGenerator):
        base_dir = Path("results") / "maniskill_zeroshot" / task_name.lower()
        last_reward_filename = "last_reward_zeroshot.py"
    else:
        base_dir = Path("results") / "maniskill_fewshot" / task_name.lower()
        last_reward_filename = "last_reward_fewshot.py"
        
    # 确保基础目录存在
    base_dir = Path("/home/yy/text2reward") / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存目录: {base_dir}")
    
    best_code = None
    best_success_rate = -float('inf')
    
    # 记录历史最佳代码和成功率
    history_best_code = None
    history_best_success_rate = -float('inf')
    
    # 构建属性和方法的提示信息
    mapping_info = "\n注意：对于此任务，请只使用以下映射中定义的属性和方法：\n"
    for old_attr, new_attr in map_dict.items():
        mapping_info += f"- {old_attr} -> {new_attr}\n"
    
    # 更新提示模板的内容
    prompt_template.template = prompt_template.template + mapping_info
    
    def evaluate_samples(samples, iteration, task_name):
        """评估样本函数"""
        print(f"\n{'='*50}")
        print(f"迭代 {iteration} - 开始评估过程")
        print(f"总样本数: {len(samples)}")
        
        temp_dir = f"temp/iteration_{iteration}"
        os.makedirs(temp_dir, exist_ok=True)
        
        all_code_string = [sample['code'] for sample in samples]
        all_rewards = [sample['reward_items'] for sample in samples]
        
        print("\n1. 保存样本数据...")
        with open(f'{temp_dir}/all_code_string.json', 'w', encoding='utf-8') as f:
            json.dump(all_code_string, f)
        with open(f'{temp_dir}/all_reward.json', 'w', encoding='utf-8') as f:
            json.dump(all_rewards, f)
        
        print("\n2. 计算样本相似度...")
        try:
            print("使用 BGE 模型计算相似度...")
            model = FlagModel(
                model_name_or_path='/home/yy/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f',
                use_fp16=True
            )
            all_rewards_str = [' '.join(reward_list) if reward_list else 'empty_reward' for reward_list in all_rewards]
            embeddings = model.encode(all_rewards_str, batch_size=12)
            similarity_matrix = cosine_similarity(embeddings)
            print("BGE 相似度计算成功")
            
        except Exception as e:
            print(f"BGE 计算失败: {e}")
            try:
                print("使用 TF-IDF 计算相似度...")
                # 确保文本不为空且包含有效词汇
                all_rewards_str = [' '.join(reward_list) if reward_list else 'empty_reward' for reward_list in all_rewards]
                
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=1,  # 至少出现1次的词才会被保留
                    stop_words=None  # 不移除任何停用词
                )
                embeddings = vectorizer.fit_transform(all_rewards_str)
                similarity_matrix = cosine_similarity(embeddings)
                print("TF-IDF 计算成功")
                
            except Exception as e:
                print(f"TF-IDF 失败: {e}")
                print("使用编辑距离计算相似度...")
                similarity_matrix = calculate_edit_distance_matrix(all_rewards)
                print("编辑距离计算成功")
        
        # 不再保存到Excel文件，直接使用计算得到的相似度矩阵
        print("\n3. 选择样本进行评估...")
        n_samples = len(samples)
        n_eval = min(3, n_samples)
        
        # 直接使用已计算好的similarity_matrix选择最相似的样本
        top_indices = np.argsort(np.sum(similarity_matrix, axis=1))[-n_eval:]
        
        print(f"\n选择了 {len(top_indices)} 个最相似的样本进行评估:")
        
        # 修复频率统计
        print("\n奖励项频率统计：")
        selected_rewards = []
        for idx in top_indices:
            print(f"\n样本 {idx}:")
            print(f"代码内容:\n{samples[idx]['code']}")  # 打印完整代码
            reward_items = extract_reward_items(samples[idx]['code'])
            print(f"提取的奖励项: {reward_items}")  # 打印提取的奖励项
            selected_rewards.extend(reward_items)
        
        # 计算频率
        if selected_rewards:  # 确保有奖励项
            d = defaultdict(int)
            total_count = len(selected_rewards)
            
            # 统计每个奖励项出现的次数
            for reward_item in selected_rewards:
                d[reward_item] += 1
            
            # 计算频率并打印
            print("\n各奖励项出现频率：")
            for item, count in d.items():
                freq = round(count / total_count, 3)
                print(f"{item}: {freq:.3f}")
        else:
            print("\n警告：未找到任何奖励项！")
        
        # 继续原有的评估流程
        evaluated_results = []
        for i, sample in enumerate(samples):
            if i in top_indices:
                print(f"\n评估样本 {i}...")
                success_rate = evaluate_reward_function(
                    task_name=task_name,
                    reward_path=sample['reward_path'],
                    train_steps=cfg.train_max_steps,
                    eval_episodes=10
                )
                evaluated_results.append({
                    'success_rate': success_rate,
                    'evaluated': True
                })
                print(f"样本 {i} 评估完成，成功率: {success_rate:.3f}")
            else:
                evaluated_results.append({
                    'success_rate': 0,
                    'evaluated': False
                })
        
        return evaluated_results
    
    for iter_num in range(cfg.iteration):
        print(f"\n{'='*50}")
        print(f"迭代 {iter_num + 1}/{cfg.iteration}")
        
        # 如果有历史最佳代码，将其添加到提示中
        if history_best_code is not None:
            print("\n使用历史最佳代码作为示例...")
            instruction_with_example = f"""
            {instruction}
            
            这是一个表现良好的奖励函数示例（成功率: {history_best_success_rate:.2f}）：
            ```python
            {history_best_code}
            ```
            
            请参考这个示例，生成一个更好的奖励函数。你可以：
            1. 分析这个奖励函数的优点
            2. 找出可能的改进点
            3. 尝试不同的奖励项组合
            4. 调整奖励权重
            """
            current_instruction = instruction_with_example
        else:
            print("\n使用原始指令（无示例）...")
            current_instruction = instruction
        
        # 生成样本时使用当前指令
        sample_results = []
        for response_id in range(cfg.sample):
            print(f"\n生成样本 {response_id + 1}/{cfg.sample}")
            try:
                # 生成代码
                specific_code, general_code = generator.generate_code(
                    instruction=current_instruction,
                    map_dict=map_dict
                )
                
                if specific_code:
                    # 保存当前样本
                    iter_dir = base_dir / f"iter_{iter_num}" / f"sample_{response_id}"
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    
                    reward_path = iter_dir / "specific.py"
                    print(f"保存奖励函数到: {reward_path}")
                    
                    with open(reward_path, "w") as f:
                        f.write(specific_code)
                    
                    sample_results.append({
                        'code': specific_code,
                        'reward_path': str(reward_path),
                        'reward_items': extract_reward_items(specific_code)  # 提取奖励项
                    })
                    print(f"样本 {response_id + 1} 生成成功")
                    
            except Exception as e:
                print(f"样本 {response_id + 1} 生成失败: {e}")
                continue
        
        print(f"\n成功生成 {len(sample_results)}/{cfg.sample} 个样本")
        
        # 第二步：评估样本
        print("\n2. 开始评估样本...")
        evaluated_results = evaluate_samples(sample_results, iter_num, task_name)
        
        # 第三步：更新最佳结果
        print("\n3. 更新最佳结果...")
        for i, result in enumerate(evaluated_results):
            if result['evaluated'] and result['success_rate'] > best_success_rate:
                best_success_rate = result['success_rate']
                best_code = sample_results[i]['code']
                print(f"发现新的最佳结果！成功率: {best_success_rate:.3f}")
                
                # 保存最佳代码
                with open(base_dir / last_reward_filename, "w") as f:
                    f.write(best_code)
                print(f"已保存最佳奖励函数")
        
        print(f"\n当前迭代最佳成功率: {best_success_rate:.3f}")
        
        # 更新历史最佳
        if best_success_rate > history_best_success_rate:
            history_best_code = best_code
            history_best_success_rate = best_success_rate
            print(f"\n更新历史最佳记录！成功率: {history_best_success_rate:.3f}")
        
        print(f"{'='*50}\n")
    
    return best_code, best_success_rate

def file_to_string(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def extract_reward_items(code_string):
    """从代码中提取奖励项"""
    reward_items = []
    lines = code_string.split('\n')
    
    # 找到各个奖励组件的名称
    components = set()
    
    # 1. 从初始化语句中查找
    for line in lines:
        line = line.strip()
        if '=' in line and 'reward_' in line and '#' not in line:  # 排除注释行
            var_name = line.split('=')[0].strip()
            if var_name != 'reward' and 'weight_' not in var_name:  # 排除最终的reward和权重
                components.add(var_name)
    
    # 2. 从注释中查找（通常包含完整的奖励项说明）
    for line in lines:
        line = line.strip()
        if '#' in line and 'reward_' in line.lower():
            # 提取注释中的reward_开头的词
            words = line.split()
            for word in words:
                if word.startswith('reward_'):
                    components.add(word.strip(',.()[]'))
    
    # 3. 从reward计算语句中查找
    reward_expr = []
    in_reward_block = False
    for line in lines:
        line = line.strip()
        if 'reward = (' in line:  # 开始多行reward表达式
            in_reward_block = True
            reward_expr.append(line)
        elif in_reward_block and ')' not in line:  # 继续收集reward表达式
            reward_expr.append(line)
        elif in_reward_block and ')' in line:  # 结束reward表达式
            reward_expr.append(line)
            in_reward_block = False
    
    # 4. 检查每个组件是否在最终的奖励计算中使用
    reward_str = ' '.join(reward_expr)
    for component in components:
        if component in reward_str or f"weight_{component.replace('reward_', '')}" in reward_str:
            reward_items.append(component)
    
    return sorted(list(set(reward_items)))  # 去重并排序 