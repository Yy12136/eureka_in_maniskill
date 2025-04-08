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

def get_reward_mapping():
    """定义奖励项的同义词映射"""
    return {
        # 距离/接近相关
        'reward_distance': ['reward_dist', 'reward_approach', 'reward_reach', 'reward_near'],
        # 抓取相关
        'reward_grasp': ['reward_grip', 'reward_hold', 'reward_contact'],
        # 提升相关
        'reward_lift': ['reward_height', 'reward_elevation', 'reward_raise', 'reward_up'],
        # 稳定性相关
        'reward_stability': ['reward_stable', 'reward_balance', 'reward_steady'],
        # 控制相关
        'reward_control': ['reward_effort', 'reward_energy', 'reward_smooth'],
    }

def normalize_reward_items(reward_items):
    """标准化奖励项列表"""
    mapping = get_reward_mapping()
    normalized = []
    for item in reward_items:
        found = False
        for standard_name, variants in mapping.items():
            if item in variants or item == standard_name:
                normalized.append(standard_name)
                found = True
                break
        if not found:
            normalized.append(item)
    return sorted(normalized)

def find_similar_groups(samples, similarity_matrix, threshold=0.95):
    """基于相似度矩阵找出相似的样本组"""
    n_samples = len(samples)
    similar_groups = []
    used_indices = set()

    for i in range(n_samples):
        if i in used_indices:
            continue
        
        # 检查奖励项是否完全相同（考虑同义词）
        current_items = normalize_reward_items(samples[i]['reward_items'])
        current_group = {i}
        
        for j in range(i + 1, n_samples):
            if j in used_indices:
                continue
            other_items = normalize_reward_items(samples[j]['reward_items'])
            
            # 如果标准化后的奖励项完全相同
            if current_items == other_items:
                current_group.add(j)
        
        if len(current_group) > 1:  # 只保存有相似样本的组
            similar_groups.append(current_group)
            used_indices.update(current_group)
    
    return similar_groups


def calculate_reward_frequencies(samples):
    """计算所有样本中每个奖励项的出现频率"""
    total_samples = len(samples)
    reward_counts = defaultdict(int)
    
    # 统计每个奖励项在多少个样本中出现
    for sample in samples:
        reward_items = set(sample['reward_items'])  # 使用集合去重
        for item in reward_items:
            reward_counts[item] += 1
    
    # 计算频率并排序
    frequencies = {
        item: count/total_samples 
        for item, count in reward_counts.items()
    }
    
    # 按频率降序排序
    return dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))

def analyze_evaluation_results(samples, evaluated_results):
    """分析评估结果，生成反馈"""
    feedback = []
    
    # 按成功率排序
    sorted_indices = sorted(
        range(len(evaluated_results)), 
        key=lambda i: evaluated_results[i]['success_rate'] if evaluated_results[i]['evaluated'] else -1,
        reverse=True
    )
    
    # 分析表现最好的样本
    best_idx = sorted_indices[0]
    best_result = evaluated_results[best_idx]
    best_items = samples[best_idx]['reward_items']
    feedback.append(f"最佳表现的奖励组合: {best_items}")
    feedback.append(f"- 成功率: {best_result['success_rate']:.3f}")
    feedback.append(f"- 平均回报: {best_result['mean_reward']:.3f}")
    feedback.append(f"- 策略损失: {best_result['policy_loss']:.3f}")
    feedback.append(f"- 值函数损失: {best_result['value_loss']:.3f}")
    
    # 分析表现最差的样本
    worst_idx = sorted_indices[-1]
    worst_result = evaluated_results[worst_idx]
    worst_items = samples[worst_idx]['reward_items']
    feedback.append(f"\n最差表现的奖励组合: {worst_items}")
    feedback.append(f"- 成功率: {worst_result['success_rate']:.3f}")
    feedback.append(f"- 平均回报: {worst_result['mean_reward']:.3f}")
    feedback.append(f"- 策略损失: {worst_result['policy_loss']:.3f}")
    feedback.append(f"- 值函数损失: {worst_result['value_loss']:.3f}")
    
    return "\n".join(feedback)

def perform_ablation_analysis(samples, evaluated_results):
    """执行消融分析"""
    # 统计每个奖励项的效果
    item_performance = defaultdict(lambda: defaultdict(list))
    
    # 收集每个奖励项在不同组合中的表现
    for i, result in enumerate(evaluated_results):
        if not result['evaluated']:
            continue
        
        reward_items = samples[i]['reward_items']
        metrics = {
            'success_rate': result['success_rate'],
            'mean_reward': result['mean_reward'],
            'policy_loss': result['policy_loss'],
            'value_loss': result['value_loss']
        }
        
        for item in reward_items:
            for metric_name, value in metrics.items():
                item_performance[item][metric_name].append(value)
    
    # 分析每个奖励项
    analysis = []
    for item, metrics in item_performance.items():
        # 计算各项指标的统计数据
        success_stats = {
            'mean': np.mean(metrics['success_rate']),
            'max': np.max(metrics['success_rate']),
            'min': np.min(metrics['success_rate'])
        }
        reward_stats = {
            'mean': np.mean(metrics['mean_reward']),
            'max': np.max(metrics['mean_reward']),
            'min': np.min(metrics['mean_reward'])
        }
        
        # 综合评估该奖励项
        is_effective = (
            success_stats['mean'] > 0.5 and  # 成功率阈值
            reward_stats['mean'] > 0 and     # 平均回报为正
            np.mean(metrics['policy_loss']) < 1.0  # 策略损失阈值
        )
        
        if is_effective:
            analysis.append(f"建议保留 {item}:")
        else:
            analysis.append(f"建议考虑移除 {item}:")
            
        analysis.append(f"- 成功率: 平均 {success_stats['mean']:.3f}, 最高 {success_stats['max']:.3f}, 最低 {success_stats['min']:.3f}")
        analysis.append(f"- 平均回报: 平均 {reward_stats['mean']:.3f}, 最高 {reward_stats['max']:.3f}, 最低 {reward_stats['min']:.3f}")
        analysis.append(f"- 策略损失: {np.mean(metrics['policy_loss']):.3f}")
        analysis.append(f"- 值函数损失: {np.mean(metrics['value_loss']):.3f}\n")
    
    return "\n".join(analysis)

def generate_next_iteration_prompt(samples, evaluated_results, score_std, score_range):
    """生成下一次迭代的提示"""
    # 生成评估反馈
    prompt_e = analyze_evaluation_results(samples, evaluated_results)
    
    # 生成消融分析
    prompt_g = perform_ablation_analysis(samples, evaluated_results)
    
    # 根据得分差异度生成探索建议
    exploration_advice = ""
    if score_std > 0.15 or score_range > 0.3:  # 样本得分差异大
        exploration_advice = """
探索建议：
1. 尝试更多新颖的奖励项组合
2. 探索未充分验证的奖励项
3. 可以考虑更激进的组合方式
4. 尝试调整现有奖励项的权重
"""
    else:  # 中等差异
        exploration_advice = """
探索建议：
1. 在现有效果好的组合基础上小幅调整
2. 可以尝试添加或移除个别奖励项
3. 微调奖励项的权重
"""
    
    # 组合提示
    combined_prompt = f"""
基于当前迭代的评估结果：

1. 评估反馈：
{prompt_e}

2. 消融分析：
{prompt_g}

3. 探索建议：
{exploration_advice}
"""
    
    return combined_prompt

def calculate_sample_usefulness_scores(samples, reward_frequencies):
    """计算每个样本的有用性得分"""
    # 计算所有样本的得分
    sample_scores = []
    total_score = 0
    
    for i, sample in enumerate(samples):
        # 计算当前样本的奖励项频率总和
        score = sum(reward_frequencies[item] for item in sample['reward_items'])
        total_score += score
        sample_scores.append({
            'index': i,
            'raw_score': score,
            'items': sample['reward_items']
        })
    
    # 计算归一化得分
    for score in sample_scores:
        score['usefulness_score'] = score['raw_score'] / total_score
    
    # 按得分从小到大排序
    sorted_scores = sorted(sample_scores, key=lambda x: x['usefulness_score'])
    
    # 计算得分差异度
    scores = [s['usefulness_score'] for s in sorted_scores]
    score_std = np.std(scores)  # 标准差
    score_range = max(scores) - min(scores)  # 范围
    
    return sorted_scores, score_std, score_range

def evaluate_samples(samples, iteration, task_name, train_max_steps):
    """评估样本函数"""
    print(f"\n{'='*50}")
    print(f"迭代 {iteration} - 开始评估过程")
    print(f"总样本数: {len(samples)}")
    
    # 计算并保存奖励项频率
    reward_frequencies = calculate_reward_frequencies(samples)
    
    # 计算样本有用性得分
    usefulness_scores, score_std, score_range = calculate_sample_usefulness_scores(samples, reward_frequencies)
    
    # 创建迭代目录
    iter_dir = f"/home/yy/text2reward/results/iteration_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)
    
    # 保存频率统计和有用性得分
    with open(f'{iter_dir}/reward_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== 奖励项频率统计 ===\n")
        for item, freq in reward_frequencies.items():
            f.write(f"{item}: {freq:.3f}\n")
        
        f.write("\n=== 样本有用性得分（由小到大排序）===\n")
        for score in usefulness_scores:
            f.write(f"样本 {score['index']}:\n")
            f.write(f"奖励项: {score['items']}\n")
            f.write(f"有用性得分: {score['usefulness_score']:.3f}\n")
        
        f.write(f"\n得分统计：\n")
        f.write(f"标准差: {score_std:.3f}\n")
        f.write(f"范围: {score_range:.3f}\n")
    
    # 在终端打印奖励项频率统计
    print("\n=== 奖励项频率统计 ===")
    print("格式：奖励项: 频率 (出现次数/总样本数)")
    total_samples = len(samples)
    reward_counts = defaultdict(int)
    
    # 统计每个奖励项在多少个样本中出现
    for sample in samples:
        reward_items = set(sample['reward_items'])  # 使用集合去重
        for item in reward_items:
            reward_counts[item] += 1
    
    # 按频率降序排序并打印
    sorted_frequencies = sorted(
        [(item, count/total_samples, count) 
         for item, count in reward_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    for item, freq, count in sorted_frequencies:
        print(f"{item}: {freq:.3f} ({count}/{total_samples})")
    
    # 打印频率和得分信息
    print("\n样本有用性得分（由小到大）：")
    for score in usefulness_scores:
        print(f"样本 {score['index']}: {score['usefulness_score']:.3f}")
    
    # 提取所有样本的奖励项列表
    all_rewards = [sample['reward_items'] for sample in samples]
    
    # 计算相似度矩阵
    try:
        print("\n1. 使用 BGE 模型计算相似度...")
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
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                stop_words=None
            )
            embeddings = vectorizer.fit_transform(all_rewards_str)
            similarity_matrix = cosine_similarity(embeddings)
            print("TF-IDF 计算成功")
        except Exception as e:
            print(f"TF-IDF 失败: {e}")
            print("使用编辑距离计算相似度...")
            similarity_matrix = calculate_edit_distance_matrix(all_rewards)
            print("编辑距离计算成功")
    
    # 找出相似组（只考虑完全相同的情况）
    similar_groups = find_similar_groups(samples, similarity_matrix)
    
    print("\n2. 完全相同的奖励项组：")
    for group in similar_groups:
        print(f"\n相同组：{sorted(list(group))}")
        normalized_items = normalize_reward_items(samples[list(group)[0]]['reward_items'])
        print(f"标准化后的奖励项: {normalized_items}")
        for idx in group:
            print(f"样本 {idx} 原始奖励项: {samples[idx]['reward_items']}")
    
    # 修改样本选择逻辑，考虑有用性得分
    selected_samples = set()
    
    # 从每个相同组中选择得分最低的样本
    for group in similar_groups:
        group_scores = [s for s in usefulness_scores if s['index'] in group]
        selected_samples.add(min(group_scores, key=lambda x: x['usefulness_score'])['index'])
    
    # 添加所有未分组的样本
    all_grouped = set().union(*similar_groups) if similar_groups else set()
    remaining_samples = set(range(len(samples))) - all_grouped
    selected_samples.update(remaining_samples)
    
    # 按有用性得分排序进行评估
    selected_samples = sorted(selected_samples, 
                            key=lambda idx: next(s['usefulness_score'] 
                                               for s in usefulness_scores if s['index'] == idx))
    
    print(f"\n选择 {len(selected_samples)} 个样本进行评估（按有用性得分排序）：")
    for idx in selected_samples:
        score = next(s['usefulness_score'] for s in usefulness_scores if s['index'] == idx)
        print(f"样本 {idx}: {samples[idx]['reward_items']}, 得分: {score:.3f}")
    
    # 评估选中的样本
    evaluated_results = []
    for i in range(len(samples)):
        if i in selected_samples:
            print(f"\n评估样本 {i}...")
            success_rate = evaluate_reward_function(
                task_name=task_name,
                reward_path=samples[i]['reward_path'],
                train_steps=train_max_steps,
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
    
    # 将得分信息添加到返回结果中
    evaluation_result = {
        'results': evaluated_results,
        'score_std': score_std,
        'score_range': score_range,
        'usefulness_scores': usefulness_scores
    }
    
    # 生成下一次迭代的提示
    next_iteration_prompt = generate_next_iteration_prompt(
        samples, evaluated_results, score_std, score_range)
    
    return evaluation_result, next_iteration_prompt

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
    history_best_code = None
    history_best_success_rate = -float('inf')
    current_instruction = instruction  # 初始指令
    
    for iter_num in range(cfg.iteration):
        print(f"\n{'='*50}")
        print(f"迭代 {iter_num + 1}/{cfg.iteration}")
        
        # 生成样本时使用当前指令
        sample_results = []
        for response_id in range(cfg.sample):
            print(f"\n生成样本 {response_id + 1}/{cfg.sample}")
            try:
                # 生成代码
                specific_code, general_code = generator.generate_code(
                    instruction=current_instruction,  # 使用更新后的指令
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
                        'reward_items': extract_reward_items(specific_code)
                    })
                    print(f"样本 {response_id + 1} 生成成功")
            except Exception as e:
                print(f"样本 {response_id + 1} 生成失败: {e}")
                continue
        
        # 评估样本
        print("\n2. 开始评估样本...")
        evaluation_result, next_iteration_prompt = evaluate_samples(
            sample_results, iter_num, task_name, cfg.train_max_steps)
        
        # 更新最佳结果
        print("\n3. 更新最佳结果...")
        for i, result in enumerate(evaluation_result['results']):
            if result['evaluated'] and result['success_rate'] > best_success_rate:
                best_success_rate = result['success_rate']
                best_code = sample_results[i]['code']
                print(f"发现新的最佳结果！成功率: {best_success_rate:.3f}")
                
                # 保存最佳代码
                with open(base_dir / last_reward_filename, "w") as f:
                    f.write(best_code)
                print(f"已保存最佳奖励函数")
        
        # 检查是否需要停止迭代
        if evaluation_result['score_std'] < 0.05 and evaluation_result['score_range'] < 0.1:
            print("\n检测到样本得分收敛（标准差<0.05，范围<0.1）")
            print("原因：")
            print("1. 所有样本的有用性得分非常接近")
            print("2. 已经收敛到稳定的奖励组合")
            print("3. 进一步探索可能收益有限")
            print("\n提前结束迭代")
            break
        
        # 更新历史最佳
        if best_success_rate > history_best_success_rate:
            history_best_code = best_code
            history_best_success_rate = best_success_rate
            print(f"\n更新历史最佳记录！成功率: {history_best_success_rate:.3f}")
        
        # 更新下一次迭代的指令
        if history_best_code is not None:
            current_instruction = f"""
            {instruction}
            
            这是一个表现良好的奖励函数示例（成功率: {history_best_success_rate:.2f}）：
            ```python
            {history_best_code}
            ```
            
            基于上一次迭代的分析：
            {next_iteration_prompt}
            
            请参考这些信息，生成一个更好的奖励函数。
            """
        else:
            current_instruction = f"""
            {instruction}
            
            基于上一次迭代的分析：
            {next_iteration_prompt}
            
            请参考这些信息，生成一个好的奖励函数。
            """
        
        print("\n已更新下一次迭代的指令")
        
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