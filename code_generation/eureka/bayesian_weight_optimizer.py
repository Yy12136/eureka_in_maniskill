import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import os


class BayesianWeightOptimizer:
    def __init__(self, reward_frequencies, initial_weights):
        """
        初始化贝叶斯优化器
        reward_frequencies: dict, 奖励项的频率统计
        initial_weights: dict, 初始权重值
        """
        self.weights_names = list(initial_weights.keys())
        self.n_weights = len(self.weights_names)
        self.reward_frequencies = reward_frequencies
        
        # 修改特征长度计算：频率越高，length scale 越小
        self.length_scales = []
        for name in self.weights_names:
            reward_name = name.replace('weight_', 'reward_')
            freq = reward_frequencies.get(reward_name, 0.1)
            # 使用 (1 - freq) 作为 length scale，确保在 (0, 1) 范围内
            self.length_scales.append(1.0 - min(freq, 0.95))  # 防止 length scale 太小
            
        # 确保bounds和length_scales的维度匹配
        self.length_scales = np.array(self.length_scales)
        self.bounds = [(0.05, 0.8)] * self.n_weights
        
        # 存储观测数据
        self.X = []  # 权重组合
        self.y = []  # 对应的评估结果
        
        # 将初始权重转换为numpy数组
        self.initial_weights = np.array([initial_weights[name] for name in self.weights_names])
        
    def normalize_weights(self, weights):
        """确保权重和为1"""
        return weights / np.sum(weights)
        
    def kernel(self, p1, p2):
        """各向异性 Matérn 5/2 核函数"""
        d = 0
        for i in range(self.n_weights):
            d += ((p1[i] - p2[i]) / self.length_scales[i]) ** 2
        d = np.sqrt(d)
        
        k = (1 + np.sqrt(5) * d + 5 * d**2 / 3) * np.exp(-np.sqrt(5) * d)
        return k
    
    def acquisition_function(self, x, X, y, beta=2.0):
        """Upper Confidence Bound (UCB) 采集函数"""
        if len(X) == 0:
            return 1.0
            
        # 计算均值和方差
        K = np.array([[self.kernel(x1, x2) for x2 in X] for x1 in X])
        K = K + 1e-6 * np.eye(len(X))
        
        k = np.array([self.kernel(x, x2) for x2 in X])
        K_inv = np.linalg.inv(K)
        
        mu = k.T @ K_inv @ y
        sigma2 = self.kernel(x, x) - k.T @ K_inv @ k
        
        if sigma2 <= 0:
            return 0
            
        sigma = np.sqrt(sigma2)
        
        # UCB = μ + β * σ
        ucb = mu + beta * sigma
        return ucb
    
    def optimize(self, current_score=None, previous_weights=None, beta=2.0, exploration_weight=0.1, 
                extra_metrics=None):
        """优化权重
        Args:
            current_score: 当前权重的评估分数
            previous_weights: 上一次的权重值
            beta: UCB采集函数的探索参数
            exploration_weight: 探索权重，控制新权重与旧权重的差异程度
            extra_metrics: 额外的评估指标，字典格式
                {
                    'avg_reward': float,  # 平均奖励
                    'loss': float,        # 策略损失
                    'value_loss': float   # 值函数损失
                }
        """
        if current_score is not None:
            # 使用previous_weights而不是initial_weights
            current_weights = np.array(list(previous_weights.values())) if previous_weights else self.initial_weights
            
            # 确保权重维度正确
            if len(current_weights) != self.n_weights:
                print(f"警告：当前权重维度({len(current_weights)})与期望维度({self.n_weights})不匹配")
                current_weights = self._adjust_weights_dimension(current_weights)
            
            # 存储观测数据和额外指标
            self.X.append(current_weights.tolist())
            self.y.append(current_score)
            
            # 存储额外指标
            if extra_metrics:
                if not hasattr(self, 'extra_metrics_history'):
                    self.extra_metrics_history = []
                self.extra_metrics_history.append(extra_metrics)
        
        # 确保数据维度正确
        X = np.array(self.X, dtype=np.float64)
        y = np.array(self.y)
        
        # 优化采集函数
        def objective(x):
            x_norm = self.normalize_weights(x)
            acquisition_value = -self.acquisition_function(x_norm, X, y, beta=beta)
            
            # 添加探索项，鼓励与上一次权重的差异
            if previous_weights is not None:
                prev_weights_array = np.array(list(previous_weights.values()))
                exploration_term = -exploration_weight * np.sum((x - prev_weights_array)**2)
                acquisition_value += exploration_term
                
                # 如果有额外指标历史，使用它们来调整目标函数
                if hasattr(self, 'extra_metrics_history') and self.extra_metrics_history:
                    last_metrics = self.extra_metrics_history[-1]
                    # 根据损失值调整探索
                    if last_metrics['loss'] > 0:
                        loss_penalty = last_metrics['loss'] * 0.1
                        acquisition_value -= loss_penalty
                    # 根据值函数损失调整
                    if last_metrics['value_loss'] > 0:
                        value_loss_penalty = last_metrics['value_loss'] * 0.1
                        acquisition_value -= value_loss_penalty
            
            return acquisition_value
        
        # 使用当前权重作为初始值
        x0 = np.array(self.X[-1]) if len(self.X) > 0 else self.initial_weights
        
        res = minimize(
            objective,
            x0=x0,
            bounds=self.bounds,
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            method='SLSQP'
        )
        
        next_weights = {name: value for name, value in zip(self.weights_names, res.x)}
        acquisition_values = -res.fun
        
        # 归一化权重
        normalized_weights = self.normalize_weights_dict(next_weights)
        
        return normalized_weights, acquisition_values

    def _adjust_weights_dimension(self, weights):
        """调整权重维度的辅助函数"""
        if len(weights) > self.n_weights:
            return weights[:self.n_weights]
        else:
            temp = np.ones(self.n_weights) * 0.1
            temp[:len(weights)] = weights
            return temp

    def normalize_weights_dict(self, weights):
        """归一化权重字典，确保和为1且保持最小值"""
        # 确保最小值
        weights = {name: max(0.05, val) for name, val in weights.items()}
        # 归一化
        weights_sum = sum(weights.values())
        return {name: val/weights_sum for name, val in weights.items()}

    def initialize_bounds(self, n_dims):
        """根据权重维度初始化边界"""
        self.bounds = [(0.05, 0.8) for _ in range(n_dims)]  # 使用与__init__中相同的边界 