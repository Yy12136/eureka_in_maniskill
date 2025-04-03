import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from FlagEmbedding import FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from Levenshtein import ratio as levenshtein_ratio
from collections import defaultdict
from typing import List


def calculate_edit_distance_matrix(strings: List[str]) -> np.ndarray:
    n = len(strings)
    matrix = np.zeros((n, n), dtype='float')
    
    for i in tqdm(range(n), desc="Calculating edit distance"):
        for j in range(i, n):
            dist = levenshtein_ratio(strings[i], strings[j])
            matrix[i, j] = dist
            matrix[j, i] = dist
    
    return matrix


def calculate_sample_similarity(all_code_string: List[str], task_path: str) -> None:
    start_time = time.time()
    methods = ["bge-m3", "tf-idf", "edit_dist", "e5"]
    sim_methods = ["cosine", "sharpen_cosine"]
    
    for method in methods:
        for sim_method in sim_methods:
            try:
                if method == "bge-m3":
                    model = FlagModel('/Dataset4/lyj/bge-m3', 
                                    use_fp16=True,
                                    device='cuda:0')
                    embeddings = model.encode(all_code_string, 
                                           batch_size=12,
                                           max_length=8192)
                
                elif method == 'e5':
                    model = SentenceTransformer('/Dataset4/lyj/multilingual-e5-large-instruct',
                                              device='cuda:0')
                    embeddings = model.encode(all_code_string, 
                                           convert_to_tensor=True,
                                           normalize_embeddings=True)
                    embeddings = embeddings.cpu().numpy()
                
                elif method == "tf-idf":
                    vectorizer = TfidfVectorizer(
                        max_features=1000,
                        ngram_range=(1, 2),
                    )
                    embeddings = vectorizer.fit_transform(all_code_string)
                
                elif method == "edit_dist":
                    similarity_matrix = calculate_edit_distance_matrix(all_code_string)
                    
                if method != "edit_dist":
                    if sim_method == "cosine":
                        similarity_matrix = cosine_similarity(embeddings)
                    elif sim_method == "sharpen_cosine":
                        similarity_matrix = cosine_similarity(embeddings)**3
                
                mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
                new_column = np.array([row[mask[i]].mean() for i, row in enumerate(similarity_matrix)])
                result_matrix = np.hstack([similarity_matrix, new_column.reshape(-1, 1)])
                
                df = pd.DataFrame(result_matrix).round(4)
                df.to_excel(
                    f"{task_path}/similarity_matrix_{method}_{sim_method}.xlsx",
                    index=False,
                    header=False
                )
                print(f"成功完成 {method} {sim_method} 方法")
                
            except Exception as e:
                print(f"警告: {method} {sim_method} 方法失败: {str(e)}")
                continue


def calculate_items_frequency(all_reward: List[str]):
    d = defaultdict(int)
    count = 0
    for reward in all_reward:
        for r in reward:
            d[r] += 1
            count += 1
    
    for key, value in list(d.items()):
        d[key] = round(value / count, 3)
    
    return dict(d)  # 返回频率字典而不是打印


if __name__ == '__main__':
    task_path = '/home/kxy/WHR/Eureka/temp/task/AllegroHand'
    with open(f'{task_path}/all_code_string.json', 'r', encoding='utf-8') as json_file:
        all_code_string = json.load(json_file)
    
    with open(f'{task_path}/all_reward.json', 'r', encoding='utf-8') as json_file:
        all_reward = json.load(json_file)

    # calculate_sample_similarity(all_code_string, task_path)
    
    calculate_items_frequency(all_reward)