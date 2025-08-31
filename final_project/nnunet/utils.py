# utils.py

import os
import json
import random
import logging
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
from sklearn.model_selection import KFold
import pandas as pd

import config

def set_seed(seed: int):
    """
    设置所有相关的随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    # 确保CUDA卷积操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to {seed}")

def create_patient_folds(patient_labels: Dict[str, int], num_folds: int, seed: int) -> List[List[str]]:
    """
    在病人层面（Patient-Level）上将数据集划分为K个互不相交的折。

    Args:
        patient_labels (Dict[str, int]): 包含所有病人ID (.nii文件名) 及其标签的字典。
        num_folds (int): 要划分的折数。
        seed (int): 用于KFold的随机种子，确保划分是可复现的。

    Returns:
        List[List[str]]: 一个列表，其中每个子列表包含该折的病人ID。
    """
    logging.info(f"Creating {num_folds} folds for {len(patient_labels)} patients...")
    # 确保病人ID是排序的，让KFold在不同机器上表现一致
    patient_ids = np.array(sorted(list(patient_labels.keys())))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    folds = []
    # kf.split返回的是训练集和测试集的索引
    # 我们用测试集的索引来定义每一折的内容
    for _, test_index in kf.split(patient_ids):
        fold_pids = patient_ids[test_index].tolist()
        folds.append(fold_pids)
    
    # 打印每折信息以供核对
    for i, fold_pids in enumerate(folds):
        logging.info(f"  - Fold {i+1} created with {len(fold_pids)} patients.")
        
    return folds

def setup_logging(fold_num: int) -> None:
    """
    为指定的折配置日志记录，同时输出到文件和控制台。
    """
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    log_file = os.path.join(fold_dir, f"experiment_log.log")
    
    # 清除之前可能存在的handlers，避免日志重复打印
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='a'), # 'a' for append mode
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured for Fold {fold_num}. Log file: {log_file}")


def summarize_results(all_fold_results: List[Dict[str, Any]]):
    """
    在所有折都完成后，计算并打印最终的性能总结。

    Args:
        all_fold_results (List[Dict[str, Any]]): 一个列表，包含每一折测试后返回的指标字典。
    """
    if not all_fold_results:
        logging.info("No results to summarize.")
        return

    # 将结果列表转换为Pandas DataFrame以便于计算
    results_df = pd.DataFrame(all_fold_results)
    
    # 计算平均值和标准差
    summary = results_df.agg(['mean', 'std']).round(4)
    
    # 使用根日志记录器，确保即使在循环外也能打印
    root_logger = logging.getLogger()
    
    # 清除之前的 handlers，只保留 StreamHandler 用于最终总结
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(logging.StreamHandler())
    
    root_logger.info("\n" + "="*60)
    root_logger.info(" K-Fold Cross-Validation Final Summary ".center(60, "="))
    root_logger.info("="*60)
    
    root_logger.info("\nIndividual Fold Test Results:")
    # 添加Fold列以方便查看
    results_df.index = [f"Fold {i+1}" for i in range(len(results_df))]
    root_logger.info("\n" + results_df.round(4).to_string())
    
    root_logger.info("\nAggregated Statistics (Mean ± Std Dev):")
    
    # 格式化输出
    for metric in summary.columns:
        mean_val = summary.loc['mean', metric]
        std_val = summary.loc['std', metric] if pd.notna(summary.loc['std', metric]) else 0.0
        root_logger.info(f"  - {metric:<15}: {mean_val:.4f} ± {std_val:.4f}")
        
    root_logger.info("\n" + "="*60)

    # 将详细结果和总结都保存到文件
    detailed_path = os.path.join(config.OUTPUT_DIR, "final_detailed_results.csv")
    summary_path = os.path.join(config.OUTPUT_DIR, "final_summary_stats.csv")
    
    results_df.to_csv(detailed_path, float_format='%.4f')
    summary.to_csv(summary_path, float_format='%.4f')
    
    root_logger.info(f"Detailed fold results saved to {detailed_path}")
    root_logger.info(f"Summary statistics saved to {summary_path}")

def load_pid_map() -> Dict[str, str]:
    """
    加载并返回 PID 到 Case ID 的映射字典。
    """
    try:
        with open(config.PID_TO_CASE_ID_MAP_PATH, 'r') as f:
            pid_map = json.load(f)
        # JSON加载的key是字符串，我们需要确保它们是干净的PID
        # 您的映射文件已经是 "pid": id 的形式，所以直接返回
        return pid_map
    except FileNotFoundError:
        logging.error(f"PID map file not found at: {config.PID_TO_CASE_ID_MAP_PATH}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from PID map file: {config.PID_TO_CASE_ID_MAP_PATH}")
        raise