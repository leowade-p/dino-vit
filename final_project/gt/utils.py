# utils.py

import os
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Union

import numpy as np
import torch
from sklearn.model_selection import KFold
import pandas as pd
import json
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
    patient_ids = np.array(sorted(list(patient_labels.keys())))
    
    # KFold提供的是索引，我们需要用它来获取病人ID
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    folds = []
    for i, (train_index, val_index) in enumerate(kf.split(patient_ids)):
        # kf.split返回的是测试集的索引，我们这里把它当作每一折的病人列表
        fold_pids = patient_ids[val_index].tolist()
        folds.append(fold_pids)
        logging.info(f"  - Fold {i+1} created with {len(fold_pids)} patients.")
        
    return folds

def get_image_files_for_pids(image_dirs: List[str], pids: List[str]) -> List[str]:
    """
    根据给定的病人ID列表，从指定的图片目录中查找所有对应的图片文件。

    Args:
        image_dirs (List[str]): 包含所有图片(.png)的目录列表。
        pids (List[str]): 病人ID列表 (例如, ['1859176.nii', '1860366.nii'])。

    Returns:
        List[str]: 找到的所有图片文件的绝对路径列表。
    """
    # 从 '1859176.nii' 中提取 '1859176' 作为前缀
    pid_prefixes = {pid.split('.')[0] for pid in pids}
    
    image_files = []
    for directory in image_dirs:
        if not os.path.isdir(directory):
            logging.warning(f"Image directory not found: {directory}. Skipping.")
            continue
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                file_prefix = filename.split('_')[0]
                if file_prefix in pid_prefixes:
                    image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)

def setup_logging(fold_num: int) -> None:
    """
    为指定的折配置日志记录，同时输出到文件和控制台。
    """
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    log_file = os.path.join(fold_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 清除之前可能存在的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
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
    
    logging.info("\n" + "="*50)
    logging.info(" K-Fold Cross-Validation Final Summary ".center(50, "="))
    logging.info("="*50)
    
    logging.info("\nIndividual Fold Results:")
    logging.info(results_df.round(4).to_string())
    
    logging.info("\nAggregated Statistics (Mean ± Std Dev):")
    
    # 格式化输出
    for metric in summary.columns:
        mean_val = summary.loc['mean', metric]
        std_val = summary.loc['std', metric]
        logging.info(f"  - {metric:<15}: {mean_val:.4f} ± {std_val:.4f}")
        
    logging.info("\n" + "="*50)

    # 将结果保存到文件
    summary_path = os.path.join(config.OUTPUT_DIR, "final_summary.csv")
    results_df.to_csv(summary_path, index=False, float_format='%.4f')
    logging.info(f"Detailed results saved to {summary_path}")

import os
from collections import defaultdict
from typing import Dict, List, Union

def group_slice_paths_by_pid(
    image_dirs: List[str],
    pids: Union[List[str], None] = None,
    ext: str = ".png",
) -> Dict[str, List[str]]:
    """
    将各目录中的切片按“文件名前缀 (= 病人 ID)”归组并
    依据下划线后的数字序号排序。

    参数
    ----
    image_dirs : List[str]
        图片所在的所有文件夹，如 [".../train2017", ".../val2017"]
    pids : List[str] | None
        若给定，则仅保留这些 pid；传 None 表示全部收集
    ext : str
        图片后缀，默认 ".png"

    返回
    ----
    dict : {pid: [slice_path0, slice_path1, ...]}（slice 号升序）
    """
    pid_whitelist = set(pids) if pids is not None else None
    groups: Dict[str, List[tuple]] = defaultdict(list)

    for folder in image_dirs:
        for fname in os.listdir(folder):
            if not fname.endswith(ext):
                continue

            name_wo_ext = os.path.splitext(fname)[0]         # "1850473_230"
            #print(name_wo_ext)
            parts = name_wo_ext.split('_', 1)
            #print(parts)
            if len(parts) < 2:
                continue                                     # 没有下划线，跳过
            pid, idx_str = parts
            pid=pid+'.nii'
            if pid_whitelist is not None and pid not in pid_whitelist:
                continue

            try:
                idx = int(idx_str.split('_')[0])            # 取第 1 段做 slice 序号
            except ValueError:
                idx = -1                                     # 若解析失败，放到最前
            # print(3)
            groups[pid].append((idx, os.path.join(folder, fname)))
            #print(groups[pid])

    # 对每个病人按 idx 排序，只保留路径
    return {
        pid: [path for _, path in sorted(items, key=lambda t: t[0])]
        for pid, items in groups.items()
    }



# group_slice_paths_by_pid(config.IMAGE_DIRS,['1852423.nii'])