# utils.py

import os
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
def group_slice_paths_by_pid(image_dirs: List[str], pids: List[str]) -> Dict[str, List[str]]:
    """
    将图片按病人ID分组：
      输入 pids 形如 ['1859176.nii', '1860366.nii']，
      文件名形如 '1859176_012.png'，前缀与 pid 去掉扩展名一致。
    返回 {pid: [slice_path_按序排序,...]}，pid 使用传入的原始键（含 .nii 或 .nii.gz）。
    """
    # 建立 prefix -> pid 的映射（pid 保持传入的原始字符串作为最终 key）
    prefix_to_pid: Dict[str, str] = {}
    for pid in pids:
        prefix = pid.split('.')[0]
        if prefix in prefix_to_pid and prefix_to_pid[prefix] != pid:
            logging.warning(f"Duplicate prefix found for {prefix}: {prefix_to_pid[prefix]} vs {pid}")
        prefix_to_pid[prefix] = pid

    # 收集
    groups: Dict[str, List[str]] = {pid: [] for pid in pids}
    allowed_exts = {".png", ".jpg", ".jpeg"}  # 如只用 png 可改为 {".png"}

    for d in image_dirs:
        if not os.path.isdir(d):
            logging.warning(f"Image directory not found: {d}. Skipping.")
            continue
        with os.scandir(d) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in allowed_exts:
                    continue
                file_prefix = entry.name.split('_')[0]
                if file_prefix in prefix_to_pid:
                    pid_key = prefix_to_pid[file_prefix]
                    groups[pid_key].append(entry.path)

    # 排序（按下划线后的数字；失败则回退到字典序）
    def _slice_idx(path: str) -> int:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        parts = name.split('_')
        if len(parts) > 1:
            tok = parts[1]
            digits = ''.join(ch for ch in tok if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    pass
        # 回退：提取整串里的数字，或 0
        digits = ''.join(ch for ch in name if ch.isdigit())
        try:
            return int(digits) if digits else 0
        except ValueError:
            return 0

    for pid, files in groups.items():
        if not files:
            logging.warning(f"No slices found for {pid}.")
            continue
        files = sorted(files, key=_slice_idx)
        # 去重（保持顺序）
        seen = set()
        dedup = []
        for f in files:
            if f not in seen:
                dedup.append(f); seen.add(f)
        groups[pid] = dedup

    return groups
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